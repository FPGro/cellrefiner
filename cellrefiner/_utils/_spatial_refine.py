import numpy as np
from scipy.sparse import issparse, lil_matrix
from scipy.spatial import Delaunay, distance

try:
    import cupy as cp
except ImportError:
    cp = None

def pre_cal1(N):
    degree = np.diag(np.sum(N, axis=1))
    L = degree-N
    L_inv = np.linalg.pinv(L, rcond=np.finfo(float).tiny)
    n = len(N)
    q = np.multiply(N, np.diag(L_inv) + np.reshape(np.diag(L_inv.T), (-1, 1))-L_inv.T-L_inv)
    return q


def sparsify(W1, q):
    n = len(W1)
    P = np.minimum(1, np.log(n)*q)
    H = np.ones((n, n))*np.finfo(float).tiny
    rand = np.random.rand(n, n)
    idx = np.where(rand < P)
    H[idx] = W1[idx]/P[idx]
    kept = len(H[H > np.finfo(float).tiny])/n**2
    percent = .40
    time = 0
    while kept < percent:
        if time > 300:
            break
        rand = np.random.rand(n, n)
        idx = np.where(rand < P)
        H[idx] = W1[idx]/P[idx]
        kept = len(H[H > np.finfo(float).tiny])/n**2
        time += 1
        print('kept is:', kept)

    H = H/np.amax(H)
    return H


def glvs(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


def cal_glvs(pos):
    # input should be a nx2 matrix of coordinates
    # distance from tissue edge to compute points
    x_offset = 0.1*(np.amax(pos[:, 0])-np.amin(pos[:, 0]))
    y_offset = 0.1*(np.amax(pos[:, 1])-np.amin(pos[:, 1]))
    x, y = np.meshgrid(np.linspace(np.amin(pos[:, 0])-x_offset, np.amax(pos[:, 0])+x_offset, 100),
                       np.linspace(np.amin(pos[:, 1])-y_offset, np.amax(pos[:, 1])+y_offset, 100))
    Sigma = np.array([[10000, 0], [0,  10000]])
    pts = np.empty(x.shape + (2,))
    pts[:, :, 0] = x
    pts[:, :, 1] = y
    z = np.zeros((pts.shape[0], pts.shape[1]))

    # Fix overflow warning by using more stable computation
    for i in range(pos.shape[0]):
        gaussian_vals = glvs(pts, pos[i, :], Sigma)
        # Clip extreme values to prevent overflow
        gaussian_vals = np.clip(gaussian_vals, 0, 1e10)
        z = z + gaussian_vals

    return z


def gen_w(adata_sc, db):
    """
    Generate LR affinity matrix
    """

    a = adata_sc.X.toarray() if issparse(adata_sc.X) else adata_sc.X
    
    # ligand expression matrix
    tl = np.zeros((a.shape[0], len(db['interaction_name'])))
    # receptor expression matrix
    tr = np.zeros((a.shape[0], len(db['interaction_name'])))
    for i in range(len(db['interaction_name'])):
        int_name = db['interaction_name'][i].split('_')  # interaction
        lig = int_name[0]  # ligand
        rec = int_name[1:]  # receptor/s
        lig_ind = adata_sc.var_names == lig  # ligand indices as boolean array
        if sum(lig_ind) > 0:
            tl[:, i] = a[:, lig_ind].flatten()
            check = 0
            for j in range(len(rec)):  # see if all receptors are present
                if sum(adata_sc.var_names == rec[j]) > 0:
                    check += 1

            if check == len(rec):
                rec_ct = a[:, adata_sc.var_names == rec[0]]
                for j in range(len(rec)):
                    rec_ct = np.minimum(
                        rec_ct, a[:, adata_sc.var_names == rec[j]])

                tr[:, i] = rec_ct.flatten()

    # calculate cell by cell affinity matrix
    W = np.add(np.matmul(tl, tr.T), np.matmul(
        tr, tl.T))  # T_L * T_R' + T_R * T_L'

    # Fix division by zero warning
    max_W = np.amax(W)
    if max_W > 0:
        W = np.divide(W, max_W)
    else:
        print("Warning: W matrix is all zeros, using identity matrix")
        W = np.eye(W.shape[0]) * 1e-6

    return W

def compute_correlation_matrix_gpu(X_data):
    """Compute correlation matrix with robust error handling"""
    try:
        # Use CuPy for GPU computation
        corr_matrix = cp.corrcoef(X_data)
        # Handle edge cases
        if corr_matrix.ndim == 0:
            corr_matrix = cp.array([[1.0]])
        corr_matrix = cp.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        corr_matrix = cp.maximum(corr_matrix, 0)  # Remove negative correlations
        return corr_matrix
    except Exception as e:
        print(f"Warning: Correlation computation failed: {e}")
        n_cells = X_data.shape[0]
        return cp.eye(n_cells, dtype=cp.float32)
        
def compute_correlation_matrix_cpu(X_data):
    """Compute correlation matrix with robust error handling"""
    try:
        # Use NumPy for CPU computation
        corr_matrix = np.corrcoef(X_data)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        corr_matrix = np.maximum(corr_matrix, 0)  # Remove negative correlations
        return corr_matrix
    except Exception as e:
        print(f"Warning: Correlation computation failed: {e}")
        n_cells = X_data.shape[0]
        return np.eye(n_cells, dtype=np.float32)
    
def H_matrix_vectorized_gpu(positions_i, positions_j, h_weights, mask):
    """Vectorized H matrix force computation"""
    diff = positions_j - positions_i
    norms = cp.linalg.norm(diff, axis=1)
    valid_mask = (norms > 1e-8) & (h_weights > 0) & mask
    forces = cp.zeros_like(diff)
    
    if cp.any(valid_mask):
        forces[valid_mask] = (h_weights[valid_mask, cp.newaxis] * diff[valid_mask] / norms[valid_mask, cp.newaxis])
    
    return forces

def H_matrix_vectorized_cpu(positions_i, positions_j, h_weights, mask):
    """Vectorized H matrix force computation"""
    diff = positions_j - positions_i
    norms = np.linalg.norm(diff, axis=1)
    valid_mask = (norms > 1e-8) & (h_weights > 0) & mask
    forces = np.zeros_like(diff)
    
    if np.any(valid_mask):
        forces[valid_mask] = (h_weights[valid_mask, np.newaxis] * diff[valid_mask] / norms[valid_mask, np.newaxis])

    return forces

def F_spot_optimized_gpu(cell_positions, spot_positions, rS):
    """Optimized spot force computation"""
    n_cells = cell_positions.shape[0]
    # Vectorized distance computation
    cell_expanded = cell_positions[:, cp.newaxis, :]
    spot_expanded = spot_positions[cp.newaxis, :, :]
    diff = spot_expanded - cell_expanded
    distances = cp.linalg.norm(diff, axis=2)
    
    # Find closest spot for each cell
    closest_indices = cp.argmin(distances, axis=1)
    closest_distances = distances[cp.arange(n_cells), closest_indices]
    
    # Compute forces
    forces = cp.zeros((n_cells, 2))
    valid_mask = closest_distances >= rS
    
    if cp.any(valid_mask):
        valid_cells = cp.where(valid_mask)[0]
        closest_spots = closest_indices[valid_cells]
        
        force_diff = (spot_positions[closest_spots] - 
                        cell_positions[valid_cells])
        force_distances = closest_distances[valid_cells]
        
        force_magnitudes = cp.minimum((force_distances - rS)**2, 30)
        forces[valid_cells] = (force_magnitudes[:, cp.newaxis] * force_diff / force_distances[:, cp.newaxis])
    
    return forces

def F_spot_optimized_cpu(cell_positions, spot_positions, rS):
    """Optimized spot force computation"""
    n_cells = cell_positions.shape[0]

    forces = np.zeros((n_cells, 2))
    for i in range(n_cells):
        diff = spot_positions - cell_positions[i:i+1, :]
        distances = np.linalg.norm(diff, axis=1)
        closest_idx = np.argmin(distances)
        closest_dist = distances[closest_idx]
        
        if closest_dist >= rS:
            force_magnitude = min((closest_dist - rS)**2, 30)
            force_direction = diff[closest_idx] / closest_dist
            forces[i] = force_magnitude * force_direction

    return forces

def V_xy_vectorized_gpu(positions_i, positions_j, V0, U0, xi1, xi2, mask):
    """Vectorized spatial force computation"""
    diff = positions_j - positions_i
    r2 = cp.sum(diff**2, axis=1)
    r = cp.sqrt(r2)
    valid_mask = (r > 1e-8) & mask
    forces = cp.zeros_like(diff)

    if cp.any(valid_mask):
        r_valid = r[valid_mask]
        r2_valid = r2[valid_mask]
        diff_valid = diff[valid_mask]
        xi1_sq = xi1 * xi1
        xi2_sq = xi2 * xi2
        exp1 = cp.exp(-r2_valid / xi1_sq)
        exp2 = cp.exp(-r2_valid / xi2_sq)
        dVdr = (-2 * r_valid * V0 / xi1_sq * exp1 + 2 * r_valid / xi2_sq * U0 * exp2)
        forces[valid_mask] = (dVdr[:, cp.newaxis] * diff_valid / r_valid[:, cp.newaxis])

    return forces

def V_xy_vectorized_cpu(positions_i, positions_j, V0, U0, xi1, xi2, mask):
    """Vectorized spatial force computation"""
    diff = positions_j - positions_i
    r2 = np.sum(diff**2, axis=1)
    r = np.sqrt(r2)
    valid_mask = (r > 1e-8) & mask
    forces = np.zeros_like(diff)
        
    if np.any(valid_mask):
        r_valid = r[valid_mask]
        r2_valid = r2[valid_mask]
        diff_valid = diff[valid_mask]
        xi1_sq = xi1 * xi1
        xi2_sq = xi2 * xi2
        exp1 = np.exp(-r2_valid / xi1_sq)
        exp2 = np.exp(-r2_valid / xi2_sq)
        dVdr = (-2 * r_valid * V0 / xi1_sq * exp1 + 2 * r_valid / xi2_sq * U0 * exp2)
        forces[valid_mask] = (dVdr[:, np.newaxis] * diff_valid / r_valid[:, np.newaxis])
        
    return forces

def F_gc_vectorized_gpu(positions_i, positions_j, correlations, mask):
    """Vectorized gene force computation"""
    # Compute differences for all valid pairs
    diff = positions_j - positions_i
    norms = cp.linalg.norm(diff, axis=1)
    valid_mask = (norms > 1e-8) & (correlations > 0) & mask
    forces = cp.zeros_like(diff)
    if cp.any(valid_mask):
        forces[valid_mask] = (correlations[valid_mask, cp.newaxis] * diff[valid_mask] / norms[valid_mask, cp.newaxis])
    
    return forces

def F_gc_vectorized_cpu(positions_i, positions_j, correlations, mask):
    """Vectorized gene force computation"""
    # Compute differences for all valid pairs
    diff = positions_j - positions_i
    norms = np.linalg.norm(diff, axis=1)
    valid_mask = (norms > 1e-8) & (correlations > 0) & mask
    forces = np.zeros_like(diff)
    if np.any(valid_mask):
        forces[valid_mask] = (correlations[valid_mask, np.newaxis] * diff[valid_mask] / norms[valid_mask, np.newaxis])
    
    return forces

def estimate_scale(xc: np.ndarray):
    nc = xc.shape[0]
    distance_matrix = lil_matrix((nc, nc))
    tri = Delaunay(xc)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                d = distance.euclidean(xc[simplex[i]],xc[simplex[j]])
                distance_matrix[simplex[i], simplex[j]] = d
                distance_matrix[simplex[j], simplex[i]] = d

    dc = np.zeros(nc)
    for cid in range(nc):
        _,j=distance_matrix[cid].nonzero()
        dc[cid] = np.mean(distance_matrix[cid,j]) if len(j)>0 else np.nan # some points might overlap with others
    
    dc = dc[~np.isnan(dc)]
    return np.median(dc)/2