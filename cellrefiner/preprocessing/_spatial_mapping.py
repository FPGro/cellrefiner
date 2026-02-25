import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial import distance_matrix
import scanpy as sc
from anndata import AnnData
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import scipy
import ot
import torch
import gudhi
import networkx as nx
from torch_geometric.nn import DeepGraphInfomax, GCNConv
import torch.nn as nn
from .._utils import gen_w, pre_cal1, sparsify, cal_glvs, glvs, estimate_scale
from .._utils import compute_correlation_matrix_cpu, compute_correlation_matrix_gpu
from .._utils import H_matrix_vectorized_cpu, H_matrix_vectorized_gpu
from .._utils import F_spot_optimized_cpu, F_spot_optimized_gpu
from .._utils import V_xy_vectorized_gpu, V_xy_vectorized_cpu
from .._utils import F_gc_vectorized_gpu, F_gc_vectorized_cpu
from typing import Optional, Union, List, Iterable, Tuple
import warnings

def spatial_mapping(
        ad_st: AnnData,
        ad_sc: AnnData,
        db: DataFrame,
        scale: Optional[float] = None,
        cluster_key_sc: Optional[str] = None,
        spatial_key: str = 'spatial',
        pca_key: str = 'X_pca',
        uns_key: str = 'rank_genes_groups',
        n_rank_gene: Optional[int] = 100,
        n_cell: int = 5,
        device: str = 'cuda:0',
        enable_cupy: bool = True,
        enable_lr_force = False,
        return_mapping = False,
        seed: int = 0
    ) -> AnnData:
    """
    Perform mapping of single-cell data to spatial transcriptomics data and spatial refinement.

    Parameters
    ----------
    ad_st : AnnData
        Spatial transcriptomics AnnData object.
        
        Must contain spatial coordinates in `.obsm[spatial_key]`.
    ad_sc : AnnData
        Single-cell RNA-seq AnnData object.
    db : DataFrame
        Ligand-receptor interaction database.
    scale : float
        Spatial scale parameter that determines the interaction distance, representing the size of spatial transcriptomics spot.
    cluster_key_sc : str, Optional
        Key in `ad_sc.obs` that contains cell type annotations, used for `scanpy.tl.rank_genes_groups(ad_sc, groupby=cluster_key_sc)`
    spatial_key : str, default 'spatial'
        Key in `ad_st.obsm` that contains spatial coordinates
    pca_key : str, default 'X_pca'
        Key in `ad_sc.obsm` that contains PCA embeddings.

        If not in `ad_sc.obsm`, `scanpy.pp.pca()` will be computed.
    uns_key : str, default 'rank_genes_groups'
        Key in `ad_sc.uns` containing ranked genes results from `scanpy.tl.rank_genes_groups()`

        If not present, `scanpy.tl.rank_genes_groups(ad_sc, groupby=cluster_key_sc)` will be computed.
    n_rank_gene : int or None, default 100
        Number of top-ranked genes for each cell type that will be used in spatial mapping.

        If None, all genes will be used.
    n_cell : int, default 5
        Number of cells to map to each spatial location.
    device : str, default 'cuda:0'
        Device used by pytorch.
    enable_lr_force : bool, default False
        Whether to enable ligand-receptor force.
    enable_cupy : bool, default True
        Whether to enable CuPy. If CuPy is not available, will automatically fall back to CPU.
    seed : int, default 0
        random seed

    Returns
    -------
    AnnData
        AnnData object containing mapped cells with refined spatial coordinates.
        
        `.obsm['spatial']`: Refined spatial coordinates. If `.obsm['spatial']` is present in ad_sc, 
        then stored as `.obsm['spatial_refined']`.

        Same gene expression data as input single-cell RNA-seq data

    Examples
    --------
    >>> adata_cr = spatial_mapping(adata_st,adata_sc,db_lr,scale=125,cluster_key_sc = 'cell_type')
    """

    np.random.seed(seed)
    ad_sc0 = ad_sc.copy()
    if n_rank_gene is not None:
        if uns_key not in ad_sc.uns:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sc.tl.rank_genes_groups(ad_sc, groupby=cluster_key_sc, use_raw=False)
        markers_df = pd.DataFrame(ad_sc.uns[uns_key]['names'])
        markers_df = markers_df.iloc[:n_rank_gene, :]
        markers = list(np.unique(markers_df.melt().value.values))
        ad_sc = ad_sc[:, ad_sc.var_names.isin(markers)].copy()
    # mapping
    x_coord = ad_st.obsm[spatial_key]
    M, log = map_fgw(ad_st, ad_sc, x_coord, seed, device)
    # refine
    W = gen_w(ad_sc, db)
    x_range = np.abs(np.max(x_coord[:, 0]) - np.min(x_coord[:, 0]))
    # parameters
    if scale is None:
        scale = estimate_scale(x_coord)
    m_val = scale/x_range*5000
    U0 = 0.1 / (2.85 / m_val)
    V0 = 1.1 / (2.85 / m_val)
    xi1 = 1.21 / (2.85 / m_val)
    xi2 = 1.9 / (2.85 / m_val)
    iterations = 10
    dt = 20
    xsr = scale/x_range*5000
    x_r = scale/x_range*5000
    z_cutoff = 0.4  # level set cutoff for defining tissue boundary

    x_coord = x_coord / x_range * 5000
    a = np.tile(x_coord[:, 0], (n_cell, 1)).T.flatten()
    b = np.tile(x_coord[:, 1], (n_cell, 1)).T.flatten()
    xs = np.concatenate(([a], [b]), axis=0).T
    xc = xs + np.random.normal(0, xsr, size=xs.shape)

    # Neighbor computation (keep on CPU for now as it's a one-time operation)
    # neigh = NearestNeighbors(n_neighbors=5)
    # neigh.fit(xc)
    # x_id = neigh.kneighbors(xs)  # first entry is distance, second is indices
    x_id1 = []  # list of boolean arrays for neighboring spots
    for i in range(xs.shape[0]):
        x_id1.append(np.linalg.norm(xs - xs[i, :], axis=1) < x_r)

    # create spot by cell index matrix for the top n cells
    cell5 = np.zeros((M.shape[1], n_cell))
    gmap1 = M.copy()
    for i in range(gmap1.shape[1]):
        cell5[i, :] = np.argpartition(gmap1[:, i], -n_cell)[-n_cell:]
        gmap1[cell5[i, :].astype(int), :] = 0

    cell5m = cell5.flatten().astype(int)
    # cell_codes = pd.Categorical(adata_sc.obs['Cell_type']).codes[cell5m]

    W1 = W[cell5m, :]
    W1 = W1[:, cell5m]
    W1 = W1 / np.max(W1)

    degree = np.diag(np.sum(W1, axis=1))
    L = degree - W1

    # Fix determinant computation warning
    if enable_lr_force:
        try:
            det_L = np.linalg.det(L)
            # Check for valid determinant (not NaN or inf)
            if np.isfinite(det_L) and det_L > 1e-10:
                q = pre_cal1(W1)
                H = sparsify(W1, q)
            else:
                print("Warning: Laplacian matrix is singular or ill-conditioned, using zero matrix")
                H = np.zeros(np.shape(W1)) 
        except np.linalg.LinAlgError:
            print("Warning: Determinant computation failed, using zero matrix")
            H = np.zeros(np.shape(W1))
    else:
        H = np.zeros(np.shape(W1))
    adata_cr = ad_sc0[cell5m, :].copy()

    if pca_key not in adata_cr.obsm:
        sc.pp.pca(adata_cr)
        pca_key = 'X_pca'
    X_sc2m2 = adata_cr.obsm[pca_key]

    # check gpu avaliablity
    try:
        import cupy as cp
        enable_cupy = cp.cuda.runtime.getDeviceCount() > 0
        print("GPU acceleration available with CuPy")
    except ImportError:
        enable_cupy = False
        print("CuPy not available")

    if enable_cupy:
        final_positions = spatial_refine_gpu(xs, xc, X_sc2m2, x_id1, H, z_cutoff, x_r, V0, U0, xi1, xi2, dt, iterations)
    else:
        final_positions = spatial_refine_cpu(xs, xc, X_sc2m2, x_id1, H, z_cutoff, x_r, V0, U0, xi1, xi2, dt, iterations)

    if 'spatial' in adata_cr.obsm:
        adata_cr.obsm['spatial_cr'] = final_positions * x_range / 5000
        print(".obsm['spatial'] exist. Add refined spatial coordinates to .obsm['spatial_cr']")
    else:
        adata_cr.obsm['spatial'] = final_positions * x_range / 5000

    adata_cr.uns['spatial_mapping'] = dict(scale=scale,n_cell=n_cell,n_rank_gene=n_rank_gene)
    adata_cr.uns['OT_log'] = log
    if return_mapping:
        return adata_cr, cell5, M
    else:
        return adata_cr

def spatial_refine_cpu(xs, xc, X_sc2m2, x_id1, H, z_cutoff, x_r, V0, U0, xi1, xi2, dt, iterations):

    neighbor_indices = []
    for i, neighbors in enumerate(x_id1):
        neighbor_indices.append(np.where(neighbors)[0])

    correlation_matrix_cpu = compute_correlation_matrix_cpu(X_sc2m2)

    # Initialize position arrays
    pos_s_cpu = np.tile(xs, [iterations + 1, 1, 1])
    pos_cpu = np.tile(xc, [iterations + 1, 1, 1])
    F_gc_const_cpu = np.asarray(
        (np.linspace(1, 0, iterations) ** 2), dtype=np.float32)

    # Convert Sigma for tissue boundary calculations
    z_val = z_cutoff * np.amax(cal_glvs(pos_cpu[0, :, :]))
    Sigma = np.array([[10000, 0], [0, 10000]])
    # Main simulation loop
    for i in range(iterations):
        current_positions = pos_cpu[i, :, :].copy()
        # Spot forces
        spot_forces = F_spot_optimized_cpu(
            current_positions, pos_s_cpu[i, :, :], x_r)
        current_positions += spot_forces
        # Neighbor and gene forces
        n_cells = current_positions.shape[0]
        all_i_indices = []
        all_j_indices = []
        for j in range(n_cells):
            neighbors = neighbor_indices[j]
            valid_neighbors = neighbors[neighbors != j]
            if len(valid_neighbors) > 0:
                all_i_indices.extend([j] * len(valid_neighbors))
                all_j_indices.extend(valid_neighbors.tolist())

        if len(all_i_indices) > 0:
            i_idx = np.asarray(all_i_indices)
            j_idx = np.asarray(all_j_indices)
            mask = np.ones(len(all_i_indices), dtype=bool)

            # Get positions, correlations, and H weights
            pos_i = current_positions[i_idx]
            pos_j = current_positions[j_idx]
            correlations = correlation_matrix_cpu[i_idx, j_idx]
            h_weights = H[i_idx, j_idx]

            # Compute forces
            spatial_forces = V_xy_vectorized_cpu(pos_j, pos_i, V0, U0, xi1, xi2, mask)
            gene_forces = F_gc_vectorized_cpu(pos_j, pos_i, correlations, mask)
            h_matrix_forces = H_matrix_vectorized_cpu(pos_j, pos_i, h_weights, mask)

            # Apply force updates
            force_updates = np.zeros_like(current_positions)
            np.add.at(force_updates, i_idx, -dt * spatial_forces)
            np.add.at(force_updates, i_idx, F_gc_const_cpu[i] * gene_forces)
            np.add.at(force_updates, i_idx,
                      F_gc_const_cpu[i] * h_matrix_forces)

            current_positions += force_updates

        # Enforce tissue boundary
        pos_cpu_temp = current_positions.copy()
        z2 = np.zeros(pos_cpu_temp.shape[0])
        for j in range(pos_cpu_temp.shape[0]):
            z2[j] = glvs(pos_cpu_temp[j:j+1, :], pos_cpu[0, j, :], Sigma)

        z_ind = z2 < z_val
        if np.any(z_ind):
            pos_cpu_temp[z_ind, :] = pos_cpu[i, z_ind, :] + 0.1 * (pos_cpu_temp[z_ind, :] - pos_cpu[i, z_ind, :])
            current_positions = pos_cpu_temp
        
        pos_cpu[i + 1, :, :] = current_positions
        return pos_cpu[-1, :, :]

def spatial_refine_gpu(xs, xc, X_sc2m2, x_id1, H, z_cutoff, x_r, V0, U0, xi1, xi2, dt, iterations):

    import cupy as cp

    # Convert data to GPU arrays
    xs_gpu = cp.asarray(xs, dtype=cp.float32)
    xc_gpu = cp.asarray(xc, dtype=cp.float32)
    X_sc2m2_gpu = cp.asarray(X_sc2m2, dtype=cp.float32)
    H_gpu = cp.asarray(H, dtype=cp.float32)

    # Convert neighbor lists to GPU format
    neighbor_indices = []
    for i, neighbors in enumerate(x_id1):
        neighbor_indices.append(cp.asarray(np.where(neighbors)[0]))

    correlation_matrix_gpu = compute_correlation_matrix_gpu(X_sc2m2_gpu)

    # Initialize position arrays
    pos_s_gpu = cp.tile(xs_gpu, [iterations + 1, 1, 1])
    pos_gpu = cp.tile(xc_gpu, [iterations + 1, 1, 1])
    pos_cpu = cp.asnumpy(pos_gpu)
    F_gc_const_gpu = cp.asarray(
        (np.linspace(1, 0, iterations) ** 2), dtype=cp.float32)

    # Convert Sigma for tissue boundary calculations
    z_val = z_cutoff * np.amax(cal_glvs(pos_cpu[0, :, :]))
    Sigma = np.array([[10000, 0], [0, 10000]])
    # Main simulation loop
    for i in range(iterations):
        current_positions = pos_gpu[i, :, :].copy()
        # Spot forces
        spot_forces = F_spot_optimized_gpu(
            current_positions, pos_s_gpu[i, :, :], x_r)
        current_positions += spot_forces
        # Neighbor and gene forces
        n_cells = current_positions.shape[0]
        all_i_indices = []
        all_j_indices = []
        for j in range(n_cells):
            neighbors = neighbor_indices[j]
            valid_neighbors = neighbors[neighbors != j]
            if len(valid_neighbors) > 0:
                all_i_indices.extend([j] * len(valid_neighbors))
                all_j_indices.extend(valid_neighbors.tolist())

        if len(all_i_indices) > 0:
            i_idx = cp.asarray(all_i_indices)
            j_idx = cp.asarray(all_j_indices)
            mask = cp.ones(len(all_i_indices), dtype=bool)

            # Get positions, correlations, and H weights
            pos_i = current_positions[i_idx]
            pos_j = current_positions[j_idx]
            correlations = correlation_matrix_gpu[i_idx, j_idx]
            h_weights = H_gpu[i_idx, j_idx]

            # Compute forces
            spatial_forces = V_xy_vectorized_gpu(pos_j, pos_i, V0, U0, xi1, xi2, mask)
            gene_forces = F_gc_vectorized_gpu(pos_j, pos_i, correlations, mask)
            h_matrix_forces = H_matrix_vectorized_gpu(
                pos_j, pos_i, h_weights, mask)

            # Apply force updates
            force_updates = cp.zeros_like(current_positions)
            cp.add.at(force_updates, i_idx, -dt * spatial_forces)
            cp.add.at(force_updates, i_idx, F_gc_const_gpu[i] * gene_forces)
            cp.add.at(force_updates, i_idx, F_gc_const_gpu[i] * h_matrix_forces)

            current_positions += force_updates

        # Enforce tissue boundary
        pos_cpu_temp = cp.asnumpy(current_positions)
        z2 = np.zeros(pos_cpu_temp.shape[0])
        for j in range(pos_cpu_temp.shape[0]):
            z2[j] = glvs(pos_cpu_temp[j:j+1, :], pos_cpu[0, j, :], Sigma)

        z_ind = z2 < z_val
        if np.any(z_ind):
            pos_cpu_temp[z_ind, :] = pos_cpu[i, z_ind, :] + 0.1 * (pos_cpu_temp[z_ind, :] - pos_cpu[i, z_ind, :])
            current_positions = cp.asarray(pos_cpu_temp)

        pos_gpu[i + 1, :, :] = current_positions
        pos_cpu = cp.asnumpy(pos_gpu)

        # Periodic memory cleanup for GPU
        if (i + 1) % 3 == 0:
            cp.get_default_memory_pool().free_all_blocks()

    return cp.asnumpy(pos_gpu[-1, :, :])


def map_fgw(ad_st: AnnData, ad_sc: AnnData, st_location, seed:int, device: str):

    torch.manual_seed(seed)
    shared_genes = list(set(ad_st.var_names).intersection(set(ad_sc.var_names)))
    ad_st = ad_st[:, shared_genes].copy()
    ad_sc = ad_sc[:, shared_genes].copy()

    # Extract expression matrices
    sc_expr = ad_sc.X
    st_expr = ad_st.X

    # Convert sparse matrices to dense if needed
    if scipy.sparse.issparse(sc_expr):
        sc_expr = sc_expr.toarray()
    if scipy.sparse.issparse(st_expr):
        st_expr = st_expr.toarray()
    
    spatial_regularization_strength = 0.1
    z_dim = 50
    lr = 1e-3  # learning rate for spaceflow
    epochs = 1000
    max_patience = 50
    min_stop = 100

    # SpaceFlow graph generation
    spatial_graph = graph_alpha(st_location)

    # generating model for spaceflow embedding
    model = DeepGraphInfomax(
        hidden_channels=z_dim, encoder=GraphEncoder(ad_st.shape[1], z_dim),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)

    expr = torch.tensor(st_expr).float().to(device)

    edge_list = sparse_mx_to_torch_edge_list(spatial_graph).to(device)

    model.train()
    min_loss = np.inf
    patience = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_params = model.state_dict()

    for epoch in range(epochs):
        train_loss = 0.0
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
        z, neg_z, summary = model(expr, edge_list)
        loss = model.loss(z, neg_z, summary)

        coords = torch.tensor(st_location, dtype=torch.float32).to(device)

        z_dists = torch.cdist(z, z, p=2)
        z_dists = torch.div(z_dists, torch.max(z_dists)).to(device)
        sp_dists = torch.cdist(coords, coords, p=2)
        sp_dists = torch.div(sp_dists, torch.max(sp_dists)).to(device)
        n_items = z.size(dim=0) * z.size(dim=0)

        penalty_1 = torch.div(
            torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
        loss = loss + spatial_regularization_strength * penalty_1

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if train_loss > min_loss:
            patience += 1
        else:
            patience = 0
            min_loss = train_loss
            best_params = model.state_dict()
        if patience > max_patience and epoch > min_stop:
            break

    model.load_state_dict(best_params)
    z, _, _ = model(expr, edge_list)
    embedding = z.cpu().detach().numpy()

    # spatial cost matrix using spaceflow embedding
    A1d = cosine_similarity(embedding)
    A = np.multiply(A1d, distance_matrix(st_location, st_location))
    A /= A.max()
    M, log = mapper(sc_expr, st_expr, A)  # run mapping
    return M, log # numpy matrix output (cell by spot)


def graph_alpha(spatial_locs, n_neighbors=10):
    """
    Construct a geometry-aware spatial proximity graph of the spatial spots of cells by using alpha complex.
    :param adata: the annData object for spatial transcriptomics data with adata.obsm['spatial'] set to be the spatial locations.
    :type adata: class:`anndata.annData`
    :param n_neighbors: the number of nearest neighbors for building spatial neighbor graph based on Alpha Complex
    :type n_neighbors: int, optional, default: 10
    :return: a spatial neighbor graph
    :rtype: class:`scipy.sparse.csr_matrix`
    """
    A_knn = kneighbors_graph(
        spatial_locs, n_neighbors=n_neighbors, mode='distance')
    estimated_graph_cut = A_knn.sum() / float(A_knn.count_nonzero())
    spatial_locs_list = spatial_locs.tolist()
    n_node = len(spatial_locs_list)
    alpha_complex = gudhi.AlphaComplex(points=spatial_locs_list)
    simplex_tree = alpha_complex.create_simplex_tree(
        max_alpha_square=estimated_graph_cut ** 2)
    skeleton = simplex_tree.get_skeleton(1)
    initial_graph = nx.Graph()
    initial_graph.add_nodes_from([i for i in range(n_node)])
    for s in skeleton:
        if len(s[0]) == 2:
            initial_graph.add_edge(s[0][0], s[0][1])

    extended_graph = nx.Graph()
    extended_graph.add_nodes_from(initial_graph)
    extended_graph.add_edges_from(initial_graph.edges)

    for i in range(n_node):
        try:
            extended_graph.remove_edge(i, i)
        except:
            pass

    return nx.to_scipy_sparse_array(extended_graph, format='csr')


class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


def sparse_mx_to_torch_edge_list(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edge_list = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return edge_list


def cosine_similarity(X_sf):

    # Normalize columns
    X_sf_normalized = X_sf.T / np.linalg.norm(X_sf.T, axis=0, keepdims=True)

    # Dot product
    similarity_matrix = np.matmul(X_sf_normalized.T, X_sf_normalized)

    return similarity_matrix


def mapper(sc_expr, st_expr, A):
    """
    Assign cells from single-cell data to spots in spatial data using Fused Gromov-Wasserstein,
    allowing multiple cells per spot (up to max_cells_per_spot).

    Parameters:
    -----------
    ad_sc : AnnData
            Single-cell RNA-seq data
    ad_st : AnnData
            Spatial transcriptomics data
    A : numpy.ndarray
            Cost matrix corresponding to the spatial data
    max_cells_per_spot : int
            Maximum number of cells that can be assigned to each spot

    Returns:
    --------
    numpy.ndarray
            Array of spot indices for each cell. Length equals number of cells that were assigned.
    """

    # Normalize
    sc_expr_norm = sc_expr / \
        np.sqrt(np.sum(sc_expr**2, axis=1, keepdims=True) + 1e-10)
    st_expr_norm = st_expr / \
        np.sqrt(np.sum(st_expr**2, axis=1, keepdims=True) + 1e-10)

    # Cosine similarity matrix
    similarity_matrix = np.dot(sc_expr_norm, st_expr_norm.T)

    # Convert to distance/cost matrix (1 - similarity) for FGW
    M = 1 - similarity_matrix

    n_cells = sc_expr.shape[0]
    n_spots = st_expr.shape[0]

    # Single cell cost matrix (cosine similarity)
    C1 = 1-np.dot(sc_expr_norm, sc_expr_norm.T)
    C1 /= C1.max()

    C2 = A

    # Create uniform distributions
    a = np.ones(n_cells) / n_cells  # source distribution
    b = np.ones(n_spots) / n_spots  # target distribution

    # Solve Fused Gromov-Wasserstein
    T, log = ot.gromov.fused_gromov_wasserstein(
        M, C1, C2, a, b,
        loss_fun='square_loss',
        alpha=0.5,  # Balance between structure and feature matching
        armijo=False,
        log=True,
        max_iter=1000000
    )
    log['M_info'] = {'shape':M.shape,'max':M.max(),'min':M.min()}
    log['C1_info'] = {'shape':C1.shape,'max':C1.max(),'min':C1.min()}
    log['C2_info2'] = {'shape':C2.shape,'max':C2.max(),'min':C2.min()}
    return T, log