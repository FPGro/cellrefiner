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
from .._utils import gen_w, pre_cal1, sparsify, cal_glvs, glvs, compute_correlation_matrix


def spatial_mapping(
        ad_st: AnnData,
        ad_sc: AnnData,
        db: DataFrame,
        cluster_key: str,
        spatial_key: str = 'spatial',
        embedding_key: str = 'X_pca',
        uns_key: str = 'rank_genes_groups',
        n_cell: int = 5,
        device: str = 'cuda:0'
):
    """

    """

    M = map_fgw(ad_st, ad_sc, cluster_key, spatial_key, uns_key, device)
    adata_sc.var_names = adata_sc.var_names.str.lower()
    adata_st.var_names = adata_st.var_names.str.lower()

    shared_genes = list(
        set(adata_st.var_names).intersection(set(adata_sc.var_names)))
    adata_st = adata_st[:, shared_genes]
    adata_sc = adata_sc[:, shared_genes]

    W = gen_w(adata_sc, db)
    x_coord = adata_st.obsm[spatial_key]
    scale = np.abs(np.max(x_coord[:, 0]) - np.min(x_coord[:, 0]))
    # parameters
    m_val = .016/scale*5000
    U0 = 0.1 / (2.85 / m_val)  # .1
    V0 = 1.1 / (2.85 / m_val)  # 1.1
    xi1 = 1.21 / (2.85 / m_val)  # 1.21
    xi2 = 1.9 / (2.85 / m_val)  # 1.9
    iterations = 10
    dt = 20
    xsr = .016/scale*5000
    x_r = .016/scale*5000

    Sigma = np.array([[10000, 0], [0, 10000]])
    z_cutoff = 0.4  # level set cutoff for defining tissue boundary

    scale = np.abs(np.max(x_coord[:, 0]) - np.min(x_coord[:, 0]))
    x_coord = x_coord / scale * 5000
    a = np.tile(x_coord[:, 0], (n_cell, 1)).T.flatten()
    b = np.tile(x_coord[:, 1], (n_cell, 1)).T.flatten()
    xs = np.concatenate(([a], [b]), axis=0).T
    xc = xs + np.random.normal(0, xsr, size=xs.shape)

    # Neighbor computation (keep on CPU for now as it's a one-time operation)
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(xc)
    x_id = neigh.kneighbors(xs)  # first entry is distance, second is indices
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
    cell_codes = pd.Categorical(adata_sc.obs['Cell_type']).codes[cell5m]

    W1 = W[cell5m, :]
    W1 = W1[:, cell5m]
    W1 = W1 / np.max(W1)

    degree = np.diag(np.sum(W1, axis=1))
    L = degree - W1

    # Fix determinant computation warning
    try:
        det_L = np.linalg.det(L)
        # Check for valid determinant (not NaN or inf)
        if np.isfinite(det_L) and det_L > 1e-10:
            q = pre_cal1(W1)
            H = sparsify(W1, q)
        else:
            print(
                "Warning: Laplacian matrix is singular or ill-conditioned, using zero matrix")
            H = np.zeros(np.shape(W1))
    except np.linalg.LinAlgError:
        print("Warning: Determinant computation failed, using zero matrix")
        H = np.zeros(np.shape(W1))

    adata_sc1 = adata_sc[cell5m, :].copy()

    if embedding_key not in adata_sc1.obsm:
        sc.pp.pca(adata_sc1)
        embedding_key = 'X_pca'
    X_sc2m2 = adata_sc1.obsm[embedding_key]

    if GPU_AVAILABLE:
        import cupy as cp
        xs_gpu = cp.asarray(xs, dtype=cp.float32)
        xc_gpu = cp.asarray(xc, dtype=cp.float32)
        X_sc2m2_gpu = cp.asarray(X_sc2m2, dtype=cp.float32)
        H_gpu = cp.asarray(H, dtype=cp.float32)
        
        # Convert neighbor lists to GPU format
        neighbor_indices = []
        for i, neighbors in enumerate(x_id1):
            neighbor_indices.append(cp.asarray(np.where(neighbors)[0]))
    else:
        xs_gpu = xs
        xc_gpu = xc
        X_sc2m2_gpu = X_sc2m2
        H_gpu = H
        neighbor_indices = []
        for i, neighbors in enumerate(x_id1):
            neighbor_indices.append(np.where(neighbors)[0])
    correlation_matrix_gpu = compute_correlation_matrix(X_sc2m2_gpu)

def map_fgw(ad_st, ad_sc, cluster_key: str, spatial_key: str = 'spatial', uns_key: str = 'rank_genes_groups', device: str = 'cuda:0'):
    # # load data
    # ad_st=sc.read_h5ad("Mouse_Lymph_Node_pp_ST.h5ad") #spatial transcriptomic data
    # ad_sc=sc.read_h5ad("Mouse_Lymph_Node_pp_SC.h5ad") # single cell data

    # # preprocessing
    # sc.pp.filter_cells(ad_sc,min_genes=3)
    # sc.pp.filter_genes(ad_sc,min_cells=3)
    # sc.pp.normalize_total(ad_sc)
    # sc.pp.log1p(ad_sc)
    # sc.pp.highly_variable_genes(ad_sc)
    # ad_sc=ad_sc[:,ad_sc.var['highly_variable']]
    if uns_key not in ad_sc.uns:
        sc.tl.rank_genes_groups(ad_sc, groupby=cluster_key)
        markers_df = pd.DataFrame(
            ad_sc.uns['rank_genes_groups']['names']).iloc[0:100, :]
        markers = list(np.unique(markers_df.melt().value.values))
        ad_sc = ad_sc[:, ad_sc.var_names.isin(markers)].copy()

    shared_genes = list(
        set(ad_st.var_names).intersection(set(ad_sc.var_names)))
    ad_st = ad_st[:, shared_genes]
    ad_sc = ad_sc[:, shared_genes]
    locations = ad_st.obsm[spatial_key]

    spatial_regularization_strength = 0.1
    z_dim = 50
    lr = 1e-3  # learning rate for spaceflow
    epochs = 1000
    max_patience = 50
    min_stop = 100

    # SpaceFlow graph generation
    spatial_graph = graph_alpha(locations)

    # generating model for spaceflow embedding
    model = DeepGraphInfomax(
        hidden_channels=z_dim, encoder=GraphEncoder(ad_st.shape[1], z_dim),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)

    expr = torch.tensor(ad_st.X.toarray()).float().to(device)

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

        coords = torch.tensor(locations, dtype=torch.float32).to(device)

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
    A = np.multiply(A1d, distance_matrix(locations, locations))
    A /= A.max()
    M = mapper(ad_sc, ad_st, A, max_cells_per_spot=5)  # run mapping
    return M  # numpy matrix output (cell by spot)


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


def mapper(ad_sc, ad_st, A):
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

    # Extract expression matrices
    sc_expr = ad_sc.X
    st_expr = ad_st.X

    # Convert sparse matrices to dense if needed
    if scipy.sparse.issparse(sc_expr):
        sc_expr = sc_expr.toarray()
    if scipy.sparse.issparse(st_expr):
        st_expr = st_expr.toarray()

    # Normalize
    sc_expr_norm = sc_expr / \
        np.sqrt(np.sum(sc_expr**2, axis=1, keepdims=True) + 1e-10)
    st_expr_norm = st_expr / \
        np.sqrt(np.sum(st_expr**2, axis=1, keepdims=True) + 1e-10)

    # Cosine similarity matrix
    similarity_matrix = np.dot(sc_expr_norm, st_expr_norm.T)

    # Convert to distance/cost matrix (1 - similarity) for FGW
    M = 1 - similarity_matrix

    n_cells = ad_sc.shape[0]
    n_spots = ad_st.shape[0]

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

    return T
