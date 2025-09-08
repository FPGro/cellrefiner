from typing import Optional
import numpy as np
import pandas as pd
from anndata import AnnData
from ._cell_shape_modeling import SEM
from scipy.sparse import csr_matrix


def contact_communication(
        df_ligrec: pd.DataFrame,
        adata: AnnData,
        sem: Optional[SEM] = None,
        contact_key: Optional[str] = 'contacts',
        lr_delimiter: str = '-',
        heteromeric_delimiter: str = '_'
    ) -> None:
    '''
    Contact-base communication inference

    Parameters
    ----------
    df_ligrec : DataFrame
        Dataframe where each row corresponds to a ligand-receptor pair with ligands, receptors, and the associated signaling pathways in the three columns, respectively.
    adata : Anndata
        Anndata object, must contain cell-cell contact information in `.obsp[contact_key]` if `sem` is None.
    sem : SEM, optional
        Cell shape model object that contains cell-cell contact matrix and associated AnnData.

        If provided, contact matrix will be obtained from `sem.contact_matrix`.

        If both `sem` and `adata` are provided, `adata` parameter takes precedence.
    contact_key : str, default 'contacts'
        Key in `adata.obsp` containing the cell-cell contact matrix (csr_matrix).
    lr_delimiter : str, default '-'
        Delimiter used to construct ligand-receptor pair names in output.
    heteromeric_delimiter : str, default '_'
        Delimiter used to separate subunits in heteromeric complexes within `df_ligrec`.

        For example, if a receptor complex is 'TGFBR1_TGFBR2', this parameter should be '_'.
    Returns
    -------
    Sets the following fields in adata

        add `.obsp['{ligand}{lr_delimiter}{receptor}']`, contact-base communication matrix via ligand-receptor pairs 
        (rows are sender cells, columns are receiver cells)

        add `.obsp['{pathway}']`, pathway-level contact-base communication matrix

        add `.obsp['total']`, sum of all pathway communication matrix

        add `.obsm['sender_signal']`, dataFrame with sender communication strengths per cell

        add `.obsm['receiver_signal']`, dataFrame with receiver communication strengths per cell

        add `.uns['contact_signal_info']`, metadata of the analysis
            - 'lr_pair': List of L-R pair names
            - 'pathway': List of pathway names  
            - 'total': ['total']
            - 'db': Filtered ligand-receptor database
    
    Examples
    --------
    >>> db_lr = cr.pp.ligand_receptor_database()
    >>> db_lr = cr.pp.filter_lr_database(db_lr,adata_cr, min_cell_pct=0.01)
    >>> cr.tl.contact_communication(db_lr, adata)
    '''

    if df_ligrec.shape[0] == 0:
        raise ValueError("empty ligand-receptor DB")
    if sem is None:  # sem is not provided, using adata contact matrix
        contact_matrix = adata.obsp[contact_key]
    else:  # sem is provided
        if adata is None:  # adata is not provided, use sem.adata
            adata = sem.adata
        else:  # adata is provided, use input adata
            if adata is not sem.adata:  # check if same adata
                Warning(
                    'Provide adata is not an attribute of sem, sem.adata will be unchanged')
        if sem.contact_matrix is None:
            print('compute cell-cell contact')
            sem.compute_contact()
        else:
            contact_matrix = sem.contact_matrix
    df_ligrec = df_ligrec.copy()
    # get cell pair index
    nc = adata.shape[0]
    indices = contact_matrix.indices
    indptr = contact_matrix.indptr
    ci = []
    cj = []
    for i in range(nc):
        j = indices[indptr[i]:indptr[i+1]]
        ci.append(np.tile(i, len(j)))
        cj.append(j)
    ci = np.concatenate(ci)
    cj = np.concatenate(cj)

    # contact signal
    lr_keys = []
    I = np.ones(df_ligrec.shape[0], dtype=bool)
    # ligand-receptors pairs
    for i in range(df_ligrec.shape[0]):
        l = df_ligrec.iloc[i, 0]
        r = df_ligrec.iloc[i, 1]
        l_data = np.prod(
            adata[ci, l.split(heteromeric_delimiter)].X.toarray(), axis=1)
        r_data = np.prod(
            adata[cj, r.split(heteromeric_delimiter)].X.toarray(), axis=1)
        key = f'{l}{lr_delimiter}{r}'
        # .copy() is necessary. eliminate_zeros() removes indices and indptr inplace
        sig_mat = csr_matrix(
            (l_data*r_data, indices.copy(), indptr.copy()), shape=(nc, nc))
        sig_mat.eliminate_zeros()
        I[i] = sig_mat.nnz > 0
        if I[i]:
            adata.obsp[key] = sig_mat
            lr_keys.append(key)
    df_ligrec = df_ligrec[I]

    # pathway and total
    pth_keys = df_ligrec.iloc[:, 2].unique().tolist()
    for n, pth in enumerate(pth_keys):
        lr_idx = np.where(df_ligrec.iloc[:, 2] == pth)[0]
        data = csr_matrix((nc, nc))
        for i in lr_idx:
            l = df_ligrec.iloc[i, 0]
            r = df_ligrec.iloc[i, 1]
            data += adata.obsp[f'{l}{lr_delimiter}{r}'].copy()
        adata.obsp[pth] = data.copy()
        if n == 0:
            total = data.copy()
        else:
            total += data.copy()
    adata.obsp['total'] = total

    # contact signal information
    adata.uns['contact_signal_info'] = {
        'lr_pair': lr_keys, 'pathway': pth_keys, 'total': ['total'], 'db': df_ligrec}
    print("add .uns['contact_signal_info']")

    # receiver/sender signal
    signal_list = lr_keys + pth_keys + ['total']
    sdim = len(signal_list)
    signal_vec_s = np.zeros((adata.shape[0], sdim))
    signal_vec_r = np.zeros((adata.shape[0], sdim))
    for si, signal in enumerate(signal_list):
        signal_vec_s[:, si] = np.sum(
            adata.obsp[signal].toarray(), axis=1)  # sender signal
        signal_vec_r[:, si] = np.sum(
            adata.obsp[signal].toarray(), axis=0)  # receiver signal
    df_s = pd.DataFrame(index=adata.obs.index,
                        columns=signal_list, data=signal_vec_s)
    df_r = pd.DataFrame(index=adata.obs.index,
                        columns=signal_list, data=signal_vec_r)
    adata.obsm['sender_signal'] = df_s
    adata.obsm['receiver_signal'] = df_r
    print("add .obsm['sender_signal'], .obsm['receiver_signal']")


def cluster_communication(adata: AnnData,
                          cluster_key: str,
                          signal: str,
                          n_permutations: int = 100,
                          seed: int = 0):
    """
    Summarize cell-cell communication to cluster-cluster communication and compute p-values 
    by permutating cell/spot labels.

    Parameters
    ----------
    adata : Anndata
        Anndata object, must contain cell-cell contact information in `.obsp[contact_key]` if `sem` is None.
    cluster_key : str
        Key in `.obs` that contains cell type annotations
    signal : str
        Key in `.obsp` that contains communication matrix
    n_permutations : int, default 100
        Number of label permutations for computing the p-value.
    seed : int, default 0
        random seed

    Returns
    -------
    Sets the following fields in adata

        add `.uns['{cluster_key}-{signal}']`, cluster-level communication via {signal}
            - .uns['{cluster_key}-{signal}']['communication_matrix'], cluster-level communication matrix
            - .uns['{cluster_key}-{signal}']['communication_pvalue'], p-values
    
    Examples
    --------
    >>> cr.tl.cluster_communication(adata,cluster_key = 'cell_type',signal = 'NOTCH')
    """

    cluster_list = list(adata.obs[cluster_key].cat.categories)
    cluster_cell = adata.obs[cluster_key].to_numpy()
    sig_mat = adata.obsp[signal]
    rng = np.random.default_rng(seed)
    tmp_df, tmp_p_value = summarize_cluster(
        sig_mat, cluster_cell, cluster_list, rng, n_permutations=n_permutations)
    adata.uns[cluster_key+'-' + signal] = {'communication_matrix': tmp_df, 'communication_pvalue': tmp_p_value}


def summarize_cluster(X, clusterid, clusternames, rng, n_permutations):
    # Input a sparse matrix of cell signaling and output a pandas dataframe
    # for cluster-cluster signaling
    n = len(clusternames)
    X_cluster = np.empty([n, n], float)
    p_cluster = np.zeros([n, n], float)
    for i in range(n):
        tmp_idx_i = np.where(clusterid == clusternames[i])[0]
        for j in range(n):
            tmp_idx_j = np.where(clusterid == clusternames[j])[0]
            X_cluster[i, j] = X[tmp_idx_i, :][:, tmp_idx_j].mean()

    for i in range(n_permutations):
        clusterid_perm = rng.permutation(clusterid)
        X_cluster_perm = np.empty([n, n], float)
        for j in range(n):
            tmp_idx_j = np.where(clusterid_perm == clusternames[j])[0]
            for k in range(n):
                tmp_idx_k = np.where(clusterid_perm == clusternames[k])[0]
                X_cluster_perm[j, k] = X[tmp_idx_j, :][:, tmp_idx_k].mean()
        p_cluster[X_cluster_perm >= X_cluster] += 1.0
    p_cluster = p_cluster / n_permutations
    df_cluster = pd.DataFrame(
        data=X_cluster, index=clusternames, columns=clusternames)
    df_p_value = pd.DataFrame(
        data=p_cluster, index=clusternames, columns=clusternames)
    return df_cluster, df_p_value
