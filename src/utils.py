import scipy.sparse as sp
import numpy as np
import networkx as nx
import warnings
from time import perf_counter
from sklearn.model_selection import train_test_split

def load_npz(file_name):
    if not file_name.endswith('.npz'):
        file_name += '_lcc.npz'
    with np.load(f'../data/{file_name}', allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            from sklearn.preprocessing import StandardScaler
            attr_matrix = loader['attr_matrix']
            scaler = StandardScaler()
            scaler.fit(attr_matrix)
            attr_matrix = scaler.transform(attr_matrix)
            attr_matrix = sp.csr_matrix(attr_matrix)

        else:
            attr_matrix = None
        labels = loader.get('labels')
    return adj_matrix, attr_matrix, labels

    
def train_val_test_split_tabular(N, train_size=0.1, val_size=0.1, test_size=0.8, stratify=None, random_state=None):

    idx = np.arange(N)
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test

def sample_targets(nodes, size=1000, seed=2018):
    np.random.seed(seed)
    targets = np.random.choice(nodes, size=size, replace=False)
    return targets
      
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def load_adv_edges(file_name, attacker):
    name = f'../adv_edges/{file_name}_{attacker}.npz'        
    with np.load(name, allow_pickle=True) as loader:
        loader = dict(loader)
        return loader['edges'].item(), loader['time'].item()
    
    
def flip_adj(adj_matrix, flips, symmetric=True):
    if flips is None or len(flips) == 0:
        warnings.warn(
            "There is NO structure flips, the adjacency matrix stays the same.",
            UserWarning,
        )
        return adj_matrix.copy()

    flips = np.asarray(flips)
    flips = np.vstack([flips, flips[:, [1,0]]])
    row, col = flips.T
    data = adj_matrix[row, col].A

    data[data > 0.] = 1.
    data[data < 0.] = 0.
    adj_matrix = adj_matrix.tolil(copy=True)
    adj_matrix[row, col] = 1. - data
    adj_matrix = adj_matrix.tocsr(copy=False)

    adj_matrix.eliminate_zeros()

    return adj_matrix