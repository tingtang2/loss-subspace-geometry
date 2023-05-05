# adapted from https://github.com/HazyResearch/hyperbolics/blob/master/products/curv.py

import logging  #, argh
import os, sys
import networkx as nx
import numpy as np

import scipy.sparse.csgraph as csg
from multiprocessing import Pool
from numpy.random import default_rng
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

rng = default_rng(11202022)

# root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, root_dir)


def load_graph(file_name, directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    with open(file_name, "r") as f:
        for line in f:
            tokens = line.split()
            u = int(tokens[0])
            v = int(tokens[1])
            if len(tokens) > 2:
                w = float(tokens[2])
                G.add_edge(u, v, weight=w)
            else:
                G.add_edge(u, v)
    return G


def Ka(D, m, b, c, a):
    if a == m: return 0.0
    k = D[a][m]**2 + D[b][c]**2 / 4.0 - (D[a][b]**2 + D[a][c]**2) / 2.0
    k /= 2 * D[a][m]
    # print(f'{m}; {b} {c}; {a}: {k}')
    return k


def K(D, n, m, b, c):
    ks = [Ka(D, m, b, c, a) for a in range(n)]
    return np.mean(ks)


def estimate_curvature(G, D, n):
    for m in range(n):
        ks = []
        edges = list(G.edges(m))
        for i in range(len(edges)):
            for j in range(b, len(edges)):
                b = edges[i]
                c = edges[j]
                ks.append(K(D, n, b, c))
        # TODO turn ks into a cdf

    return None


# TODO: what is the correct normalization wrt n? e.g. make sure it works for d-ary trees
def sample_G(G, D, n, n_samples=100):
    samples = []
    _cnt = 0
    while _cnt < n_samples:
        m = rng.integers(0, n)
        edges = list(G.edges(m))
        # print(f"edges of {m}: {edges}")
        i = rng.integers(0, len(edges))
        j = rng.integers(0, len(edges))
        b = edges[i][1]
        c = edges[j][1]
        # TODO special case for b=c?
        if b == c: continue
        a = rng.integers(0, n)
        k = Ka(D, m, b, c, a)
        samples.append(k)
        # print(k)

        _cnt += 1

    return np.array(samples)


def sample_components(n1=5, n2=5):
    """ Sample dot products of tangent vectors """
    a1 = rng.chisquare(n1 - 1)  # ||x_1||^2
    b1 = rng.chisquare(n1 - 1)  # ||y_1||^2
    c1 = 2 * rng.beta((n1 - 1) / 2, (n1 - 1) / 2) - 1  # <x_1,y_1> normalized
    c1 = a1 * b1 * c1**2  # <x_1,y_1>^2
    a2 = rng.chisquare(n2 - 1)  # ||x_1||^2
    b2 = rng.chisquare(n2 - 1)  # ||y_1||^2
    c2 = 2 * rng.beta((n2 - 1) / 2, (n2 - 1) / 2) - 1
    c2 = a2 * b2 * c2**2
    alpha1 = a1 * b1 - c1
    alpha2 = a2 * b2 - c2
    beta = a1 * b2 + a2 * b1
    denom = alpha1 + alpha2 + beta

    return alpha1 / denom, alpha2 / denom


def sample_K(m1, m2, n1=5, n2=5, n_samples=100):
    w1s = []
    w2s = []
    for _ in range(n_samples):
        w2, w1 = sample_components(n1, n2)
        w1s.append(w1)
        w2s.append(w2)
    # match moments of K1 * w1 + K2 * w2
    w1s = np.array(w1s)
    w2s = np.array(w2s)
    coefK1 = np.mean(w1s)
    coefK2 = np.mean(w2s)
    coefK1K1 = np.mean(w1s**2)  # coefficient of K1^2
    coefK2K2 = np.mean(w2s**2)
    coefK1K2 = np.mean(2 * w1s * w2s)

    print("coefs", coefK1, coefK2, coefK1K1, coefK1K2, coefK2K2)

    # turn into quadratic a K1^2 + b K1 + c = 0
    # a = coefK1K1 - coefK1K2*coefK1/coefK2 + coefK2K2*coefK1**2/coefK2**2
    # b = coefK1K2*m1/coefK2 - 2*coefK2K2*m1*coefK1/coefK2
    # c = coefK2K2*m1**2/coefK2**2 - m2
    a = coefK2**2 * coefK1K1 - coefK2 * coefK1K2 * coefK1 + coefK2K2 * coefK1**2
    b = coefK2 * coefK1K2 * m1 - coefK2 * 2 * coefK2K2 * m1 * coefK1
    c = coefK2K2 * m1**2 - coefK2**2 * m2
    print("quadratic", a, b, c)

    K1_soln1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    K1_soln2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    K2_soln1 = (m1 - coefK1 * K1_soln1) / coefK2
    K2_soln2 = (m1 - coefK1 * K1_soln2) / coefK2

    return ((K1_soln1, K2_soln1), (K1_soln2, K2_soln2))


def djikstra_wrapper(_x):
    (mat, x) = _x
    return csg.dijkstra(mat, indices=x, unweighted=False, directed=False)


def build_distance(G, scale, num_workers=None):
    n = G.order()
    p = Pool() if num_workers is None else Pool(num_workers)

    #adj_mat_original = nx.to_scipy_sparse_matrix(G)
    adj_mat_original = nx.to_scipy_sparse_matrix(G,
                                                 nodelist=list(range(
                                                     G.order())))

    # Simple chunking
    nChunks = 128 if num_workers is not None and num_workers > 1 else n
    if n > nChunks:
        chunk_size = n // nChunks
        extra_chunk_size = (n - (n // nChunks) * nChunks)

        chunks = [
            list(range(k * chunk_size, (k + 1) * chunk_size))
            for k in range(nChunks)
        ]
        if extra_chunk_size > 0:
            chunks.append(list(range(n - extra_chunk_size, n)))
        Hs = p.map(djikstra_wrapper,
                   [(adj_mat_original, chunk) for chunk in chunks])
        H = np.concatenate(Hs, 0)
        logging.info(f"\tFinal Matrix {H.shape}")
    else:
        H = djikstra_wrapper((adj_mat_original, list(range(n))))

    H *= scale
    return H


# def match_moments(coefK1, coefK2, coefK12, coefK22, coefK1K2, m1, m2):


def convert_to_graph(X: np.ndarray, graph_type):
    if graph_type == 'knn':
        A = kneighbors_graph(X=X, n_neighbors=5, mode='distance')
    elif graph_type == 'epsilon':
        A = radius_neighbors_graph(X=X)

    return nx.from_scipy_sparse_matrix(A.tocsr())


# @argh.arg('--dataset')
def estimate(dataset='data/edges/smalltree.edges',
             n_samples=100000,
             graph_type='knn'):
    if isinstance(dataset, str):
        G = load_graph(dataset)
    else:
        G = convert_to_graph(dataset, graph_type=graph_type)
    n = G.order()

    num_workers = 16
    D = build_distance(G, 1.0, num_workers)  # load the whole matrix

    # n_samples = 100000
    n1 = 5
    n2 = 5
    samples = sample_G(G, D, n, n_samples)
    # coefs = sample_K(n1, n2, n_samples)
    print("stats", np.mean(samples), np.std(samples)**2, np.mean(samples**2))
    m1 = np.mean(samples)
    m2 = np.mean(samples**2)
    solns = sample_K(m1, m2, n1, n2, n_samples)
    print(solns)


if __name__ == '__main__':
    estimate()
    # parser = argh.ArghParser()
    # parser.set_default_command(estimate)
    # parser.dispatch()