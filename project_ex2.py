import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

mat = scipy.io.loadmat('market.mat')
dataset = mat['DI'].T


def equation(X, Y, Z):
    """Applying the formula"""
    N, T = np.shape(X)
    DI_sum = 0
    for t in range(2, T + 1):
        cov1 = np.zeros((N, t + 10 * (t - 1)))
        cov1[:, :t] = Y[:, :t]
        for z in range(10):
            cov1[:, t + z * (t - 1): t + (z + 1) * (t - 1)] = Z[z][:, :(t - 1)]
        det_cov1 = np.linalg.det(np.cov(cov1.T))

        cov2 = np.zeros((N, 12 * (t - 1)))
        cov2[:, :(t - 1)] = X[:, :(t - 1)]
        cov2[:, (t - 1):(2 * (t - 1))] = Y[:, :(t - 1)]
        for z in range(10):
            cov2[:, ((z + 2) * (t - 1)): (z + 3) * (t - 1)] = Z[z][:, :(t - 1)]
        det_cov2 = np.linalg.det(np.cov(cov2.T))

        cov3 = np.zeros((N, 11 * (t - 1)))
        cov3[:, :(t - 1)] = Y[:, :(t - 1)]
        for z in range(10):
            cov3[:, ((z + 1) * (t - 1)): (z + 2) * (t - 1)] = Z[z][:, :(t - 1)]
        det_cov3 = np.linalg.det(np.cov(cov3.T))

        cov4 = np.zeros((N, t + 11 * (t - 1)))
        cov4[:, :(t - 1)] = X[:, :(t - 1)]
        cov4[:, (t - 1):(2 * t - 1)] = Y[:, :t]
        for z in range(10):
            cov4[:, t + (z + 1) * (t - 1): t + (z + 2) * (t - 1)] = Z[z][:, :(t - 1)]
        det_cov4 = np.linalg.det(np.cov(cov4.T))

        DI_sum += 0.5 * np.log((det_cov1 / det_cov3) * (det_cov2 / det_cov4))
    return DI_sum


def diff(list1, list2):
    """gives back the the values not present in both the two lists given as input"""
    return list(set(list1).symmetric_difference(set(list2)))


def apply_DI(i, j, data):
    """function that applies the formula on the dataset"""
    ent = 0
    X = data[i]
    Y = data[j]
    Z = np.zeros((10, 190, 7))
    variables = list(range(len(data)))
    conditional_subset = diff([i, j], variables)
    for k in conditional_subset:
        Z[ent] = data[k]
        ent += 1
    return equation(X, Y, Z)


def DI_matrix(data):
    """function that performs the formula for each variable and gives back the matrix with the results"""
    empthy_matrix = np.zeros((12, 12))
    variables = list(range(len(data)))
    for ent in variables:
        to_vertex = diff([ent], variables)
        for vertex in to_vertex:
            empthy_matrix[ent, vertex] = apply_DI(ent, vertex, data)
    return empthy_matrix


def matrix_for_graph(matrix):
    """function that takes the DI_matrix and applies the trashold of 0.5 to get a useful adjacency matrix"""
    output = matrix.copy()
    for line in range(len(matrix)):
        for value in range(len(matrix[line])):
            if matrix[line, value] >= 0.5:
                output[line, value] = 1
            else:
                output[line, value] = 0
    return output


def show_directed_graph(adjacency_matrix: np.matrix, title: str):
    """Returns the graphical representation of the directed adjacency matrix"""
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_nodes_from(list(range(len(adjacency_matrix))))
    gr.add_edges_from(edges)
    pos = nx.spring_layout(gr, k=0.5)
    nx.draw(gr, pos, node_size=400, with_labels=True)
    plt.savefig(title)
    plt.show()


if __name__ == '__main__':
    matrix_DI = DI_matrix(dataset)
    np.savetxt("DI_matrix_for_graph.csv", matrix_DI, fmt="%.3f")
    show_directed_graph(matrix_for_graph(matrix_DI), "DI_matrix.png")
