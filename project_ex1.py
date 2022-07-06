import scipy.io
import math
from scipy import stats
import numpy as np
from itertools import chain, combinations, permutations
import matplotlib.pyplot as plt
import networkx as nx


def ci_test(D, X, Y, Z):
    """ci_test(D, X, Y, Z)
    Assume that variables are [0,1,2,...,p-1] and we have n samples
    Input:
    D: Matrix of data (numpy array with size n*p)
    X: index of the first variable
    Y: index of the second variable
    Z: A list of indices for variables of the conditioning set
    output = True (independent) or False (dependent)

    Example usage:
    1) Z is empty set: ci_test(D, 0, 2, [])
    2) Z={1,2}: ci_test(D, 0, 4, [1,2])

    Note that D must be a numpy array. (D=np.array([[1,2],[2,3]]))"""
    alpha = 0.06
    n = D.shape[0]
    if len(Z) == 0:
        r = np.corrcoef(D[:, [X, Y]].T)[0][1]
    else:
        sub_index = [X, Y]
        sub_index.extend(Z)
        sub_corr = np.corrcoef(D[:, sub_index].T)
        try:
            PM = np.linalg.inv(sub_corr)
        except np.linalg.LinAlgError:
            PM = np.linalg.pinv(sub_corr)
        r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
    cut_at = 0.99999
    r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1

    # Fisher’s z-transform
    res = math.sqrt(n - len(Z) - 3) * .5 * math.log1p((2 * r) / (1 - r))
    p_value = 2 * (1 - stats.norm.cdf(abs(res)))

    return p_value >= alpha


D1 = scipy.io.loadmat('D1.mat')["D"]
D2 = scipy.io.loadmat('D2.mat')["D"]
D3 = scipy.io.loadmat('D3.mat')["D"]
D4 = scipy.io.loadmat('D4.mat')["D"]
TEST = np.asmatrix([[1] * 5] * 500)

tables = [None, D1, D2, D3, D4]


def matrix_labels(dataset):
    """returns a dictionary with names of the variable as column number (1:1)"""
    matrix = tables[dataset]
    labels = {}
    for i in range(np.shape(matrix)[1]):
        labels.update({i: i})
    return labels


def powerset(iterable, lenght=None):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    if lenght is None:
        output = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    else:
        output = chain.from_iterable(combinations(s, r) for r in range(lenght + 1))
    return list(output)


def diff(list1, list2):
    """gives back the the values not present in both the two lists given as input"""
    return list(set(list1).symmetric_difference(set(list2)))


def undirected_connection(matrix, edge1, edge2):
    """gives back if there exists an undirected connection between two nodes"""
    return matrix[edge1, edge2] == 1 and matrix[edge2, edge1] == 1


def direct_connection(matrix, from_edge, to_edge):
    """gives back if there exists a directed connection between two nodes (from_edge -> to_edge)"""
    return matrix[from_edge, to_edge] == 1 and matrix[to_edge, from_edge] == -1


def exist_connection(matrix, edge1, edge2):
    """gives back if there exists a connection between two nodes (doesn't matter the direction)"""
    return abs(matrix[edge1, edge2]) == 1 and abs(matrix[edge2, edge1]) == 1


def SGS(dataset):
    """First algorithm to find the skeleton.
    Returns back the adjacency matrix of the undirected graph and the separations that have been found"""
    separations = []
    matrix = tables[dataset]
    ci_number = 0
    variables = np.shape(matrix)[1]
    list_variables = list(range(variables))
    empty_graph = np.asmatrix([[0] * variables] * variables)
    pairs = list(combinations(list(range(variables)), 2))
    for pair in pairs:
        conditional_subsets = [x for x in powerset(diff(list_variables, pair))]
        independent = False
        for Z in conditional_subsets:
            independent = ci_test(matrix, pair[0], pair[1], Z)
            ci_number += 1
            if independent:
                separations.append(({pair[0], pair[1]}, Z))
                break
        if not independent:
            empty_graph[pair[0], pair[1]] = 1
            empty_graph[pair[1], pair[0]] = 1
    print("SGS:", ci_number)
    return empty_graph, separations


def check_vertex_with_least_d_neighbors(matrix: np.matrix, d):
    """returns if there exist at least one vertex with d neighbors"""
    output = False
    for line in matrix:
        if np.sum(line) >= d:
            output = True
            break
    return output


def PC1(dataset):
    """Second algorithm to find the skeleton.
    Returns back the adjacency matrix of the undirected graph and the separations that have been found"""
    matrix = tables[dataset]
    separations = []
    ci_number = 0
    variables = np.shape(matrix)[1]
    list_variables = list(range(variables))
    empty_graph = np.asmatrix([[1] * variables] * variables)
    np.fill_diagonal(empty_graph, 0)
    pairs = list(combinations(list(range(variables)), 2))
    d = 0
    while check_vertex_with_least_d_neighbors(empty_graph, d):
        for pair in pairs:
            if undirected_connection(empty_graph, pair[0], pair[1]):
                conditional_subsets = [x for x in powerset(diff(list_variables, pair), d)]
                for Z in conditional_subsets:
                    if len(Z) == d:
                        independent = ci_test(matrix, pair[0], pair[1], Z)
                        ci_number += 1
                        if independent:
                            separations.append(({pair[0], pair[1]}, Z))
                            empty_graph[pair[0], pair[1]] = 0
                            empty_graph[pair[1], pair[0]] = 0
                            break
        d += 1
    print("PC1:", ci_number)
    return empty_graph, separations


def PC2(dataset):
    """Third algorithm to find the skeleton.
    Returns back the adjacency matrix of the undirected graph and the separations that have been found"""
    matrix = tables[dataset]
    separations = []
    ci_number = 0
    variables = np.shape(matrix)[1]
    list_variables = list(range(variables))
    empty_graph = np.asmatrix([[0] * variables] * variables)
    pairs = list(combinations(list(range(variables)), 2))
    connected_pairs = []
    for pair in pairs:
        Z = diff(list_variables, pair)
        independent = ci_test(matrix, pair[0], pair[1], Z)
        ci_number += 1
        if not independent:
            connected_pairs.append(pair)
            empty_graph[pair[0], pair[1]] = 1
            empty_graph[pair[1], pair[0]] = 1
        else:
            separations.append(({pair[0], pair[1]}, Z))
    d = 0
    while check_vertex_with_least_d_neighbors(empty_graph, d):
        for connection in connected_pairs:
            conditional_subsets = [x for x in powerset(diff(list_variables, connection), d)]
            for Z in conditional_subsets:
                if len(Z) == d:
                    independent = ci_test(matrix, connection[0], connection[1], Z)
                    ci_number += 1
                    if independent:
                        separations.append(({connection[0], connection[1]}, Z))
                        empty_graph[connection[0], connection[1]] = 0
                        empty_graph[connection[1], connection[0]] = 0
                        connected_pairs.remove(connection)
                        break
        d += 1
    print("PC2:", ci_number)
    return empty_graph, separations


def orientation(adj_matrix_separations):
    """algorithm that finds the orientation of the edges, knowing an undirected matrix.
    Returns the adjacency matrix of the semi-directed graph"""
    adj_matrix = adj_matrix_separations[0]
    output = adj_matrix.copy()
    variables = np.shape(adj_matrix)[1]
    list_variables = list(range(variables))
    for line_number in list_variables:
        line_array = np.asarray(adj_matrix[line_number])
        number_edges = np.sum(line_array)
        if number_edges >= 2:
            edges = np.where(line_array == 1)
            listOfCoordinates = list(zip(edges[0], edges[1]))
            neightbours = [i[1] for i in listOfCoordinates]
            pairs = list(combinations(neightbours, 2))
            unconnected_pairs = [pair for pair in pairs if not undirected_connection(adj_matrix, pair[0], pair[1])]
            for pair in unconnected_pairs:
                past_CI = adj_matrix_separations[1]
                pair_past_CI = [log for log in past_CI if log[0] == set(pair)]
                for subset in pair_past_CI:
                    if line_number in subset[1]:
                        break
                    output[line_number, pair[0]] = -1
                    output[line_number, pair[1]] = -1

    triplets = list(permutations(list_variables, 3))
    # Rule1
    for triplet in triplets:
        A = triplet[0]
        B = triplet[1]
        C = triplet[2]
        if direct_connection(output, A, B) and undirected_connection(output, B, C) and not exist_connection(output, A,
                                                                                                            C):
            output[C, B] = -1
    # Rule2
    for triplet in triplets:
        A = triplet[0]
        B = triplet[1]
        C = triplet[2]
        if direct_connection(output, A, B) and direct_connection(output, B, C) and undirected_connection(output, A,
                                                                                                         C):
            output[C, A] = -1

    return output


def show_undirected_graph(adjacency_matrix: np.matrix, title: str):
    """Returns the graphical representation of the undirected adjacency matrix"""
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    max_degree = sorted(gr.degree, key=lambda x: x[1], reverse=True)[0]
    print("Max degree:", max_degree[1])
    pos = nx.spring_layout(gr, k=1.2)
    nx.draw(gr, pos, node_size=400, with_labels=True)
    plt.savefig(title)
    plt.show()


def show_directed_graph(adjacency_matrix: np.matrix, title: str):
    """Returns the graphical representation of the directed adjacency matrix"""
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    max_degree = sorted(gr.degree, key=lambda x: x[1], reverse=True)[0]
    print("Max degree:", max_degree[1])
    pos = nx.spring_layout(gr, k=1.2)
    nx.draw(gr, pos, node_size=400, with_labels=True)
    plt.savefig(title)
    plt.show()


if __name__ == '__main__':
    datasets = [1, 2, 3, 4]
    tables_str = ["D1", "D2", "D3", "D4"]
    for table in datasets:
        table_name = tables_str[table - 1]
        print("Table {}:".format(table_name))
        if table != 4:
            SGS_matrix = SGS(table)
            PC1_matrix = PC1(table)
            PC2_matrix = PC2(table)
            #print("******************")
            show_directed_graph(orientation(SGS_matrix), "SGS-{}.png".format(table_name))
            show_directed_graph(orientation(PC1_matrix), "PC1-{}.png".format(table_name))
            show_directed_graph(orientation(PC2_matrix), "PC2-{}.png".format(table_name))
            print("******************")
        else:
            PC2_matrix = PC2(table)
            oriented_matrix = orientation(PC2_matrix)
            np.savetxt("D4.csv", oriented_matrix, fmt="%.0f")
            show_directed_graph(oriented_matrix, "PC2-{}.png".format(table_name))
