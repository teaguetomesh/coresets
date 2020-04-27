"""
A set of functions for solving the k-means clustering problem using QAOA
"""
import sys
import numpy as np
import scipy.linalg as linalg
import networkx as nx
import matplotlib.pyplot as plt
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit


def gen_coreset_graph():
    # Generate a graph instance with sample coreset data
    coreset_points = []
    # generate 3 points around x=-1, y=-1
    for _ in range(3):
        # use a uniformly random weight
        #weight = np.random.uniform(0.1,5.0,1)[0]
        weight = 1
        vector = np.array([np.random.normal(loc=-1, scale=0.5, size=1)[0], np.random.normal(loc=-1, scale=0.5, size=1)[0]])
        new_point = (weight, vector)
        coreset_points.append(new_point)

    # generate 3 points around x=+1, y=1
    for _ in range(2):
        # use a uniformly random weight
        #weight = np.random.uniform(0.1,5.0,1)[0]
        weight = 1
        vector = np.array([np.random.normal(loc=1, scale=0.5, size=1)[0], np.random.normal(loc=1, scale=0.5, size=1)[0]])
        new_point = (weight, vector)
        coreset_points.append(new_point)

    # Generate a networkx graph with correct edge weights
    n = len(coreset_points)
    G = nx.complete_graph(n)
    for edge in G.edges():
        v_i = edge[0]
        v_j = edge[1]
        w_i = coreset_points[v_i][0]
        w_j = coreset_points[v_j][0]
        #dot_prod = np.dot(coreset_points[v_i][1], coreset_points[v_j][1])
        dist = np.linalg.norm(coreset_points[v_i][1] - coreset_points[v_j][1])
        G[v_i][v_j]['weight'] = w_i * w_j * dist

    return coreset_points, G


def plot_coreset_graph(coreset_points, G):
    # Plot the coreset points
    xx = [cp[1][0] for cp in coreset_points]
    yy = [cp[1][1] for cp in coreset_points]
    plt.scatter(xx, yy)
    plt.hlines(0, np.amin(xx), np.amax(xx), ls='--')
    plt.vlines(0, np.amin(yy), np.amax(yy), ls='--')
    plt.show()
    plt.close()

    # Generate a networkx graph with correct edge weights
    n = len(coreset_points)
    G = nx.complete_graph(n)
    for edge in G.edges():
        v_i = edge[0]
        v_j = edge[1]
        w_i = coreset_points[v_i][0]
        w_j = coreset_points[v_j][0]
        #dot_prod = np.dot(coreset_points[v_i][1], coreset_points[v_j][1])
        dist = np.linalg.norm(coreset_points[v_i][1] - coreset_points[v_j][1])
        G[v_i][v_j]['weight'] = w_i * w_j * dist
        
    # Generate plot of the Graph
    colors = ['r' for node in G.nodes()]
    default_axes = plt.axes(frameon=False)
    pos = nx.spring_layout(G)

    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)


# Create the quantum circuit to implement the QAOA
def evolve_cost(c, angle, G, ising, qbit_map):
    if qbit_map is None:
        for edge in G.edges():
            i = edge[0]
            j = edge[1]
            phi = angle * G[i][j]['weight']
            if ising:
                c.cu1(-4*phi, i, j)
                c.u1(2*phi, i)
                c.u1(2*phi, j)
            else:
                c.cx(i,j)
                c.rz(2*phi, j)
                c.cx(i,j)
    else:
        # implement the swap network
        cover_a = [(idx-1, idx) for idx in range(1,len(qbit_map),2)]
        cover_b = [(idx, idx+1) for idx in range(1,len(qbit_map),2)]
        for l, layer in enumerate(range(len(G.nodes))):
            cover = [cover_a, cover_b][layer % 2]
            for pair in cover:
                i, j = pair
                phi = angle * G[i][j]['weight']
                c.cx(i, j)
                c.rz(2*phi, j)
                if l == len(G.nodes) - 1:
                    c.cx(i, j)
                else:
                    c.cx(j, i)
                    c.cx(i, j)


def evolve_driver(c, angle):
    c.rx(2*angle, c.qubits)

def gen_complete_qaoa_circ(P, gamma, beta, G, ising=False, topology=[]):
    """
    P (int) : number of layers to apply
    gamma (list[float]) : list with length p, contains angle parameters for cost Hamiltonian
    beta (list[float]) : list with length p, contains angle parameters for driver Hamiltonian
    G (graph) : NetworkX graph representing the MAXCUT problem instance
    ising (bool) : If true, use Cu1 gates to implement cost evolution, otherwise use cx
    """
    # create the quantum and classical registers
    n = len(G.nodes()) # get the number of vertices (qubits)
    circ = QuantumCircuit(n)

    # apply the initial layer of Hadamards
    circ.h(range(n))

    # iteratively apply the cost and driver unitaries p times
    if len(topology) == 0:
        for p in range(P):
            circ.barrier()
            evolve_cost(circ, gamma[p], G, ising, None)
            circ.barrier()
            evolve_driver(circ, beta[p])
    else:
        # implement the qaoa circuit using a swap network to minimize CNOT count
        # for now, assume a linear mapping
        initial_layout = np.arange(0,n)

        for p in range(P):
            circ.barrier()
            evolve_cost(circ, gamma[p], G, ising, initial_layout)
            circ.barrier()
            evolve_driver(circ, beta[p])

    # apply measurements to all qubits
    circ.measure_all()

    if len(topology) == 0:
        return circ
    else:
        return circ, initial_layout


# Compute the value of the cost function
def cost_function_C(x,G):
    
    E = G.edges()
    if( len(x) != len(G.nodes())):
        return np.nan
        
    C = 0;
    for edge in E:
        e1 = edge[0]
        e2 = edge[1]
        w = G[e1][e2]['weight']
        C = C + w*(x[e1]*(1-x[e2]) + x[e2]*(1-x[e1]))
        
    return C
