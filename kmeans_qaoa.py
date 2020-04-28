"""
A set of functions for solving the k-means clustering problem using QAOA
"""
import sys
import pickle
import datetime
import numpy as np
import scipy.linalg as linalg
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from copy import copy
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from qiskit.tools.monitor import job_monitor


def gen_coreset_graph(coreset=None, metric='dot'):
    """
    Generate a complete weighted graph using the provided set of coreset points

    Parameters
    ----------
    coreset : List((weight, vector))
        Set of coreset points to use. Each point should consist of a weight
        value and a numpy array as the vector
    metric : str
        Choose the desired metric for computing the edge weights.
        Options include: dot, dist

    Returns
    -------
    coreset : List((weight, vector))
        The set of points used to construct the graph
    G : NetworkX Graph
        A complete weighted graph
    H : List((coef, pauli_string))
        The equivalent Hamiltonian for the generated graph
    """
    if coreset is None:
        # Generate a graph instance with sample coreset data
        coreset = []
        # generate 3 points around x=-1, y=-1
        for _ in range(3):
            # use a uniformly random weight
            #weight = np.random.uniform(0.1,5.0,1)[0]
            weight = 1
            vector = np.array([np.random.normal(loc=-1, scale=0.5, size=1)[0],
                               np.random.normal(loc=-1, scale=0.5, size=1)[0]])
            new_point = (weight, vector)
            coreset.append(new_point)

        # generate 3 points around x=+1, y=1
        for _ in range(2):
            # use a uniformly random weight
            #weight = np.random.uniform(0.1,5.0,1)[0]
            weight = 1
            vector = np.array([np.random.normal(loc=1, scale=0.5, size=1)[0],
                               np.random.normal(loc=1, scale=0.5, size=1)[0]])
            new_point = (weight, vector)
            coreset.append(new_point)

    # Generate a networkx graph with correct edge weights
    n = len(coreset)
    G = nx.complete_graph(n)
    H = []
    for edge in G.edges():
        pauli_str = ['I']*n
        # coreset points are labelled by their vertex index
        v_i = edge[0]
        v_j = edge[1]
        pauli_str[v_i] = 'Z'
        pauli_str[v_j] = 'Z'
        w_i = coreset[v_i][0]
        w_j = coreset[v_j][0]
        if metric == 'dot':
            mval = np.dot(coreset[v_i][1], coreset[v_j][1])
        elif metric == 'dist':
            mval = np.linalg.norm(coreset[v_i][1] - coreset[v_j][1])
        else:
            raise Exception('Unknown metric: {}'.format(metric))
        G[v_i][v_j]['weight'] = w_i * w_j * mval
        H.append((w_i*w_j*mval, pauli_str))

    return coreset, G, H


def plot_coreset_graph(coreset_points, G):
    """
    Plot the coreset points on the x-y plane, and draw the networkX graph
    """
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

    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1,
                     ax=default_axes, pos=pos)


# Create the quantum circuit to implement the QAOA
def evolve_cost(c, angle, G, ising=False, qbit_map=None):
    """
    Implement the evolution according to the Cost Hamiltonian

    Parameters
    ----------
    c : QuantumCircuit
        The circuit instance to add gates to
    angle : float
        The current value of gamma
    G : NetworkX graph
        The graph for the current problem instance
    ising : bool
        If false, use CNOTs. If true, use controlled-u1 gates
    qbit_map : List
        If set to None, then the circuit will be constructed assuming an
        all-to-all connectivity. Otherwise, passing in a QPU connectivity list
        will result in a linear implementation of the QAOA relying on the
        fermionic swap networks discussed in (https://arxiv.org/abs/1711.04789)

    Returns
    -------
    Nothing, all gates added inplace
    """
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
        cover_b = [(idx-1, idx) for idx in range(2,len(qbit_map),2)]
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
    """
    Implement the evolution according to the Driver Hamiltonian

    Parameters
    ----------
    c : QuantumCircuit
        The circuit instance to add gates to
    angle : float
        The current value of beta

    Returns
    -------
    Nothing, all gates added inplace
    """
    c.rx(2*angle, c.qubits)


def gen_complete_qaoa_circ(P, gamma, beta, G, ising=False, topology=[]):
    """

    Parameters
    ----------
    P : int
        number of layers to apply
    gamma : List[float]
        list with length p, contains angle parameters for cost Hamiltonian
    beta : List[float]
        list with length p, contains angle parameters for driver Hamiltonian
    G : NetworkX graph
        graph representing the MAXCUT problem instance
    ising : bool
        If true, use Cu1 gates to implement cost evolution, otherwise use cx
    topology : List
        If set to empty list, then the circuit will be constructed assuming an
        all-to-all connectivity. Otherwise, passing in a QPU connectivity list
        will result in a linear implementation of the QAOA relying on the
        fermionic swap networks discussed in (https://arxiv.org/abs/1711.04789)

    Returns
    -------
    The QuantumCircuit implementing MAXCUT QAOA

        or, if a device topology is given

    The QuantumCircuit and the map from virtual to physical qubits
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
def cost_function_C(x, G):
    """
    Compute the value of the cost function.

    This function assumes a mapping between vertices and qubits:

            v0, v1, ..., vN -> q0, q1, ..., qN

    Parameters
    ----------
    x : List[int]
        Bitstring corresponding to a MAXCUT partitioning
    G : NetworkX Graph
        The current MAXCUT problem instance

    Returns
    -------
    float : the value of the cost function evaluated for the given x and G
    """
    E = G.edges()
    if( len(x) != len(G.nodes())):
        return np.nan

    C = 0;
    for edge in E:
        e1 = edge[0]
        e2 = edge[1]
        w = G[e1][e2]['weight']
        # Equation 6 in Overleaf Paper
        C = C + w*(1 - 2*(x[e1]*(1-x[e2]) + x[e2]*(1-x[e1])))

    return C


def reorder_bitstrings(P, nq, old_counts):
    """
    If the swap network is used to implement the cost evolution, then the
    measured bitstrings need to be reordered to account for the last SWAP layer
    that is removed in the network.

    Qiskit orders the measurement bitstrings little endian: qN,...,q1,q0
    This function applies the last SWAP layer of the network and then performs
    one final reversal so that the initial mapping between vertices and qubits:
            v0,v1,...,vN -> q0,q1,...,qN
    is preserved in the final bitstring counts:
            {q0q1...qN: 1024, q0q1...qN:2003, ...}
    """
    assert (P == 1),'Reordering only implemented for P=1'
    cover_a = [(idx-1, idx) for idx in range(1,nq,2)]
    cover_b = [(idx-1, idx) for idx in range(2,nq,2)]
    last_cover = [cover_b, cover_a][nq % 2]
    new_counts = {}
    for bitstr in old_counts.keys():
        bit_list = list(bitstr) # convert the bitstr to a list
        bit_list.reverse() # Qiskit orders the qubit bitstring little endian
        for i, j in last_cover:
            bit_list[j], bit_list[i] = bit_list[i], bit_list[j]
        bit_list.reverse() # reverse the final layer of the swap network
        new_bitstr = ''.join(bit_list)
        new_counts[new_bitstr] = old_counts[bitstr]
    return new_counts


def energy_landscape(P, step_size, shots, gammaLim, betaLim, G,
                     device_topology=[], device=None):
    """
    Use the QASM simulator to generate the energy landscape
    """
    backend = Aer.get_backend('qasm_simulator')
    a_gamma = np.arange(gammaLim[0], gammaLim[1], step_size)
    a_beta  = np.arange(betaLim[0], betaLim[1], step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)

    if device is None:
        reorder = False
        def execute_func(circ, backend, shots):
            return execute(circ, backend=backend, shots=shots)
    else:
        assert isinstance(device, IBMQBackend), 'device must be an IBMQBackend'
        reorder = True
        noise_model = NoiseModel.from_backend(device)
        basis_gates = device.configuration().basis_gates
        coupling_map= device.configuration().coupling_map
        props = device.properties()
        def execute_func(circ, backend, shots):
            return execute(circ, backend=backend, shots=shots,
                    basis_gates=basis_gates, noise_model=noise_model,
                    coupling_map=coupling_map, backend_properties=props)

    estC = []
    bitstrings = {}
    for gamma_list, beta_list in zip(a_gamma, a_beta):
        # scan across the gamma-beta plane
        C_row = []
        for gamma, beta in zip(gamma_list, beta_list):
            # TODO: this is still somewhat of a hack, because only one
            # value of gamma and beta can be passed to the circuit at a single time
            output = gen_complete_qaoa_circ(P, [gamma], [beta], G,
                                            topology=device_topology)

            if len(device_topology) == 0:
                qaoa_circ = output
            else:
                qaoa_circ, initial_layout = output

            # Simulate, either noisy or noiseless
            simulate = execute_func(qaoa_circ, backend, shots)
            counts = simulate.result().get_counts()

            # reorder the outputs if the swap network was used
            if reorder is True:
                counts = reorder_bitstrings(P, len(G.nodes), copy(counts))

            # Save the counts, indexed by the current gamma and beta values
            bitstrings['{:.3f}{:.3f}'.format(gamma, beta)] = counts

            # Evaluate the data from the simulator
            tot_C = 0
            for sample in list(counts.keys()):
                # use the sampled bitstring x to compute C(x)
                x = [int(num) for num in list(sample)]
                tmp_eng = cost_function_C(x,G)

                # compute the expectation value
                tot_C = tot_C + counts[sample]*tmp_eng

            # average the cost function over the number of shots
            avg_C = tot_C/shots
            C_row.append(avg_C)

        # save the entire row of avg_C values
        estC.append(C_row)

    # convert estC to a numpy array
    estC = np.array(estC)
    return estC, bitstrings


def plot_energy_landscape(step_size, gammaLim, betaLim, estC, bitstrings, shots,
                          coreset_points, savefigs=False):
    # Plot the energy landscape
    a_gamma = np.arange(gammaLim[0], gammaLim[1], step_size)
    a_beta  = np.arange(betaLim[0], betaLim[1], step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)

    fig = plt.figure()
    ax  = fig.gca(projection='3d')

    surf = ax.plot_surface(a_gamma, a_beta, estC, cmap=cm.coolwarm, linewidth=0,
                           antialiased=True)

    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel('Cost function')
    ax.zaxis.set_major_locator(LinearLocator(3))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.tight_layout()
    if savefigs: plt.savefig('qaoa_energylandscape.pdf')
    plt.show()
    plt.close()

    result = np.where(estC == np.amax(estC))
    a = list(zip(result[0],result[1]))[0]

    gamma  = a[0]*step_size;
    beta   = a[1]*step_size;
    optimal_counts = bitstrings['{:.3f}{:.3f}'.format(gamma, beta)]
    bs_list = [(key, optimal_counts[key]) for key in optimal_counts.keys()]
    bs_list = sorted(bs_list, key=lambda tup: tup[1], reverse=True)

    #The smallest parameters and the expectation can be extracted
    print('\n --- OPTIMAL PARAMETERS --- \n')
    print('The maximal expectation value (avg over {} shots) is:  C = {:.3f}'.format(shots, np.amax(estC)))
    print('This is attained for gamma = {0:.3f} and beta = {1:.3f}'.format(gamma,beta))
    print('The 4 most common partitionings produced at this point were:')
    print('{}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%, {}: {:.2f}%'.format(
                                         bs_list[0][0], 100*bs_list[0][1]/shots,
                                         bs_list[1][0], 100*bs_list[1][1]/shots,
                                         bs_list[2][0], 100*bs_list[2][1]/shots,
                                         bs_list[3][0], 100*bs_list[3][1]/shots))

    # Plot the centroids according to the optimal partitioning
    opt_partition = bs_list[0][0]
    c_plus = 'red'
    c_minus = 'blue'
    xx = [cp[1][0] for cp in coreset_points]
    yy = [cp[1][1] for cp in coreset_points]

    S_plus = [coreset_points[i] for i in range(len(opt_partition)) if opt_partition[i] == '1']
    xx_plus = [pt[1][0] for pt in S_plus]
    yy_plus = [pt[1][1] for pt in S_plus]
    plt.scatter(xx_plus, yy_plus, c=c_plus)

    S_minus = [coreset_points[i] for i in range(len(opt_partition)) if opt_partition[i] == '0']
    xx_minus = [pt[1][0] for pt in S_minus]
    yy_minus = [pt[1][1] for pt in S_minus]
    plt.scatter(xx_minus, yy_minus, c=c_minus)

    mu_plus = np.sum([point[0]*point[1] for point in S_plus], axis=0) / np.sum([point[0] for point in S_plus])
    mu_minus = np.sum([point[0]*point[1] for point in S_minus], axis=0) / np.sum([point[0] for point in S_minus])
    print('mu_plus:',mu_plus)
    print('mu_minus:',mu_minus)
    plt.scatter(mu_plus[0], mu_plus[1], c=c_plus, marker='*')
    plt.scatter(mu_minus[0], mu_minus[1], c=c_minus, marker='*')

    plt.hlines(0, np.amin(xx), np.amax(xx), ls='--')
    plt.vlines(0, np.amin(yy), np.amax(yy), ls='--')
    if savefigs: plt.savefig('qaoa_clustering.pdf')
    plt.show()
    plt.close()


def plot_partition(bitstring, coreset_points):
    c_plus = 'blue'
    c_minus = 'red'

    xx = [cp[1][0] for cp in coreset_points]
    yy = [cp[1][1] for cp in coreset_points]

    S_plus = [coreset_points[i] for i in range(len(bitstring)) if bitstring[i] == '1']
    xx_plus = [pt[1][0] for pt in S_plus]
    yy_plus = [pt[1][1] for pt in S_plus]
    plt.scatter(xx_plus, yy_plus, c=c_plus)

    S_minus = [coreset_points[i] for i in range(len(bitstring)) if bitstring[i] == '0']
    xx_minus = [pt[1][0] for pt in S_minus]
    yy_minus = [pt[1][1] for pt in S_minus]
    plt.scatter(xx_minus, yy_minus, c=c_minus)

    mu_plus = np.sum([point[0]*point[1] for point in S_plus], axis=0) / np.sum([point[0] for point in S_plus])
    mu_minus = np.sum([point[0]*point[1] for point in S_minus], axis=0) / np.sum([point[0] for point in S_minus])
    print('mu_plus:',mu_plus)
    print('mu_minus:',mu_minus)
    plt.scatter(mu_plus[0], mu_plus[1], c=c_plus, marker='*')
    plt.scatter(mu_minus[0], mu_minus[1], c=c_minus, marker='*')

    plt.hlines(0, np.amin(xx), np.amax(xx), ls='--')
    plt.vlines(0, np.amin(yy), np.amax(yy), ls='--')
    plt.show()
    plt.close()


def hardware_execution(device, P, gammaLim, betaLim, step_size, G, shots):
    """
    Run on a quantum processor
    """
    # create a savename based on time of execution
    d = datetime.datetime.today()
    savename = device.name()+'_'+d.strftime('%m-%d-%Y-%H%M')
    savedir  = 'HardwareRunPickles/'
    print('Running {}'.format(savename))

    # collect all gamma-beta pairs
    a_gamma = np.arange(gammaLim[0], gammaLim[1], step_size)
    a_beta  = np.arange(betaLim[0], betaLim[1], step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)
    angle_pairs = []
    for gamma_list, beta_list in zip(a_gamma, a_beta):
        for gamma, beta in zip(gamma_list, beta_list):
            angle_pairs.append((gamma, beta))

    # generate a qaoa circuit for each angle pair
    circuits = []
    for gamma, beta in angle_pairs:
        circ, initial_layout = gen_complete_qaoa_circ(P, [gamma], [beta], G,
                                   topology=device.configuration().coupling_map)
        circuits.append(circ)
    print('Generated {} circuits to cover gamma:{} -> {:.2f}, beta:{} -> {:.2f}'.format(
           len(circuits), *gammaLim, *betaLim))

    # get the max number of experiments-per-job for this device
    maxExps = device.configuration().to_dict()['max_experiments']
    print('Device {} supports {} experiments per job'.format(device.name(), maxExps))

    # break the list of circuits into chunks of this size
    batches = [circuits[i:i + maxExps] for i in range(0, len(circuits), maxExps)]
    print('Batched all {} circuits into {} total jobs'.format(len(circuits), len(batches)))

    # execute the jobs
    print('LAUNCHING JOBS')
    all_raw_counts = []
    for num, batch in enumerate(batches):
        print('Executing job {}'.format(num))
        job = execute(batch, backend=device, shots=shots,
                      initial_layout=initial_layout)
        job_monitor(job)
        raw_counts = job.result().get_counts()
        all_raw_counts.extend(raw_counts)

        # pickle the raw counts
        pklFile = open(savedir+savename+'_rawcounts_job{}.pickle'.format(num), 'wb')
        pickle.dump(raw_counts, pklFile)
        pklFile.close()
        print('raw counts saved')

    # reorder the raw counts
    good_counts = []
    for old_counts in all_raw_counts:
        new_counts = reorder_bitstrings(P, len(G.nodes), old_counts)
        good_counts.append(new_counts)

    # match the gamma-beta pairs with the execution counts
    execution_dict = {}
    for pair, run in zip(angle_pairs, good_counts):
        execution_dict['{:.3f}{:.3f}'.format(pair[0], pair[1])] = run

    # pickle the execution dictionary
    pklFile = open(savedir+savename+'_fullrun.pickle', 'wb')
    pickle.dump(execution_dict, pklFile)
    pklFile.close()
    print('execution dictionary saved')

    return execution_dict


def load_fullrun(picklename):
    pklFile = open(picklename, 'rb')
    fullrun = pickle.load(pklFile)
    pklFile.close()
    return fullrun


def compute_C_from_fullrun(fullrun, gammaLim, betaLim, step_size, G, shots):
    a_gamma = np.arange(gammaLim[0], gammaLim[1], step_size)
    a_beta  = np.arange(betaLim[0], betaLim[1], step_size)
    a_gamma, a_beta = np.meshgrid(a_gamma, a_beta)

    estC = []
    for gamma_list, beta_list in zip(a_gamma, a_beta):
        C_row = []
        for gamma, beta in zip(gamma_list, beta_list):
            counts = fullrun['{:.3f}{:.3f}'.format(gamma, beta)]

            # Evaluate the cost function
            tot_C = 0
            for sample in list(counts.keys()):
                # use the sampled bitstring x to compute C(x)
                x = [int(num) for num in list(sample)]
                tmp_eng = cost_function_C(x,G)

                # compute the expectation value
                tot_C = tot_C + counts[sample]*tmp_eng

            # average the cost function over the number of shots
            avg_C = tot_C/shots
            C_row.append(avg_C)

        # save the entire row of avg_C values
        estC.append(C_row)

    # convert estC to a numpy array
    estC = np.array(estC)
    return estC










