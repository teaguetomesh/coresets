import numpy as np
import kmeans_qaoa as kq
coreset = [(1, np.array([-2,0])), (1, np.array([-1,0])), (1, np.array([1,0])), (1, np.array([2,0]))]
coreset_points, G, H = kq.gen_coreset_graph(coreset=coreset, metric='dot')
device_topology = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]
P, gamma, beta = 1, 1, 1
cnot_circ, initial_layout = kq.gen_complete_qaoa_circ(P, [gamma], [beta], G,
                        ising=False, topology=device_topology)
print('Depth:', cnot_circ.depth())
print('Gates:', cnot_circ.count_ops())
print(cnot_circ.draw(output='text', fold=180, scale=0.5))
