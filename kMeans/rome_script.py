import numpy as np
from qiskit import IBMQ
import kmeans_qaoa as kq
#coreset = [(1, np.array([-2,0])), (1, np.array([-1,0])), (1, np.array([1,0])), (1, np.array([2,0]))]
coreset = None
coreset_points, G, H = kq.gen_coreset_graph(coreset=coreset, metric='dot')
provider = IBMQ.load_account()
rome = provider.get_backend('ibmq_rome')

P, gamma, beta = 1, 0.65, 1.3
cnot_circ, initial_layout = kq.gen_complete_qaoa_circ(P, [gamma], [beta], G,
                        ising=False, topology=rome.configuration().coupling_map)
print('Depth:', cnot_circ.depth())
print('Gates:', cnot_circ.count_ops())
print(cnot_circ.draw(output='text', fold=180, scale=0.5))
