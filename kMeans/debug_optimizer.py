import numpy as np
import kmeans_qaoa as kq
from qiskit import IBMQ

coreset = [(1, np.array([-2,0])), (1, np.array([-1,0])), (1, np.array([1,0])), (1, np.array([2,0]))]
coreset_points, G, H = kq.gen_coreset_graph(coreset=coreset, metric='dot')
device_topology = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]]
provider = IBMQ.load_account()
rome = provider.get_backend('ibmq_rome')

P = 1
shots = 8192
num_params = 2
#init_params = [2.98, 2.55]
init_params = None

opt_params, opt_cost = kq.optimize_qaoa(init_params, num_params, shots, P, G,
        device_topology, device=rome)
