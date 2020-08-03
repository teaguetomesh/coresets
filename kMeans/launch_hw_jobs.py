import kmeans_qaoa as kq
from qiskit import IBMQ
import numpy as np

provider = IBMQ.load_account()
rome = provider.get_backend('ibmq_rome')

coreset = [(1, np.array([-2,0])), (1, np.array([-1,0])), (1, np.array([1,0])), (1, np.array([2,0]))]
coreset_points, G, H = kq.gen_coreset_graph(coreset=coreset, metric='dot')

P = 1
gammaLim = [0, np.pi]
betaLim = [0, np.pi]
step_size = 0.1
shots = 8192
kq.hardware_execution(rome, P, gammaLim, betaLim, step_size, G, shots)
