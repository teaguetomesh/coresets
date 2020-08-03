import kmeans_qaoa as kq

P = 1
nq = 4
counts = {'11010': 1, '01000': 2, '00100':3}
print('old counts:', counts)
new_counts = kq.reorder_bitstrings(P, nq, counts)
print('new counts:', new_counts)
