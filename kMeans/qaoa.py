import numpy as np
import itertools
import coreset

def get_ratios(Wplus, Wminus, taylor_order):
    """"Returns taylor expansions of Wplus/Wminus and Wminus/Wplus.
    Based on taylor expnasion of 1/x around x=0.5, see paper."""
    if taylor_order == 0:
        Wplus_Wminus = 1
        Wminus_Wplus = 1
    elif taylor_order == 1:
        Wplus_Wminus = 3 - 4 * Wminus / (Wplus + Wminus)
        Wminus_Wplus = 3 - 4 * Wplus / (Wplus + Wminus)
    elif taylor_order == 2:
        Wplus_Wminus = 8 * (Wminus / (Wplus + Wminus)) ** 2 - 12 * (Wminus / (Wplus + Wminus)) + 5
        Wminus_Wplus = 8 * (Wplus / (Wplus + Wminus)) ** 2 - 12 * (Wplus / (Wplus + Wminus)) + 5
    elif taylor_order == 'inf':
        Wplus_Wminus = Wplus / Wminus
        Wminus_Wplus = Wminus / Wplus

    return Wplus_Wminus, Wminus_Wplus


def qaoa_bound(coreset_vectors, coreset_weights, data_vectors, taylor_order):    
    m = len(coreset_vectors)
    W = sum(coreset_weights)

    bitstrings = list(itertools.product([-1, 1], repeat=m))
    best_edge_sum, best_bitstring, best_cost = -np.inf, None, np.inf
    
    np.random.shuffle(bitstrings)
    for bits in bitstrings:
        if -1 not in bits or 1 not in bits:  # skip assignments that only have one cluster
            continue
        
        Wplus, Wminus = 0, 0
        for i, bit in enumerate(bits):
            if bit == +1:
                Wplus += coreset_weights[i]
            else:
                Wminus += coreset_weights[i]
        Wplus_Wminus, Wminus_Wplus = get_ratios(Wplus, Wminus, taylor_order)
            
        edge_sum = 0
        for i in range(m):
            if bits[i] == -1:
                edge_sum += Wplus_Wminus * (coreset_weights[i] ** 2) * (np.dot(coreset_vectors[i], coreset_vectors[i]))
            else:
                edge_sum += Wminus_Wplus * (coreset_weights[i] ** 2) * (np.dot(coreset_vectors[i], coreset_vectors[i]))
                
        for i in range(m):
            for j in range(i + 1, m):
                term = 2 * coreset_weights[i] * coreset_weights[j] * np.dot(coreset_vectors[i], coreset_vectors[j])
                if bits[i] != bits[j]:
                    edge_sum += -1 * term
                elif bits[i] == -1:
                    edge_sum += Wplus_Wminus * term
                elif bits[i] == +1:
                    edge_sum += Wminus_Wplus * term

        if edge_sum > best_edge_sum:
            best_edge_sum = edge_sum
            best_bitstring = bits
            
            minus1_center, plus1_center = np.zeros_like(coreset_vectors[0]), np.zeros_like(coreset_vectors[0])
            minus1_weight, plus1_weight = 0, 0
        
            for bit, v, w in zip(best_bitstring, coreset_vectors, coreset_weights):
                if bit == -1:
                    minus1_center += w * v; minus1_weight += w
                elif bit == 1:
                    plus1_center += w * v; plus1_weight += w
            minus1_center /= minus1_weight
            plus1_center /= plus1_weight

            best_cost = coreset.get_cost(data_vectors, [minus1_center, plus1_center])
            print(best_cost, best_bitstring, Wplus/Wminus, Wplus * Wminus * np.linalg.norm(minus1_center - plus1_center) ** 2)

    return best_cost
