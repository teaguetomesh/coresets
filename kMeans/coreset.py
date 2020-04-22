import numpy as np

def dist_to_B(x, B, return_closest_index=False):
    min_dist = np.inf
    closest_index = -1
    for i, b in enumerate(B):
        dist = np.linalg.norm(x - b)
        if dist < min_dist:
            min_dist = dist
            closest_index = i
    if return_closest_index:
        return min_dist, closest_index
    return min_dist


def Algorithm1(data_vectors, k):
    "D^2-sampling"
    B = []
    B.append(data_vectors[np.random.choice(len(data_vectors))])
    
    for _ in range(k - 1):
        p = np.zeros(len(data_vectors))
        for i, x in enumerate(data_vectors):
            p[i] = dist_to_B(x, B) ** 2
        p = p / sum(p)
        B.append(data_vectors[np.random.choice(len(data_vectors), p=p)])
    
    return B


# implement Algorithm 1/2 of https://arxiv.org/pdf/1703.06476.pdf
# Alg 2 requires m = Ω([dk^3 log k + k^2 log(1/delta)] / eps^2)
def Algorithm2(data_vectors, k, B, m):
    alpha = 16 * (np.log2(k) + 2)
    

    B_i_totals = [0] * k
    B_i = [np.empty_like(data_vectors) for _ in range(k)]
    for x in data_vectors:
        _, closest_index = dist_to_B(x, B, return_closest_index=True)
        B_i[closest_index][B_i_totals[closest_index]] = x
        B_i_totals[closest_index] += 1        
        
    # note that there is ambiguity between Lemma 2.2 and Algorithm 2
    # for the following few lines. E.g. should the distance be squared?
    c_phi = sum([dist_to_B(x, B) ** 2 for x in data_vectors]) / len(data_vectors)

    p = np.zeros(len(data_vectors))
    
    sum_dist = {0: 0, 1: 0}
    for i, x in enumerate(data_vectors):
        dist, closest_index = dist_to_B(x, B, return_closest_index=True)
        sum_dist[closest_index] += dist ** 2
    
    for i, x in enumerate(data_vectors):
        p[i] = 2 * alpha * dist_to_B(x, B) ** 2 / c_phi
        
        _, closest_index = dist_to_B(x, B, return_closest_index=True)
        p[i] += 4 * alpha * sum_dist[closest_index] / (B_i_totals[closest_index] * c_phi)

        p[i] += 4 * len(data_vectors) / B_i_totals[closest_index]
    p = p / sum(p)

            
    chosen_indices = np.random.choice(len(data_vectors), size=m, p=p)
    weights = [1 / (m * p[i]) for i in chosen_indices]
    
    return [data_vectors[i] for i in chosen_indices], weights


def get_cost(data_vectors, B):
    cost = 0
    for x in data_vectors:
        cost += dist_to_B(x, B) ** 2
    return cost
