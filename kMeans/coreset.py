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


def BFL16(P, B, m):
    """Algorithm 2 in https://arxiv.org/pdf/1612.00889.pdf [BFL16].
    Per Table 1, the coreset size is O(k^2 logk / eps^3) or O(dk logk / eps^2)
    Note that D(p, q) appears to be the squared distance dist(p, q) ** 2 (top of pg 23)
    We're using the best of N k-means++ initializations for the (alpha, beta) approximation.
    P is the list of points, B is a list of k-means++ cluster center initializations.
    """

    num_points_in_clusters = {i: 0 for i in range(len(B))}
    sum_distance_to_closest_cluster = 0
    for p in P:
        _, closest_index = dist_to_B(p, B, return_closest_index=True)
        num_points_in_clusters[closest_index] += 1
        sum_distance_to_closest_cluster += dist_to_B(p, B) ** 2

    Prob = np.zeros(len(P))
    for i, p in enumerate(P):
        _, closest_index = dist_to_B(p, B, return_closest_index=True)
        Prob[i] += dist_to_B(p, B) ** 2 / (2 * sum_distance_to_closest_cluster)
        Prob[i] += 1 / (2 * len(B) * num_points_in_clusters[closest_index])

    assert 0.999 <= sum(Prob) <= 1.001, 'sum(Prob) = %s; the algorithm should automatically '\
            'normalize Prob by construction' % sum(Prob)
    chosen_indices = np.random.choice(len(P), size=m, p=Prob)
    weights = [1 / (m * Prob[i]) for i in chosen_indices]

    return [P[i] for i in chosen_indices], weights


def get_cost(data_vectors, B):
    cost = 0
    for x in data_vectors:
        cost += dist_to_B(x, B) ** 2
    return cost
