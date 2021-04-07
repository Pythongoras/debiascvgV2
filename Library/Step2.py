import numpy as np
from math import exp, log
from collections import defaultdict
from copy import deepcopy




#######
# Monotone Cone and Positive Monotone Cone
#######
def find_v_mnt(beta, n, param_set=[]):
    """
    Find the optimal v in step 2 for monotone cone and positive monotone cone.
    
    @param beta np.array(p, ): the beta hat obtained by the first step.
    @param n int: sample size.
    @param param_set list[]: an empty list, since we don't need extra param this case.
    
    @return np.array(p, ): v
    
    Test:
    n, p = 100, 250
    beta = np.array([0]*int(0.7*p) + [1]*(p-int(0.7*p)))
    X = np.random.normal(size = (n,p)) @ Sigma_sqrt
    Y = X @ beta + np.random.normal(0, noise_sd, n)
    beta_hat = mnt_reg(X, Y)
    v, v_const = find_v_mnt(beta_hat, n)
    """
    proj_list = find_all_mnt_proj(beta)
    
    opt_value = float('inf')
    opt_l = -1
    
    for i, v in enumerate(proj_list):
        cur_value = sum((beta-v)**2)**0.5 + ( (i+1)/n * log(np.exp(1)*len(beta)/(i+1)) )**0.5
        if cur_value < opt_value:
            opt_value = cur_value
            opt_l = i
    return proj_list[opt_l]




#######
# Non-negative Orthant
#######
def find_v_nonneg(beta, n, param_set=[]):
    """
    Find the optimal v in step 2 for non-negative orthant.
    
    @param beta np.array(p, ): the beta hat obtained by the first step.
    @param n int: sample size.
    @param param_set list[]: an empty list, since we don't need extra param this case.
    
    @return np.array(p, ): v
    """
    p = len(beta)
    v_opt = np.zeros(p)
    value_opt = float('inf')
    
    for s in range(p):
        vs = deepcopy(beta)
        np.put(vs, np.argsort(beta)[:s], 0)
        value_s = np.sum((beta-vs)**2)**0.5 + ((p - 0.5 * s) / n)**0.5
        if value_s < value_opt:
            v_opt = vs
            value_opt = value_s
    return v_opt




#######
# LASSO
#######
def find_v_lasso(beta, n, param_set):
    """
    Find the optimal v in step 2 for LASSO.
    
    @param beta np.array(p, ): the beta hat obtained by the first step.
    @param n int: sample size.
    @param param_set list[float]: contains the radius of the l1 ball.

    @return np.array(p, ): v
    """
    sb = len(np.where(abs(beta) > 0)[0])
    p = len(beta)
    v_opt = np.zeros(p)
    value_opt = float('inf')
    
    for i in range(1, sb+1):
        vs = find_sparse_proj(beta, param_set[0], i)
        value_s = np.sum((beta-vs)**2)**0.5 + (i/n * log(p/i))**0.5
        if value_s < value_opt:
            v_opt = vs
            value_opt = value_s
    return v_opt




#######
# SLOPE and Square-root SLOPE
#######
def find_v_slope(beta, n, param_set):
    """
    Find the optimal v in step 2 for SLOPE.
    
    @param beta np.array(p, ): the beta obtained by the first step.
    @param n: the sample size.
    @param param_set list[float, float]: contains the constant C larger than the const in upper bound;
                                         the upper bound of number of non-sparse coordinates.
    
    @return np.array(p, ): v
    """
    p = len(beta)
    C, su = param_set[0], param_set[1]
    
    sort_idx = np.argsort(abs(beta))[::-1]
    non_zero_idx, zero_idx = sort_idx[:su], sort_idx[su:]
    
    non_zero_mass = np.clip(C**2 * su * log(2*p*exp(1)/su) / n - np.sum(beta[zero_idx]**2), 
                            su, None)  # clip below to make sure it's sufficiently greater than zero
    c = (non_zero_mass / su)**0.5
    
    v = np.zeros(len(beta))
    for i in non_zero_idx:
        v[i] = beta[i] + np.sign(beta[i]) * c
    return v




#######
# Auxiliary Functions
#######
def all_knots(beta):
    """
    Find all the knots of constant pieces in beta. Every knot is represented by the right index of it.
    Always include the knot at 0 and at len(beta)).
    @param beta np.array(p, ): the vector to find knots.
    @return List[int]: a list of indices.
    """
    knots_list = []
    for i in range(1, len(beta)):
        if beta[i-1] < beta[i]: knots_list.append(i)
    return [0] + knots_list + [len(beta)]
    
    
def find_all_mnt_proj(beta):
    """
    Find a list of projections to monotone cone with constant l pieces, l = 1,...,beta_const.
    
    @param beta np.array(p, ): the vector to be projected.
    @return List[numpy.array]: a list of all the projections.
    
    Test:
    bb = np.array([1,1,1,1,1,1,8,8,9]) * 1.0
    bb_projs = find_all_mnt_proj(bb)
    """
    # all knots in beta
    beta_knots = all_knots(beta)
    # if beta only has 1 constant piece, return itself
    if len(beta_knots) == 2: return [beta]
    
    # Loss[(l, j)]: loss of fitting by mean between the l-th and j-th knot (for all pairs of l < j, the 0-th is the knot at 0, n-th is at len(beta).
    # Compute partial sums to save computation. S[j] is the partial sum up to j-th knot; SS[j] the partial sum of square up to j-th knot.
    S, SS = [0], [0]
    for i in range(1, len(beta_knots)):
        S.append( S[-1] + sum(beta[beta_knots[i-1]:beta_knots[i]]) )
        SS.append( SS[-1] + sum(beta[beta_knots[i-1]:beta_knots[i]]**2) )
    # compute the loss by partial sums
    Loss = defaultdict(float)
    for l in range(len(beta_knots)):
        for j in range(l+1, len(beta_knots)):
            Loss[(l, j)] = SS[j] - SS[l] - (S[j]-S[l])**2 / (beta_knots[j]-beta_knots[l])
    
    # Initializations
    proj_list = []  # the result to return
    T_loss = defaultdict(float)  # T_loss[(k, j)] is the loss for k-piece monotone fit btw 0-th and j-th knot.
    for j in range(1, len(beta_knots)):
        T_loss[(1, j)] = Loss[(0, j)]
    left_knot = defaultdict(int)  # left_knot[(k, j)] is the knot(not the index of beta) at the left of j-th knot, with k-piece monotone fit up to j-th knot.
    knots = defaultdict(int)  # left_knot[(k, j)] is the j-th knot(not the index of beta) of the k-piece monotone fit for the whole vector.
    
    # DP
    for k in range(2, len(beta_knots)):  # k in [2, num of const in beta]. Num of const in beta = len(beta_knots) - 1.
        # k-piece monotone fit up to j-th positon
        for j in range(k, len(beta_knots)):
            # find the last knot befor j
            fit_loss = float('inf')
            left_knot[(k, j)] = -1
            for l in range(k-1, j):
                if fit_loss > (T_loss[(k-1, l)] + Loss[l, j]):
                    fit_loss = T_loss[(k-1, l)] + Loss[l, j]
                    left_knot[(k, j)] = l
            # get the loss of this fit
            T_loss[(k, j)] = T_loss[(k-1, left_knot[(k, j)])] + Loss[left_knot[(k, j)], j]
        # the knots of k-piece monotone fit for the whole vector
        knots[(k, 0)], knots[(k, k)] = 0, len(beta_knots) - 1
        for j in range(k-1, 0, -1):
            knots[(k, j)] = left_knot[(j+1, knots[(k, j+1)])]
            
        # construct k-piece proj from knots
        proj = deepcopy(beta)
        for j in range(1, k+1):
            idx_start, idx_end = beta_knots[knots[(k, j-1)]], beta_knots[knots[(k, j)]]
            np.put(proj, np.arange(idx_start, idx_end), np.mean(proj[idx_start:idx_end]))
        proj_list.append(proj)
        
    # Append the proj with one constant piece
    proj_list = [np.ones(len(beta)) * np.mean(beta)] + proj_list
    return proj_list


def find_sparse_proj(beta, l1_bound, s):
    """
    Find the projection of beta onto the s-sparse vector set with an l1_bound.
    
    @param beta np.array: the vector to be projected.
    @param l1_bound: the l1 bound of the projection.
    @param s float: the cardinality of the projection.
    """
    p = len(beta)
    
    # find the index of the most significant s coordinates
    idx_largest = np.argsort(abs(beta))[::-1][:s]
    
    # assign zero to other indices
    proj = np.zeros(p)
    for i in idx_largest:
        proj[i] = beta[i]
    
    # add the premium to the non-zero coordinates of proj
    prem = np.zeros(p)
    prem[idx_largest] = 1.0
    proj = proj + np.sign(proj) * prem * (l1_bound - np.sum(abs(proj))) / s

    return proj
