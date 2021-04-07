import os
os.chdir('/Users/yufei/Documents/2-CMU/DebiasingCvxConstrained/Code/Library')

from Step2 import all_knots

import numpy as np
from math import log

from sklearn.isotonic import IsotonicRegression



# learning_rate=0.01, stop_criteria=10**-4
def solve_omega(v, n, Sig_hat, debias_idx, param_set, learning_rate=0.01, stop_criteria=10**-4):
    """
    Solve the optimization in step 3.
    Run enough time: sum((omega - omega_prev)**2)**0.5 <= stop_criteria.
    
    @param v np.array(p, ): the v found in step 2.
    @param n int: sample size.
    @param Sig_hat np.array(p, p): the sample covariance matrix.
    @param debias_idx: the index of the coordinate which will be debiased.
    
    @param param_set list(function, function, function): contains the function to compute threshold lambda; 
                                                         function to compute proj to tangent cone;
                                                         function to compute proj to neg tangent cone.
    
    @param learning_rate: step size of the optimization is learning_rate/i. Don't change since tuned to optimal.
    @param stop_criteria: stop criteria of the optimization. Don't change since tuned to optimal.
    
    @return omega np.array(p, ): the vector used to debiase the debias_idx coordiante.
    """
    p = len(v)
    ej = np.zeros(p)
    ej[debias_idx] = 1.0
    
    # Compute the threshold of the constraint Q
    lbd = np.linalg.norm(Sig_hat) * param_set[0](v) / n**0.5
    
    # Initialize omega
    omega_prev = np.ones(p)
    omega = np.random.normal(size = p)
    
    # Subgradient descent
    i = 0.0 # iteration number
    while sum((omega - omega_prev)**2)**0.5 > stop_criteria:
        i += 1
        
        proj_pos = param_set[1](np.matmul(Sig_hat,omega) - ej, v)
        proj_neg = param_set[2](np.matmul(Sig_hat,omega) - ej, v)
        
        dot_pos, dot_neg = sum(proj_pos**2)**0.5, sum(proj_neg**2)**0.5
        
        if max(dot_pos, dot_neg) <= lbd:
            omega_grad = np.matmul(Sig_hat, omega)
        else:
            omega_grad = np.matmul( Sig_hat, (proj_pos if dot_pos > dot_neg else proj_neg) )
        omega_grad = omega_grad / sum((omega_grad)**2)**0.5
        
        # update omega_prev and omega
        omega_prev = omega
        omega = omega - (learning_rate / i) * omega_grad
        
    return omega




#######
# Auxiliary Functions
#######

# functions to compute gaussian complexity
def gw_mnt(v):
    """
    Compute the upper bound of gw(T_K(v)) when K is a monotone cone.
    Specifically, (l*log(ep/l))**0.5 where l is the # of constant pieces.
    
    @param v np.array(p, ): the v got in step 2.
    
    @return int: the upper bound of gw(T_K(v))
    """
    p = len(v)
    
    # detect the number of constant pieces
    v_const = 1
    if v[0] > v[p-1]:
        for i in range(p-1):
            if v[i] > v[i+1]: v_const += 1
    elif v[0] < v[p-1]:
        for i in range(p-1):
            if v[i] < v[i+1]: v_const += 1

    return (v_const * log(np.exp(1)*p/v_const))**0.5


def gw_nonneg(v):
    """
    Compute the upper bound of gw(T_K(v)) when K is a nonnegative orthant.
    Specifically, see Lemma 4.8.
    
    @param v np.array(p, ): the v got in step 2.
    
    @return int: the upper bound of gw(T_K(v)).
    """
    p = len(v)
    return ( p - 0.5 * (p - len(np.where(abs(v)>0)[0])) )**0.5
    
    
def gw_l1(v):
    """
    Compute the upper bound of gw(T_K(v)) when K is a nonnegative orthant and v has s non-zero coordinates.
    Specifically, (s*log(p/s))**0.5.
    
    @param v np.array(p, ): the v got in step 2.
    
    @return int: the upper bound of gw(T_K(v)).
    """
    p = len(v)
    s = len(np.where(abs(v)>0)[0])
    return (s * log(p/s))**0.5


##########
# Functions to compute projection to tangent cone
def proj_mnt_tan_cone(u, v, neg=False):
    """
    Project u onto the tangent cone or negative tangent cone of monotone cone at v.
    
    @param u np.array(p, ): the vector to calculate projection from.
    @param v np.array(p, ): the vector at whom the tangent cone forms.
    @param neg bool: True if in the proj to the neg tan cone, otherwise False
    
    @return unnormalized projection of u.
    """
    # Find all knots of v
    knots_list = all_knots(v)

    # Do isotonic regression of u on every constant piece
    for i in range(1, len(knots_list)):
        u_piece = u[knots_list[i-1]:knots_list[i]]
        if neg: iso_order = np.arange(len(u_piece),0, -1)  # The negative of tangent cone consists of mnt decreasing cones
        else: iso_order = np.arange(len(u_piece))
        u[knots_list[i-1]:knots_list[i]] = IsotonicRegression().fit_transform(iso_order, u_piece)
    return u


def proj_mnt_neg_tan_cone(u, v):
    """
    Project u onto the negative tangent cone of monotone cone at v.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    
    @return unnormalized projection of u.
    """
    return proj_mnt_tan_cone(u, v, neg=True)
    
    
def proj_posmnt_tan_cone(u, v, neg=False):
    """
    Project u onto the tangent cone or negative tangent cone of positive monotone cone at v.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    @param neg bool: True if in the proj to the neg tan cone, otherwise False
    
    @return unnormalized projection of u.
    """
    # Find all knots of v
    knots_list = all_knots(v)

    # Do isotonic regression of u on every constant piece.
    for i in range(1, len(knots_list)):
        u_piece = u[knots_list[i-1]:knots_list[i]]
        if neg: iso_order = np.arange(len(u_piece),0, -1)  # The negative of tangent cone consists of mnt decreasing cones
        else: iso_order = np.arange(len(u_piece))
        mnt_proj = IsotonicRegression().fit_transform(iso_order, u_piece)
        # if the first constant piece is 0, project to positive monotone cone.
        if i == 1 and u_piece[0] == 0:
            mnt_proj = np.where(mnt_proj > 0, mnt_proj, 0)
        # update u
        u[knots_list[i-1]:knots_list[i]] = mnt_proj
    return u


def proj_posmnt_neg_tan_cone(u, v):
    """
    Project u onto the negative tangent cone of positive monotone cone at v.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    
    @return unnormalized projection of u.
    """
    return proj_posmnt_tan_cone(u, v, neg=True)


def proj_nonneg_tan_cone(u, v):
    """
    Project u onto the tangent cone of positive monotone cone at v.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    
    @return unnormalized projection of u.
    """
    for i in range(len(v)):
        if v[i] == 0: u[i] = max(u[i], 0)
    return u


def proj_nonneg_neg_tan_cone(u, v):
    """
    Project u onto the negative tangent cone of positive monotone cone at v.
    See eq. 8 in paper.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    
    @return unnormalized projection of u.
    """
    return -proj_nonneg_tan_cone(u, -v)


def proj_l1_tan_cone(u, v):
    """
    Project u onto the tangent cone of l1 ball with diameter ||v||_1 at v.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    
    @return unnormalized projection of u.
    """
    idx_nonzero = np.where(v != 0)[0]
    idx_zero = np.where(v == 0)[0]

    f_nonzero = lambda x: np.sum((u[idx_nonzero] - x * np.sign(v[idx_nonzero]))**2)
    shrink = lambda x: np.sum((np.sign(u[idx_zero]) * np.clip(abs(u[idx_zero])-x, 0, None))**2)
    f = lambda x: f_nonzero(x) + shrink(x)
    
    # one-dimensional optimization to get t
    t = gss(f,
            0,  # lower bound is 0
            max(abs(u)),  # upper bound is max|u_i|. After t >= max|u_i|, the function will be increasing with t.
            tol=1e-5)
    
    # use t to get the projection
    proj = np.zeros(len(u))
    for i in idx_nonzero:
        proj[i] = t * np.sign(v[i])
    for i in idx_zero:
        proj[i] = u[i] if abs(u[i]) <= t else t * np.sign(u[i])
    
    # moreau's decomposition
    proj = u - proj
    return proj
    
    
def proj_l1_neg_tan_cone(u, v):
    """
    Project u onto the negative tangent cone of l1 ball with diameter ||v||_1
    at v. It's actually the tangent cone of l1 ball with bound ||v||_1 at -v.
    
    @param u: the vector to calculate projection from.
    @param v: the vector at whom the tangent cone forms.
    
    @return unnormalized projection of u.
    """
    return proj_l1_tan_cone(u, -v)
    
    
# The function to implement golden section search.
# Obtained from https://en.wikipedia.org/wiki/Golden-section_search#Probe_point_selection
gr = (5**0.5 + 1) / 2 # the golden ratio.
def gss(f, a, b, tol=1e-5):
    """Golden section search
    to find the minimum of f on [a,b]
    f: a strictly unimodal function on [a,b]

    Example:
    >>> f = lambda x: (x-2)**2
    >>> x = gss(f, 1, 5)
    >>> print("%.15f" % x)
    2.000009644875678
    """
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2
