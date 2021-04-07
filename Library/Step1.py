import numpy as np

from sklearn.isotonic import IsotonicRegression

from scipy.optimize import nnls

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False




#######
# Monotone Cone and Positive Monotone Cone
#######
def solve_beta_mnt(X, Y, param_set=[False], learning_rate=0.5, stop_criteria=10**-4):
    """
    Solve the beta for monotone cone and positive monotone cone constraint.
    
    @param X np.array(n, p): design matrix.
    @param Y np.array(n, ): response.
    @param param_set list[bool]: [True] if K is a positive mnt cone, otherwise [False].
    
    @param learning_rate float: the step size is learning_rate/i.
    @param stop_criteria float: the stop criteria.
    
    @return np.array(p, ): the coefficient estimation by monotone or pos monotone regression.
    
    Test:
    n, p = 100, 250
    Sigma_sqrt = np.eye(p)
    noise_sd = 3
    beta = np.array([0]*int(0.7*p) + [1]*(p-int(0.7*p)))
    X = np.random.normal(size = (n,p)) @ Sigma_sqrt
    Y = X @ beta + np.random.normal(0, noise_sd, n)
    
    beta_hat = solve_beta_mnt(X, Y)
    print(beta_hat)
    """
    n = len(Y)
    p = X.shape[1]
    iso_order = np.arange(p)
    
    # initialize
    beta_prev = np.ones(p)
    beta = np.random.normal(size = X.shape[1])
    
    # gradient descent
    i = 0.0  # iteration number
    while abs(sum((Y-X@beta)**2)/n - sum((Y-X@beta_prev)**2)/n) > stop_criteria:
        i += 1
        
        beta_grad = -2/n * (X.T@Y - X.T@X@beta)
        beta_prev = beta
        # update beta with projection
        beta = beta - (1/i) * learning_rate * beta_grad
        beta = IsotonicRegression().fit_transform(iso_order, beta)
        # if it's pos mnt cone, assign zero to negative coordinates
        if param_set[0]: beta = np.where(beta > 0, beta, 0)
            
    return beta




#######
# Non-negative Orthant
#######
def solve_beta_nonneg(X, Y, param_set=[]):
    """
    Solve the beta for non-negative orthant constraint.
    
    @param X np.array(n, p): design matrix.
    @param Y np.array(n, ): response.
    @param param_set list[]: an empty list, since we don't need extra param this case.
    
    @return np.array(p, ): the coefficient estimation by non-neg regression.
    """
    return nnls(X, Y)[0]




#######
# LASSO
#######
def solve_beta_lasso(X, Y, param_set):
    """
    Solve the constrained lasso with l1 norm bound t.
    
    @param X np.array(n, p): design matrix.
    @param Y np.array(n, ): response.
    @param param_set list[float]: constains the l1 norm bound
    
    @return np.array(p, ): the coefficient estimation by constrained lasso.
    """
    p = X.shape[1]
    cov_mat = X.T @ X
    cov_mat = np.concatenate((cov_mat, cov_mat), 0)
    xy = X.T @ Y * (-2.0)
    
    # QP to solve concatenated beta
    P = matrix(np.concatenate((cov_mat, cov_mat), 1), tc='d')
    q = matrix(np.concatenate((xy, xy), 0), tc='d')
    G = matrix(np.diag(np.concatenate((-1.0*np.ones(p), np.ones(p)), 0)), tc='d')
    h = matrix(np.zeros(2*p), tc='d')
    A = matrix(np.concatenate((np.ones(p), -1.0*np.ones(p)), 0).reshape((1,2*p)), tc='d')
    b = matrix(param_set[0], tc='d')
    
    # Get the solution of QP
    beta_bundle = np.array(solvers.qp(P,q,G,h,A,b)['x'])
    
    # Reconstruct beta, assign zero to the very small coordinates
    beta = beta_bundle[:p] + beta_bundle[p:]
    beta = np.where(beta > 10**-4, beta, 0)
    return np.squeeze(beta)


########
## The SLOPE and square-root SLOPE are defined in notebook since they need matlab engine.
########