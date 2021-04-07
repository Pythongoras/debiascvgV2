import numpy as np

    


"""
Implement the experiments for different cases.

@param N int: number of repetitions
@param n int: sample size
@param p int: dimension
@param Sigma_sqrt np.array(p, p): square root of population cov matrix
@param noise_sd float: the standard deviation of noise
@param debias_idx int: the index of coordinate to debias

@param param_set namedtuple(list, lsit, list): the set of parameters needed in each step
@param beta_gen_func function: the function to generate beat^*
@param step1_func function: the function to solve beta_hat in step1
@param step2_func function: the function to solve v in step2
@param step3_func function: the function to solve omega in step3

@return np.array(N, 1), np.array(N, 1): the collection of debiased beta_j;
                                        the collection of non-debiased beta_j
"""
def exp_func(N,
             n,
             p, 
             Sigma_sqrt, 
             noise_sd, 
             debias_idx,
             param_set, 
             beta_gen_func, 
             step1_func, 
             step2_func, 
             step3_func):
    z = []
    z_biased = []
    
    for i in range(N):
        print("iter:", i)
        
        # generate data
        beta = beta_gen_func(p)
#         print(beta)
        
        X = np.matmul(np.random.normal(size = (n,p)), Sigma_sqrt)
        Y = np.matmul(X, beta) + np.random.normal(0, noise_sd, n)
        
        # sample split
        X1, X2 = X[:n//2], X[n//2:]
        Y1, Y2 = Y[:n//2], Y[n//2:]
        Sigma_hat = np.matmul(X2.T, X2) / n
        
        # step 1: compute beta_hat
        beta_hat = step1_func(X1, Y1, param_set.step1)
        print( "The L2 error: ", sum((beta_hat-beta)**2)**0.5 )
        
        # step 2: get v
        v = step2_func(beta_hat, n, param_set.step2)
#         print("||v-beta*||: ", sum((v-beta)**2)**0.5)
        
        # step 3: get omega and debiase beta_hat
        omega = step3_func(v, n, Sigma_hat, debias_idx, param_set.step3)
        beta_d = v[debias_idx] + 1.0/n * np.matmul( np.matmul(omega.T,X2.T), Y2-np.matmul(X2,v) )
#         print( "Delta_j: ", np.matmul(np.matmul(omega.T,Sigma_hat)-np.zeros(p).T, v-beta) )
        
        # standardize the debiased beta_j, append to z
        z_var = np.matmul( omega.T, np.matmul(Sigma_hat,omega) ) * np.var(Y1-np.matmul(X1,beta_hat))
        z.append( (beta[debias_idx]-beta_d) / (z_var/(0.5*n))**0.5 )  # N(0,1)
#         print("\sig||\eta\T\Sig||: ", z_var**0.5)
        
        # append the biased beta_j to z_biased
        z_biased.append(beta[debias_idx]-beta_hat[debias_idx])
    
    z_biased = np.array(z_biased) / np.var(z_biased)**0.5  # N(0,1)
    return z, z_biased




#######
# Auxiliary Functions
#######
def beta_gen_mnt(p):
    """
    Generate the linear model coefficient in monotone cone case.
    
    @param p int: dimension.
    
    @return np.array(p,1): the coefficient.
    """
    return np.array([-1.0]*int(0.7*p) + [1.0]*(p-int(0.7*p)))


def beta_gen_posmnt(p):
    """
    Generate the linear model coefficient in positive monotone cone case.
    
    @param p int: dimension.
    
    @return np.array(p,1): the coefficient.
    """
    return np.array([0.0]*int(0.7*p) + [1.0]*(p-int(0.7*p)))


def beta_gen_nonneg(p):
    """
    Generate the linear model coefficient in non-negative orthant case.
    
    @param p int: dimension.
    
    @return np.array(p,1): the coefficient.
    """
    return np.clip(np.random.normal(0, 3, p), 0, None)


def beta_gen_lasso(p):
    """
    Generate the linear model coefficient in constrained lasso case.
    
    @param p int: dimension.
    
    @return np.array(p,1): the coefficient.
    """
    cardi = 0.005
    return np.array([0]*int(p-int(cardi*p)) + [1]*int(cardi*p))


def beta_gen_slope(p):
    """
    Generate the linear model coefficient in slope case.
    
    @param p int: dimension.
    
    @return np.array(p,1): the coefficient.
    """
    cardi = 0.005
    return np.array( [0]*int(p-int(cardi*p)) + list(np.arange(1, int(cardi*p)+1, 1)) )
