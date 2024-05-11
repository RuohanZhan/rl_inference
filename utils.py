import numpy as np
import subprocess
from time import time
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from functools import partial
from scipy.stats import norm
from scipy.linalg import sqrtm, inv

from sklearn.linear_model import MultiTaskLassoCV, MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

def generate_sparse_vector(n, p):
    # only the first p features of x is nonzero
    x = np.zeros(n)
    x[:p] = np.random.uniform(low=-1, high=1, size=p)
    return x

def generate_sparse_matrix(n1, p1, n2, p2):
    # only the first p*p entries of the matrix is nonzero
    x = np.zeros((n1, n2))
    x[:p1, :p2] = np.random.uniform(low=-1, high=1, size=(p1, p2))
    return x



def get_true_estimands(alpha, beta, gamma, A, B, C, L):
    thetas = [alpha]
    betas = [beta]
    kappas = [gamma]
    beta_j = np.copy(beta)
    for _ in range(L-1):
        kappa_j = kappas[-1].copy()
        kappa_j[1:] = kappa_j[1:] + np.matmul(C.T, beta_j)
        theta_j = np.matmul(A.T, beta_j)
        beta_j = np.matmul(B.T, beta_j)
        kappas.append(kappa_j)
        thetas.append(theta_j)
        betas.append(beta_j) 
    thetas = np.array(thetas[::-1])
    betas = np.array(betas[::-1])
    kappas = np.array(kappas[::-1])
    return thetas, betas, kappas

 
def nonlinear_feature_mapping(x, t, scale=1.0):
    return np.exp(scale * (x * t)) - 1

def identity(x):
    x0 = x[..., 0:1]
    return x0

def sparse_feature_mapping(degree, x):
    """
    only the first entry of x are informative
    (x0, x0 ** 2, ..., x0 ** d)
    """
    x0 = x[..., 0:1]
    X = []
    for d in np.arange(1, degree + 1):
        X.append(x0 ** d)
    return np.concatenate(X, axis = -1)


def sparse_feature_mapping_true(degree, x, t):
    """
    only the first entry of x are informative
    (x0, x0 ** 2)
    """
    return t * sparse_feature_mapping(degree, x)



def rct(x, floor):
    """
    Batch RCT
    ======================
    Input:
     - x: shape (n, dim_x)
    Returns:
     - treatment status, shape (n, 1)
     - treatment assignment probability, shape (n, 1)
    """
    n = len(x)
    probs = 0.5 * np.ones(n)
    treatment = np.random.choice(a=2, size=n)
    return treatment[...,np.newaxis], probs[...,np.newaxis]

def generate_noise(noise_dis, noise_std, L, n, dim_x):
    if noise_dis == 'normal':
        xi = np.random.normal(loc=0.0, scale=noise_std, size=(L-1, n, dim_x))
        epsilon = np.random.normal(loc=0.0, scale=noise_std, size=n)
    elif noise_dis == 'uniform':
        xi = np.random.uniform(low=-np.sqrt(3) * noise_std, high=np.sqrt(3) * noise_std, size=(L-1, n, dim_x))
        epsilon = np.random.uniform(low=-np.sqrt(3) * noise_std, high=np.sqrt(3) * noise_std, size=n)
    elif noise_dis == 'exp':
        scale = noise_std / np.log(2)
        xi = np.random.exponential(scale=scale, size=(L-1, n, dim_x)) - scale
        epsilon = np.random.exponential(scale=scale, size=n) - scale
    else:
        raise NotImplementedError 
    return xi, epsilon
    

def get_observations_sparse_linear_markovian(n, x_dim, L, A, B, C, alpha, beta, gamma, agents, phi, agent_psi, noise_dis, noise_std, floor):
    """
    n: batch_size
    dim_x: x dimension
    L: horizon
    agent: treatment assignment probs
    psi: true feature mapping
    """
    x = np.random.uniform(low=-1, high=1, size=(n, x_dim)) # X1
    x1 = np.copy(x)
    X = [x] # content
    T = [] # treatment
    P = [] # treated probs
    Phi = [] # features
    
    xi, epsilon = generate_noise(noise_dis, noise_std, L, n, x_dim)
    
    for l in range(L-1):
        t, p = agents[l](agent_psi(x), floor)
        T.append(t)
        P.append(p)
        phi_l = phi(x, t)
        x = phi_l @ A.T + x @ B.T + x1 @ C.T + xi[l] # (n, dim_x)
        X.append(x) # size (L, n, dim_x)
        Phi.append(phi_l)
    
    t, p = agents[L-1](agent_psi(x), floor)
    T.append(t)  # size (L, n, 1)
    P.append(p)  # size (L, n, 1)                            
    phi_L = phi(x, t)
    Phi.append(phi_L) # size (L, n, dim_phi)
    y = phi_L @ alpha + x @ beta + sm.add_constant(x1) @ gamma + epsilon # size n
    return np.array(y), np.transpose(X, (1, 0, 2)), np.transpose(T, (1, 0, 2)),  np.transpose(Phi, (1, 0, 2)), np.transpose(P, (1, 0, 2))


def get_y_residual(Phi, Q, y, H, h0):
    """
    H: (L, dim_feature, dim_feature)
    Phi: (L, dim_feature)
    Q: (L, dim_feature)
    y: scalar
    """
    r = H @ ((Phi - Q)[:,:,np.newaxis]) # size(L, dim_feature, 1)
    r = np.concatenate([[h0], r.flatten()]) # add policy value
    return -y * r

def get_jacobian_matrix(Phi, Q, H, h0):
    """
    H: (L, dim_feature, dim_feature)
    Phi: (L, dim_feature)
    Q: (L, dim_feature)
    """
    L, dim_feature = Phi.shape
    J = np.zeros((L * dim_feature + 1, L * dim_feature + 1))
    J[0,0] = -1
    for l2 in range(L):
        J[0,(l2*dim_feature+1):((l2+1)*dim_feature+1)] = -h0 * Phi[l2]
    for l1 in range(L):
        for l2 in range(l1, L):
            J[(l1*dim_feature + 1):((l1+1)*dim_feature + 1), 
              (l2*dim_feature + 1):((l2+1)*dim_feature + 1)] = - H[l1] @ ((Phi[l1]-Q[l1])[:, np.newaxis]) @ (Phi[l2][np.newaxis, :])
    return J


def get_identity_weight_matrix(n, L, dim_feature):
    H = np.zeros((n, L, dim_feature, dim_feature))
    for i in range(n):
        for l in range(L):
            H[i,l] = np.identity(dim_feature)
    return H

def estimate_baseline_policy_value(thetas, y, Phi):
    n, L, dim_feature = np.shape(Phi)
    estimates = y - np.dot(np.reshape(Phi, [n, -1]), thetas)
    return np.mean(estimates), np.std(estimates) / np.sqrt(n)

def estimate_baseline_policy_var(thetas, y, Phi):
    n, L, dim_feature = np.shape(Phi)
    estimates = y - np.dot(np.reshape(Phi, [n, -1]), thetas)
    return np.var(estimates)

def get_root_inv(Cov):
    n, L, dim_feature, dim_feature = Cov.shape
    Cov_root_inv = np.zeros_like(Cov)
    for i in range(n):
        for l in range(L):
            Cov_root_inv[i, l] = np.diag(1.0/np.sqrt(np.diag(Cov[i, l])))
    return Cov_root_inv

def get_residual_expectation(betas, kappas, X):
    """
    theta_list: list of structure parameters, doesn't include baseline policy value
    Structure mean nested model.
    For time j, calculate E[y-\sum_{k>=j}\theta_k * \phi_k | X_j].
    """
    n, L, dim_X = X.shape
    R = np.zeros((n, L))
    for l in range(L):
        R[:, l] = X[:, l, :] @ betas[l] + sm.add_constant(X[:, 0, :]) @ kappas[l]
    return R # size (n, L)
    # residual = np.concatenate([np.zeros((n, 1)), residual], axis=1) # for baseline policy value
    

def get_residual_second_moment(betas, kappas, X, noise_std):
    """
    theta_list: list of structure parameters, doesn't include baseline policy value
    Structure mean nested model.
    For time j, calculate E[y-\sum_{k>=j}\theta_k * \phi_k | X_j].
    """
    n, L, dim_X = X.shape
    R2 = np.zeros((n, L))
    for l in range(L):
        R2[:, l] = (X[:, l, :] @ betas[l] + sm.add_constant(X[:, 0, :]) @ kappas[l]) ** 2 
    # add exogeneous noise
    for l in range(L-1):
        R2[:, l] += ((np.sum(betas[l+1:L] ** 2) + 1) * (noise_std ** 2))
    R2[:, L-1] += (noise_std ** 2)
    return R2 # size (n, L)

def get_oracle_H0(betas, kappas, noise_std):
    value_var =  (np.sum((betas[0] + kappas[0][1:]) ** 2) / 3.0  # Variance from X1
                  + (np.sum(betas[1:] ** 2) + 1) * (noise_std ** 2)) # Variance from exogeneous noise
    return 1.0 / np.sqrt(value_var)

def get_oracle_cov_sqrt_root(X, Phi, A, B, C, noise_std):
    n, L, dim_feature = Phi.shape
    w = np.zeros((n, L, dim_feature, dim_feature))
    
    phi0_cov = np.zeros((2, 2))
    for i in range(n):
        xi0 = np.array([X[i, 0, 0], X[i, 0, 0]**2])
        phi0_cov += np.matmul(xi0[:, np.newaxis], xi0[np.newaxis, :])
    phi0_cov = phi0_cov / n
    w[:, 0] = inv(sqrtm(phi0_cov)) # we don't need to estimate the cov_phi for the first stage
    
    for l in range(L-1):
        E_x = Phi[:, l] @ A.T + X[:, l] @ (B.T) + X[:, 0] @ C.T # [n, dim_x]
        for i in range(n):
            E_x0 = E_x[i, 0]
            # hard encode given the feature mapping psi = sparse_feature_mapping
            w[i, l+1] = inv(sqrtm(np.array([[E_x0**2 + noise_std**2, E_x0**3 + 3 * E_x0 * noise_std**2],
                                 [E_x0**3 + 3 * E_x0 * noise_std**2, E_x0**4 + 6 * E_x0**2 * noise_std**2 + 3 * noise_std**4]])))
    return w


def get_estimate_cov_sqrt_root_homoskedasticity(X, T, feature_dim, psi, current_batch_size):
    """
    Estimate W = E_{t-1}[\psi(X)\psi(X)^T]^{-1/2}
    psi: feature mapping
    """
    n, L, _ = X.shape
    w = np.zeros((current_batch_size, L, feature_dim, feature_dim))

    # we don't need to estimate the cov_phi for the first stage
    psi0_cov = np.zeros((feature_dim, feature_dim))
    psi0 = psi(X[:, 0])
    psi0_cov = psi0.T @ psi0 / n 
    w[:, 0] = inv(sqrtm(psi0_cov)) 

    def _estimate_lasso_model(x, x1):
        model = MultiTaskLassoCV()
        x1pred = cross_val_predict(model, x, x1, cv=5)
        res = x1 - x1pred
        return x1pred, res
        
    for l in range(1, L):
        xx0 = X[:, 0]
        xxlm = X[:, l-1]
        tt0 = T[:, 0]
        ttlm = T[:, l-1]
        if l == 1:
            x = np.hstack((xx0, tt0*xx0))
        else:
            x = np.hstack((xx0, xxlm, tt0*xx0, tt0*xxlm))
        x1 = X[:, l]
        x1pred, res = _estimate_lasso_model(x, x1)
        for i in range(current_batch_size):
            x1_i = x1pred[[-current_batch_size+i]]
            psi_i = psi(x1_i + res)
            cov = psi_i.T @ psi_i / psi_i.shape[0]
            w[i,l] = inv(sqrtm(cov))
    return w

def get_estimate_cov_sqrt_root_degree_2(X, T, current_batch_size):
    """
    hard code estimation given the feature mapping psi = sparse_feature_mapping
    """
    
    n, L, dim_x = X.shape
    w = np.zeros((current_batch_size, L, 2, 2))
    
    phi0_cov = np.zeros((2, 2))
    for i in range(n):
        xi0 = np.array([X[i, 0, 0], X[i, 0, 0]**2])
        phi0_cov += np.matmul(xi0[:, np.newaxis], xi0[np.newaxis, :])
    phi0_cov = phi0_cov / n
    w[:, 0] = inv(sqrtm(phi0_cov)) # we don't need to estimate the cov_phi for the first stage
    
    def _estimate_lasso_model(x, y):
        model = sm.OLS(y, x)
        results = model.fit()
        yhat = results.predict(x)
        return yhat
    
    for l in range(1, L):
        
        xx0 = X[:, 0]
        xxlm = X[:, l-1]
        tt0 = T[:, 0]
        ttlm = T[:, l-1]
        
        
        if l == 1:
            x = np.hstack((xx0, tt0*xx0))
        else:
            x = np.hstack((xx0, xxlm, tt0*xx0, tt0*xxlm))
        
        y = X[:, l, 0]
        yhat = _estimate_lasso_model(x, y)
        xi = yhat - y
        xi2hat = np.mean(xi ** 2)
        xi3hat = np.mean(xi ** 3)
        xi4hat = np.mean(xi ** 4)
        
        for i in range(current_batch_size):
            yi = yhat[-current_batch_size+i]
            cov = np.array([[yi**2 + xi2hat, yi**3 + 3 * yi * xi2hat + xi3hat],
                           [yi**3 + 3 * yi * xi2hat + xi3hat, yi**4 + xi4hat + 6 * yi**2 * xi2hat + 4*yi*xi3hat]])
            w[i,l] = inv(sqrtm(cov))
    return w

def get_estimate_cov_sqrt_root_degree_1(X, T, current_batch_size):
    n, L, dim_x = X.shape
    w = 1.0 / np.abs(X)
    w = w[..., np.newaxis]
    return w[-current_batch_size:]



def estimate_residual_second_moment(Y, Phi, X, thetas, current_batch_size, sum_vars):
    """
    batches: list of batch size till the current obs
    
    """
    n, L, dim_x = X.shape
    dim_feature = Phi.shape[-1]
    betas, kappas = [], []
    for l in range(L):
        res = Y - np.reshape(Phi[:, l:], [n,-1]) @ thetas[l*dim_feature :]
        x = np.concatenate([X[:,0], X[:,l]], axis=1)
        x = sm.add_constant(x)
        model = sm.OLS(res, x)
        # alphas = np.zeros(1 + dim_x * 2)
        # alphas[1:] = 1 / np.sqrt(n) # do not add L1 penalization to the intercept
        # results = model.fit_regularized(method='elastic_net', alpha=alphas, L1_wt=1.0)
        results = model.fit()
        sum_vars[l] += np.sum((res[-current_batch_size:] - results.predict(x[-current_batch_size:])) ** 2)
        kappas.append(results.params[0:(dim_x + 1)])
        betas.append(results.params[(dim_x + 1):])
    g_estimate = get_residual_expectation(betas, kappas, X[-current_batch_size:])
    return g_estimate ** 2 + sum_vars / n, sum_vars


def estimate_linear_params(Y, Phi, Q, H, H0):
    """
    Solve theta via weighted empirical Z-estimator.
    Y: (n)
    Phi: (n, L, dim_feature)
    Q: (n, L, dim_feature)
    H: (n, L, dim_feature, dim_feature)
    """
    n, L, dim_feature = Phi.shape
    J = np.zeros((L * dim_feature + 1, L * dim_feature + 1)) # add policy value
    Yh = np.zeros(L * dim_feature + 1)
    for Y_t, Phi_t, Q_t, H_t, H0_t in zip(Y, Phi, Q, H, H0):
        J = J + get_jacobian_matrix(Phi_t, Q_t, H_t, H0_t)
        Yh = Yh + get_y_residual(Phi_t, Q_t, Y_t, H_t, H0_t)
    theta_hat = np.matmul(np.linalg.inv(J), Yh)
    cov_sqrt_inv_hat = - J / np.sqrt(n)
    return theta_hat, cov_sqrt_inv_hat



def get_estimates(Y, X, T, psi, P, thetas, betas, kappas, A, B, C, noise_std, estimate_cov_sqrt_root, fit_oracle=True, current_batch_size=None, sum_vars=None, prev_feasible_weights=None, verbose=False, prev_H0=None):
    """
    Inputs:
      - Y: [n]
      - X: [n, L, x_dim]
      - T: [n, L, 1]
      - P: [n, L, 1] probability of assigning the treatment
      
    
    Estimate structure parameters:
      - naive: standard Z-estimation
      - consistent: consistent-weighted Z-estimation
      - bX_var: fixed effect variance, where the randomness comes from initial context X_1
      - oracle: oracle-weighted Z-estimation
      - feasible: feasible-weighted Z-estimation
    Then estimate the baseline policy value
    """
    n, L, x_dim = X.shape
    def get_feature(psi, X, T):
        Phi1 = psi(X) 
        Phi = T * Phi1
        Q = P * Phi1
        Cov_T = P * (1 - P)
        return Phi, Q, Cov_T[..., np.newaxis]
    Phi, Q, Cov_T = get_feature(psi, X, T)
    feature_dim = Phi.shape[-1]
    
    identity_weights_params = get_identity_weight_matrix(n, L, feature_dim)
    identity_weights_policy_value = np.ones(n)
    naive_estimate, naive_cov_sqrt_inv = estimate_linear_params(Y, Phi, Q, identity_weights_params, identity_weights_policy_value)
    

    consistent_weights = 1.0 / np.sqrt(Cov_T) * identity_weights_params
    consistent_estimate, consistent_cov_sqrt_inv = estimate_linear_params(Y, Phi, Q, consistent_weights, identity_weights_policy_value)
    
    if fit_oracle:
        oracle_r2 = get_residual_second_moment(betas, kappas, X, noise_std)[..., np.newaxis, np.newaxis]
        oracle_w = get_oracle_cov_sqrt_root(X, Phi, A, B, C, noise_std)
        oracle_weights = oracle_w / np.sqrt(Cov_T * oracle_r2)
        oracle_estimate, oracle_cov_sqrt_inv = estimate_linear_params(Y, Phi, Q, oracle_weights, get_oracle_H0(betas, kappas, noise_std) * np.ones(n))
    else:
        oracle_estimate, oracle_cov_sqrt_inv = None, None
 
    if current_batch_size is not None:
        # Estimate feasible weights
        feasible_weights = np.copy(consistent_weights)
        feasible_weights[:-current_batch_size] = prev_feasible_weights
        
        r2_est, sum_vars = estimate_residual_second_moment(Y, Phi, X, consistent_estimate.flatten()[1:], current_batch_size, sum_vars)
        r2_est = r2_est[..., np.newaxis, np.newaxis]
        estimate_w = estimate_cov_sqrt_root(X, T, feature_dim, psi, current_batch_size) # estimate_cov_sqrt_root(X, T, current_batch_size)
        feasible_weights[-current_batch_size:] = estimate_w / np.sqrt(Cov_T[-current_batch_size:] * r2_est)
        
        # estimate H_0
        baseline_policy_var = estimate_baseline_policy_var(consistent_estimate[1:], Y, Phi)
        feasible_H0 = np.zeros(n)
        feasible_H0[:-current_batch_size] = prev_H0
        feasible_H0[-current_batch_size:] = 1.0 / np.sqrt(baseline_policy_var)
        
        # Get feasible estimates
        feasible_estimate, feasible_cov_sqrt_inv = estimate_linear_params(Y, Phi, Q, feasible_weights, feasible_H0)
        return_weights = feasible_weights
        return_H0 = feasible_H0
    else:
        (feasible_estimate, 
         feasible_cov_sqrt_inv, 
         feasible_baseline_value_estimate, 
         feasible_baseline_value_stderr) = None, None, None, None
        return_weights = consistent_weights
        return_H0 = np.ones(n)
        
    
    if verbose:
        print(f"true theta: {thetas.flatten()}")
        print(f"naive estimate: {naive_estimate}")
        print(f"consistent estimate: {consistent_estimate}")
        print(f"feasible estimate: {feasible_estimate}")
        print(f"oracle estimate: {oracle_estimate}")

        
    return {"naive_estimate": naive_estimate, 
            "naive_cov_sqrt_inv": naive_cov_sqrt_inv,
            "consistent_estimate": consistent_estimate, 
            "consistent_cov_sqrt_inv": consistent_cov_sqrt_inv,
            "oracle_estimate": oracle_estimate, 
            "oracle_cov_sqrt_inv": oracle_cov_sqrt_inv,
            "feasible_estimate": feasible_estimate, 
            "feasible_cov_sqrt_inv": feasible_cov_sqrt_inv,
           }, return_weights, sum_vars, return_H0


def policy(psi, floor, mu=0.0):
    """
    phi: batch features at a period. [n, feature_dim]
    Policy:
     - epsilon greedy with epsilon = floor
    """
    psi_mu = psi @ mu #[n, 1]
    p = np.ones(len(psi)) * floor / 2 + (1 - floor) * (np.array(psi_mu > 0).astype(np.float64))
    treatment = np.array([np.random.choice(a=[0, 1.0], p=[1 - p_tr, p_tr]) for p_tr in p])
    return treatment[..., np.newaxis], p[..., np.newaxis]


def fit_agent(X, Phi, Y):
    """
    X: [n, L, x_dim]
    Phi: [n, L, feature_dim]
    Y: [n]
    """
    x1 = X[:,0] #[n, x_dim]
    x_dim = x1.shape[-1]
    x1 = sm.add_constant(x1) # intercept + x1 (n, dim_x + 1)
    
    n, L, feature_dim = Phi.shape
    f_dim = L * feature_dim
    feature = np.reshape(Phi, (n, f_dim))
    x = np.concatenate([feature, x1], axis=1)
    model = sm.OLS(Y, x)
    alpha = np.zeros(f_dim + 1 + x_dim)
    alpha[f_dim + 1:] = 1 / np.sqrt(n) # only add L1 penalty to X1
    results = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=1.0)  
    policies = []
    for l in range(L):
        policies.append(partial(policy, mu=results.params[(l * feature_dim): ((l + 1) * feature_dim)]))
    return policies


def append_results(results_set, new_results):
    for r in results_set:
        results_set[r].append(new_results[r])
    return results_set

def compose_filename(prefix, extension):
    """
    Creates a unique filename.
    Useful when running in parallel on Sherlock.
    """
    # Tries to find a commit hash
    try:
        commit = subprocess\
            .check_output(['git', 'rev-parse', '--short', 'HEAD'],
                          stderr=subprocess.DEVNULL)\
            .strip()\
            .decode('ascii')
    except subprocess.CalledProcessError:
        commit = ''

    # Other unique identifiers
    rnd = str(int(time() * 1e8 % 1e8))
    ident = filter(None, [prefix, commit, rnd])
    basename = "_".join(ident)
    fname = f"{basename}.{extension}"
    return fname

def compute_coverage(theta_hat, cov_hat, true_theta, confidence_level=0.95):
    scales = np.sqrt(np.diag(cov_hat))
    confidence_intervals = norm.ppf((confidence_level + 1 ) / 2.0) * np.sqrt(np.diag(cov_hat))
    coverages = np.abs(theta_hat - true_theta) <= confidence_intervals
    return coverages, confidence_intervals