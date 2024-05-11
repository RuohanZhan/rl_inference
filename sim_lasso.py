import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from functools import partial
from time import time
from os.path import join
import argparse
from utils import *
from joblib import Parallel, delayed


parser = argparse.ArgumentParser(description='Process DGP settings')
parser.add_argument(
    '-e',
    '--exp_name',
    type=str,
    default='honest_adaptive_linear',
    help='name of experiment')
parser.add_argument(
    '-n',
    '--noise_std',
    type=float,
    default=1.0,
    help='standard deviation of noise')
parser.add_argument(
    '-d',
    '--noise_dis',
    type=str,
    default='uniform',
    help='distribution of noise')
parser.add_argument(
    '-f',
    '--floor_decay',
    type=float,
    default=0.5,
    help='assignment probability decay rate')
parser.add_argument(
    '-s',
    '--n_sims',
    type=int,
    default=50,
    help='number of simulations')
parser.add_argument(
    '-b',
    '--n_batches',
    type=int,
    default=100,
    help='number of batches')
parser.add_argument(
    '-se',
    '--seed',
    type=int,
    default=1,
    help='random seed')

args = parser.parse_args()


# This is a trick to ensure that each simulation has a different seed
name_dir = args.exp_name
rnd = int(time() * 1e8 % 1e8)
SEED = args.seed + rnd

# Setup

"""
Data Generating Process
"""
x_dim = 10
x_informative_dim = 2 # only the first two entries matter in the state transition
feature_dim = 2 # [x0, x0**2]

L = 2

"""
Initial exploration
"""
n_exploration = 1000
noise_std = args.noise_std
noise_dis = args.noise_dis





# Episodic RL
apply_floor = True
floor_decay = args.floor_decay
floor_start = 0.5
batch_size = 100
n_batches = args.n_batches

# Feature mapping
phi = partial(sparse_feature_mapping_true, 2)
psi = partial(sparse_feature_mapping, 2)


n_sims = args.n_sims

estimate_cov_sqrt_root_func = get_estimate_cov_sqrt_root_homoskedasticity

begin_time = time()

def exp(s):
    seed = SEED + s 
    np.random.seed(seed)
    
    results_set = {"naive_estimate":[],
              "naive_cov_sqrt_inv":[],
              "consistent_estimate":[],
              "consistent_cov_sqrt_inv":[],
              "oracle_estimate":[],
              "oracle_cov_sqrt_inv":[],
              "feasible_estimate":[],
              "feasible_cov_sqrt_inv":[],
               "true_thetas": [],
                   "b": [],
              }
    
    """
    A: dense matrix, how current-stage feature affects next-stage state
    B: sparse matrix, how current-stage state affects next-stage state
    C: sparse matrix, how initial state affects next-stage state
    """
    A = generate_sparse_matrix(x_dim, x_dim, feature_dim, feature_dim)
    B = generate_sparse_matrix(x_dim, x_dim, x_dim, x_informative_dim)
    C = generate_sparse_matrix(x_dim, x_dim, x_dim, x_informative_dim)

    """
    gamma: sparse vector of how x0 affects the final outcome
    beta: sparse vector of how x{L-1} affects the final outcome
    alpha: dense vector, how phi{L-1} affects the final outcome
    """
    gamma = generate_sparse_vector(x_dim, x_informative_dim) 
    theta_0 = 0 # baseline policy value
    gamma = np.concatenate([[theta_0], gamma]) # add intercept for baseline policy value
    beta = generate_sparse_vector(x_dim, x_informative_dim)
    alpha = generate_sparse_vector(feature_dim, feature_dim)
    
    # True estimands
    thetas, betas, kappas = get_true_estimands(alpha, beta, gamma, A, B, C, L)
    
    
    # Exploration
    rct_agents = [rct for _ in range(L)]
    Y, X, T, Phi, P = get_observations_sparse_linear_markovian(n_exploration, x_dim, L, A, B, C, alpha, beta, gamma, rct_agents, phi, psi, "normal", noise_std, 0)

    results, return_weights, sum_vars, return_H0 = get_estimates(Y, X, T, psi, P, thetas, betas, kappas, A, B, C, noise_std,
                                                                 estimate_cov_sqrt_root=estimate_cov_sqrt_root_func, fit_oracle=True,
                                                 current_batch_size=n_exploration, sum_vars=np.zeros(L), verbose=False)
    results['true_thetas'] = np.concatenate([[theta_0], thetas.flatten()])
    results['b'] = 0
    results_set = append_results(results_set, results)


    
    # Episodic RL
    for b in range(1, n_batches + 1):
        prev_idx = np.arange(len(X))
        agents = fit_agent(X, T * psi(X), Y)

        floor = 0.0
        if apply_floor:
            floor = floor_start *  b ** (-floor_decay)

        # Draw observations
        Y_b, X_b, T_b, Phi_b, P_b = get_observations_sparse_linear_markovian(batch_size, x_dim, L, A, B, C, alpha, beta, gamma, agents, phi, psi, "normal", noise_std, floor)

        Y = np.concatenate([Y, Y_b], axis=0)
        X = np.concatenate([X, X_b], axis=0)
        T = np.concatenate([T, T_b], axis=0)
        Phi = np.concatenate([Phi, Phi_b], axis=0)
        P = np.concatenate([P, P_b], axis=0)


        results, return_weights, sum_vars, return_H0 = get_estimates(Y, X, T, psi, P, thetas, betas, kappas,  A, B, C, noise_std,
                                                                     estimate_cov_sqrt_root=estimate_cov_sqrt_root_func, fit_oracle=True,
                                                                 current_batch_size=batch_size, sum_vars=sum_vars, prev_feasible_weights=return_weights, prev_H0=return_H0, verbose=False)
        results['true_thetas'] = np.concatenate([[theta_0], thetas.flatten()])
        results['b'] = b
        results_set = append_results(results_set, results)



    # Save everything
    filename = compose_filename(name_dir + "_" + str(s), "npz") #f'{name_dir}_{s}.npz' 
    save_path = join('results', filename)
    print(f"Save path: \t{save_path}")
    np.savez(save_path,
             # naive
             naive_Theta=results_set['naive_estimate'],
             naive_Cov_sqrt_inv=results_set['naive_cov_sqrt_inv'],
             # consistent
             consistent_Theta=results_set['consistent_estimate'], 
             consistent_Cov_sqrt_inv=results_set['consistent_cov_sqrt_inv'],
             # oracle
             oracle_Theta=results_set['oracle_estimate'], 
             oracle_Cov_sqrt_inv=results_set['oracle_cov_sqrt_inv'],
             # feasible
             feasible_Theta=results_set['feasible_estimate'], 
             feasible_Cov_sqrt_inv=results_set['feasible_cov_sqrt_inv'],
             # data
             Y=Y, X=X, T=T, P=P, noise_std=noise_std, 
             thetas=results_set['true_thetas'],
             floor_decay=floor_decay, noise_dis=args.noise_dis)

Parallel(n_jobs=-1, verbose=5)(delayed(exp)(s) for s in range(n_sims))

print(f"Finished in {time() - begin_time} seconds.")
