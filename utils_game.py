import cvxpy as cp
import itertools, random

from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *
from scipy.special import gamma
from scipy.special import loggamma, digamma
from scipy.optimize import curve_fit
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from config import cfg

import torch
import  numpy as np
def server_modulated_weights(q_ck, tst_x, tst_y, client_params_list, model_func, data_obj):

    modulation = 0.0025 # 0.0005 - low, 0.005 - high; clients opt out without further incentive, behaves as FedAvg on full participation
                       # a more selfish server should give more incentive
                       # on uniform data server wants everyone to participate, a higher modulation is thus not beneficial for server
                       
    q_s = np.asarray([1.0]*len(q_ck)).astype('float32')
    q_s = q_s.reshape((len(q_ck),1))

    for clnt in range(len(client_params_list)):
        tst_model = model_func().to(device)
        tst_model = set_client_from_params(tst_model, client_params_list[clnt])

        val_loss, val_acc = get_acc_loss(tst_x, tst_y, tst_model, data_obj.dataset)
        q_s[clnt] = q_ck[clnt] - modulation*val_loss

        del(tst_model)

    q_s_filtered = np.where(q_s < 0, 0, q_s)
    q_s = q_s_filtered/np.sum(q_s_filtered)
    # breakpoint()
    return q_s


def calculate_clnt_knowledge_inc(q_kr, q_krminusone, clnt_self_index, loss_g, loss_gminusone, positive_cor_prob, negative_cor_prob):

    alpha = 0.6 # 0.75
    c_plus = 0.5 # 0.1 - low, 0.5 - high for CIFAR10 with 10 clients
    # c_plus = 0.1 # CIFAR100 20 clients
    c_minus = 0.01 # 0.01 - low
    corr_positive = 0.0
    corr_negative = 0.0

    del_q_kr = q_kr - q_krminusone
    del_fk_ws = loss_g - loss_gminusone
    # breakpoint()
    for clnt in range(len(q_kr)):
        if (clnt!= clnt_self_index):
            # parallel
            if (del_q_kr[clnt]>=0 and del_fk_ws<=0) or (del_q_kr[clnt]<=0 and del_fk_ws>=0):
                positive_cor_prob[clnt_self_index][clnt] = alpha + (1-alpha)*positive_cor_prob[clnt_self_index][clnt]
            else:
                positive_cor_prob[clnt_self_index][clnt] = (1-alpha)*positive_cor_prob[clnt_self_index][clnt]

            # anti-parallel
            if (del_q_kr[clnt]>0 and del_fk_ws>0) or (del_q_kr[clnt]<0 and del_fk_ws<0):
                negative_cor_prob[clnt_self_index][clnt] = alpha + (1-alpha)*negative_cor_prob[clnt_self_index][clnt]
            else:
                negative_cor_prob[clnt_self_index][clnt] = (1-alpha)*negative_cor_prob[clnt_self_index][clnt]

    corr_positive = np.sum(positive_cor_prob[clnt_self_index] >= 0.6)
    corr_negative = np.sum(negative_cor_prob[clnt_self_index] >= 0.6)
    # print(positive_cor_prob)
    # exit()
    p_k = c_plus*corr_positive - c_minus*corr_negative

    return p_k, positive_cor_prob, negative_cor_prob


# for the clients
def get_ONS_weights(modulated_weight_list, tst_perf_clnts, i, n_clnt, positive_cor_prob, negative_cor_prob, 
                    grad_history, hess_history, alpha = 1.0, beta = 0.05):
    # calculates the ONS weight strategy for all the clients
    # dot_product: 1.0 + <p, r>
    # to change the dot product for each client based on positive_cor and negative_cor
    # r = (delta*positive_cor + gamma*negative_cor) * r
    delta = 1.0 
    gamma = -0.5
    
    # populate 1.0 at the diagonal elements of positive_cor_prob and negative_cor_prob
    np.fill_diagonal(positive_cor_prob, 1.0)
    cor_prob = delta*positive_cor_prob + gamma*negative_cor_prob
    cor_prob[cor_prob < 0.0] = 1e-3
    loss_vec = tst_perf_clnts[:,i,0].reshape(-1)
    personalized_r = loss_vec[None,:]*cor_prob # out: n_clnt x n_clnt
    # breakpoint()
    # bounding the personalized_r for lipschitz continuity of the loss = log(1.0 + <p, personalized_r>)
    means = personalized_r.mean(axis=1)
    means = np.maximum(means, 1e-12)
    personalized_r = personalized_r/means[:,None]
    # Weibull CDF (shape=2, scale=1) => 1 - e^{- (x^2)}
    personalized_r = 1.0 - np.exp(-personalized_r**2) # out: n_clnt x n_clnt
    personalized_r = personalized_r/n_clnt

    p = modulated_weight_list.reshape(-1) # out: n_clnt
    dot_prod = np.sum(p[None, :] * personalized_r, axis=1, keepdims=True) + 1.0 # out: n_clnt x 1
    # breakpoint()
    # loss = log(1.0 + <p, personalized_r>)
    grads = -personalized_r/dot_prod # out: n_clnt x n_clnt
    hess = grads[:, :, None] * grads[:, None, :]
    epsilon = 1e-6
    for idx in range(hess.shape[0]):
        H = hess[idx]
        H = 0.5 * (H + H.T) # Ensure symmetry explicitly for numerical stability
        # Add ridge term to diagonal to fix tiny negatives to get a PSD
        H += epsilon * np.eye(H.shape[0])
        hess[idx] = H
    # breakpoint()
    if i == 0:
        grad_history = grads
        hess_history = hess
    else:
        grad_history = 0.9*grad_history + 0.1*grads
        hess_history = 0.9*hess_history + 0.1*hess

    # formulate and solve the optimization
    obj_terms = []
    p_var = cp.Variable((n_clnt, n_clnt), nonneg=True)

    for clnt in range(n_clnt):
        lin = cp.sum(cp.multiply(grad_history[clnt], p_var[clnt]))
        # L2 term
        l2  = 0.5 * alpha * cp.sum_squares(p_var[clnt])
        # quadratic Hessian term
        quad = 0.5 * beta * cp.quad_form(p_var[clnt], hess_history[clnt])
        # maximize entropy
        # regularizer = 0.1 * cp.sum(p_var[i] * cp.log(p_var[i]))
        obj_terms.append(lin + l2 + quad)

    objective = cp.Minimize(cp.sum(obj_terms))

    constraints = []

    for i in range(n_clnt):
        constraints.append(cp.sum(p_var[i]) == 1)

    prob = cp.Problem(objective, constraints)
    _ = prob.solve()
    
    return p_var.value, grad_history, hess_history


def get_ONS_weights(modulated_weight_list, tst_perf_clnts, i, n_clnt, positive_cor_prob, negative_cor_prob,
                    grad_history, hess_history, alpha = 1.0, beta = 0.05):
    # calculates the ONS weight strategy for all the clients
    # dot_product: 1.0 + <p, r>
    # to change the dot product for each client based on positive_cor and negative_cor
    # r = (delta*positive_cor + gamma*negative_cor) * r
    delta = 1.0 
    gamma = -0.5
    ratio_client = 0.5
    ratio_server = 0.5
    # populate 1.0 at the diagonal elements of positive_cor_prob and negative_cor_prob
    # np.fill_diagonal(positive_cor_prob, 1.0)
    # cor_prob = delta*positive_cor_prob + gamma*negative_cor_prob
    # cor_prob[cor_prob < 0.0] = 1e-3
    loss_vec = tst_perf_clnts[:,i,0].reshape(-1)
    personalized_r = loss_vec[None,:] # out: n_clnt x n_clnt
    # breakpoint()
    # bounding the personalized_r for lipschitz continuity of the loss = log(1.0 + <p, personalized_r>)
    means = personalized_r.mean(axis=1)
    means = np.maximum(means, 1e-12)
    personalized_r = personalized_r/means[:,None]
    # Weibull CDF (shape=2, scale=1) => 1 - e^{- (x^2)}
    personalized_r = 1.0 - np.exp(-personalized_r**2) # out: n_clnt x n_clnt
    personalized_r = personalized_r/n_clnt
 
    p = modulated_weight_list.reshape(-1) # out: n_clnt
    dot_prod = np.sum(p[None, :] * personalized_r, axis=1, keepdims=True) + 1.0 # out: n_clnt x 1
    # breakpoint()
    # loss = log(1.0 + <p, personalized_r>)
    grads = -personalized_r/dot_prod # out: n_clnt x n_clnt
    hess = grads[:, :, None] * grads[:, None, :]
    epsilon = 1e-6
    for idx in range(hess.shape[0]):
        H = hess[idx]
        H = 0.5 * (H + H.T) # Ensure symmetry explicitly for numerical stability
        # Add ridge term to diagonal to fix tiny negatives to get a PSD
        H += epsilon * np.eye(H.shape[0])
        hess[idx] = H
    # breakpoint()
    if i == 0:
        grad_history += grads
        hess_history += hess
    else:
        grad_history = 0.9*grad_history + 0.1*grads
        hess_history = 0.9*hess_history + 0.1*hess
 
    # formulate and solve the optimization
    obj_terms = []
    p_var = cp.Variable((n_clnt, n_clnt), nonneg=True)
 
    for clnt in range(n_clnt):
        lin = cp.sum(cp.multiply(grad_history[clnt], p_var[clnt]))
        # L2 term
        l2  = 0.5 * alpha * cp.sum_squares(p_var[clnt])
        # quadratic Hessian term
        quad = 0.5 * beta * cp.quad_form(p_var[clnt], hess_history[clnt])
        # maximize entropy
        # regularizer = 0.1 * cp.sum(p_var[i] * cp.log(p_var[i]))
        obj_terms.append(lin + l2 + quad)
 
    objective = cp.Minimize(cp.sum(obj_terms))
 
    constraints = []
 
    for i in range(n_clnt):
        constraints.append(cp.sum(p_var[i]) == 1)
 
    prob = cp.Problem(objective, constraints)
    _ = prob.solve()
 
    return p_var.value, grad_history, hess_history


#combined ONS
def get_combinedONS_weights(modulated_weight_list, tst_perf_clnts, i, n_clnt, positive_cor_prob, negative_cor_prob, bound_r,
                    grad_history, hess_history, alpha = 1.0, beta = 0.05):
    # calculates the ONS weight strategy for all the clients
    # dot_product: 1.0 + <p, r>
    # to change the dot product for each client based on positive_cor and negative_cor
    # r = (delta*positive_cor + gamma*negative_cor) * r
    delta = 1.0 
    gamma = -0.5
    ratio_client = cfg['beta_clnt']
    ratio_server = cfg['beta_srv']
    # populate 1.0 at the diagonal elements of positive_cor_prob and negative_cor_prob
    np.fill_diagonal(positive_cor_prob, 1.0)
    cor_prob = delta*positive_cor_prob + gamma*negative_cor_prob
    cor_prob[cor_prob < 0.0] = 1e-3
    loss_vec = tst_perf_clnts[:,i,0].reshape(-1)
    # print(loss_vec)
    # exit()
    #from client losses
    personalized_r = loss_vec[None,:]*cor_prob # out: n_clnt x n_clnt
    # breakpoint()
    # bounding the personalized_r for lipschitz continuity of the loss = log(1.0 + <p, personalized_r>)
    means = personalized_r.mean(axis=1)
    means = np.maximum(means, 1e-12)
    personalized_r = personalized_r/means[:,None]
    # Weibull CDF (shape=2, scale=1) => 1 - e^{- (x^2)}
    personalized_r = 1.0 - np.exp(-personalized_r**2) # out: n_clnt x n_clnt
    personalized_r = personalized_r/n_clnt

    p = modulated_weight_list.reshape(-1) # out: n_clnt
    dot_prod = np.sum(p[None, :] * personalized_r, axis=1, keepdims=True) + 1.0 # out: n_clnt x 1
    # breakpoint()
    #from loss on server heldoutset
    personalized_r_srv = bound_r[None,:]*cor_prob # out: n_clnt x n_clnt
    # breakpoint()
    # bounding the personalized_r for lipschitz continuity of the loss = log(1.0 + <p, personalized_r>)
    means = personalized_r_srv.mean(axis=1)
    means = np.maximum(means, 1e-12)
    personalized_r_srv = personalized_r_srv/means[:,None]
    # Weibull CDF (shape=2, scale=1) => 1 - e^{- (x^2)}
    personalized_r_srv = 1.0 - np.exp(-personalized_r_srv**2) # out: n_clnt x n_clnt
    personalized_r_srv = personalized_r_srv/n_clnt

    p = modulated_weight_list.reshape(-1) # out: n_clnt
    dot_prod = np.sum(p[None, :] * personalized_r, axis=1, keepdims=True) + 1.0 # out: n_clnt x 1
    dot_prod_srv = np.sum(p[None, :] * personalized_r_srv, axis=1, keepdims=True) + 1.0 # out: n_clnt x 1
    # loss = log(1.0 + <p, personalized_r>)
    grads = (-ratio_client*personalized_r/dot_prod)+(-ratio_server*personalized_r_srv/dot_prod_srv) # out: n_clnt x n_clnt
    hess = grads[:, :, None] * grads[:, None, :]
    epsilon = 1e-6
    for idx in range(hess.shape[0]):
        H = hess[idx]
        H = 0.5 * (H + H.T) # Ensure symmetry explicitly for numerical stability
        # Add ridge term to diagonal to fix tiny negatives to get a PSD
        H += epsilon * np.eye(H.shape[0])
        hess[idx] = H
    # breakpoint()
    if i == 0:
        grad_history = grads
        hess_history = hess
    else:
        grad_history = 0.9*grad_history + 0.1*grads
        hess_history = 0.9*hess_history + 0.1*hess

    # formulate and solve the optimization
    obj_terms = []
    p_var = cp.Variable((n_clnt, n_clnt), nonneg=True)

    for clnt in range(n_clnt):
        lin = cp.sum(cp.multiply(grad_history[clnt], p_var[clnt]))
        # L2 term
        l2  = 0.5 * alpha * cp.sum_squares(p_var[clnt])
        # quadratic Hessian term
        quad = 0.5 * beta * cp.quad_form(p_var[clnt], hess_history[clnt])
        # maximize entropy
        # regularizer = 0.1 * cp.sum(p_var[i] * cp.log(p_var[i]))
        obj_terms.append(lin + l2 + quad)

    objective = cp.Minimize(cp.sum(obj_terms))

    constraints = []

    for i in range(n_clnt):
        constraints.append(cp.sum(p_var[i]) == 1)

    prob = cp.Problem(objective, constraints)
    _ = prob.solve()
    
    return p_var.value, grad_history, hess_history


#for the server(stop this)
def get_ONS_weights_server(tst_x, tst_y, client_params_list, model_func, data_obj, 
                           server_grad_history, server_hess_history, modulated_weight_list, 
                           alpha = 1.0, beta = 0.05):
    
    n_clnt = client_params_list.shape[0]
    p = modulated_weight_list
    p = p.squeeze(1)
    r = np.zeros(n_clnt, dtype=np.float32)

    for clnt in range(len(client_params_list)): 
        tst_model = model_func().to(device)
        tst_model = set_client_from_params(tst_model, client_params_list[clnt])

        val_loss, val_acc = get_acc_loss(tst_x, tst_y, tst_model, data_obj.dataset)
        r[clnt] = val_loss
        del(tst_model)

    means = r.mean()
    means = max(means, 1e-12)
    bound_r = r/means
    # Weibull CDF (shape=2, scale=1) => 1 - e^{- (x^2)}
    bound_r = 1.0 - np.exp(-bound_r**2) # out: n_clnt x n_clnt
    bound_r = bound_r/n_clnt
    # dot product: 1.0 + <p,r>
    dot_prod = np.dot(p,bound_r) + 1.0
    grads = -bound_r/dot_prod
    hess = np.outer(grads,grads)
    
    epsilon = 1e-6
    for idx in range(hess.shape[0]):
        H = hess
        H = 0.5 * (H + H.T) # Ensure symmetry explicitly for numerical stability
        # Add ridge term to diagonal to fix tiny negatives to get a PSD
        H += epsilon * np.eye(H.shape[0])
        hess = H

    server_grad_history = 0.9*server_grad_history + 0.1*grads
    server_hess_history = 0.9*server_hess_history + 0.1*hess

    p_var = cp.Variable(n_clnt, nonneg=True)
    
    lin = p_var@server_grad_history
    l2  = 0.5 * alpha * cp.sum_squares(p_var)
    quad = 0.5 * beta * cp.quad_form(p_var, server_hess_history)

    objective = lin + l2 + quad
    _ = cp.Problem(cp.Minimize(objective), [cp.sum(p_var) == 1, p_var >= 0]).solve()

    return p_var.value, server_grad_history, server_hess_history, r


def get_r_server(tst_x, tst_y, client_params_list, model_func, data_obj, 
                           server_grad_history, server_hess_history, modulated_weight_list, i,
                           alpha = 1.0, beta = 0.05):
    
    n_clnt = client_params_list.shape[0]
    p = modulated_weight_list
    p = p.squeeze(1)
    r = np.zeros(n_clnt, dtype=np.float32)

    for clnt in range(len(client_params_list)): 
        tst_model = model_func().to(device)
        tst_model = set_client_from_params(tst_model, client_params_list[clnt])

        val_loss, val_acc = get_acc_loss(tst_x, tst_y, tst_model, data_obj.dataset)
        r[clnt] = val_loss
        del(tst_model)

    means = r.mean()
    means = max(means, 1e-12)
    bound_r = r/means
    # Weibull CDF (shape=2, scale=1) => 1 - e^{- (x^2)}
    bound_r = 1.0 - np.exp(-bound_r**2) # out: n_clnt x n_clnt
    bound_r = bound_r/n_clnt
    # # dot product: 1.0 + <p,r>
    # dot_prod = np.dot(p,bound_r) + 1.0
    # grads = -bound_r/dot_prod
    # hess = np.outer(grads,grads)
    
    # epsilon = 1e-6
    # for idx in range(hess.shape[0]):
    #     H = hess
    #     H = 0.5 * (H + H.T) # Ensure symmetry explicitly for numerical stability
    #     # Add ridge term to diagonal to fix tiny negatives to get a PSD
    #     H += epsilon * np.eye(H.shape[0])
    #     hess = H
    # if i == 1:
    #     server_grad_history = grads
    #     server_hess_history = hess
    # else:
    #     server_grad_history = 0.9*server_grad_history + 0.1*grads
    #     server_hess_history = 0.9*server_hess_history + 0.1*hess


    # p_var = cp.Variable(n_clnt, nonneg=True)
    
    # lin = p_var@server_grad_history
    # l2  = 0.5 * alpha * cp.sum_squares(p_var)
    # quad = 0.5 * beta * cp.quad_form(p_var, server_hess_history)

    # objective = lin + l2 + quad
    # _ = cp.Problem(cp.Minimize(objective), [cp.sum(p_var) == 1, p_var >= 0]).solve()

    # return bound_r, p_var.value, server_grad_history, server_hess_history
    return bound_r, server_grad_history, server_hess_history


def solve_ONS_weight_request(ons_weights):
    client_mean_request = ons_weights.mean(axis = 0)
    avg_mean = client_mean_request.mean(axis = 0)
    bias = 1.0/(1.0+np.exp(-1.0*(client_mean_request - avg_mean))) # n_clnt x 1
    weight_req = client_mean_request + bias/50.0
    print(f'bias: {bias}, client_req: {client_mean_request}')
    return weight_req


def B(alpha):
    """
    Computes B(alpha) = (Gamma(alpha_1)*...*Gamma(alpha_k)) / Gamma(alpha_1 + ... + alpha_k)
    where alpha is an array or list of shape parameters alpha_i > 0.
    """
    alpha = np.asarray(alpha, dtype=float)
    return np.prod(gamma(alpha)) / gamma(np.sum(alpha))


def fit_poly_clnt_util_old(data_fit, sweep):
    x = sweep
    y = data_fit

    a = cp.Variable()
    b = cp.Variable()
    c = cp.Variable()

    res = a*x**2 + b*x + c - y
    prob = cp.Problem(
                cp.Minimize(cp.sum_squares(res)),
                [a <= -1e-5] # to fit a concave curve , a<0
            )
    prob.solve()

    return np.array([a.value, b.value, c.value], dtype=np.float32)


def fit_poly_clnt_util(data_fit,sweep):
    x = sweep
    y = data_fit

    a = cp.Variable()
    b = cp.Variable()
    c = cp.Variable()

    res = a*x**2 + b*x + c - y

    constraints = [
        # 1. Fit a concave curve
        a <= -1e-5, 
    
        # 2. Max value of the parabola (at the vertex) <= 1
        c + cp.quad_over_lin(b, -4*a) <= 1,
    
        # 3. Min value >= 0 (checking boundaries at x=0 and x=1)
        c >= 0,
        a + b + c >= 0
    ]

    prob = cp.Problem(cp.Minimize(cp.sum_squares(res)), constraints)
    prob.solve()
    return np.array([a.value, b.value, c.value], dtype=np.float32)




def generate_quad(h_max=0.5, k_range=(0.6, 0.8), a_range=(2, 6)):
    h = np.random.uniform(0.0, h_max)
    k = np.random.uniform(*k_range)
    a = np.random.uniform(*a_range)
    
    a_std = -a
    b = 2 * a * h
    c = -a * h**2 + k
    return np.array([a_std, b, c])


def as_vec(x, S):
    """
    Helper:  if x is scalar  -> return [x, x, …, x] length S
             if x is list/np.ndarray of length S -> return np.array(x)
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:                # scalar
        return np.full(S, arr)
    if arr.size != S:
        raise ValueError("Coefficient length != num_strategies")
    return arr


def client_utility_vec(q_col, a, b, c):
        """Utility of a single client as a vector in ℝ^{S}."""
        return a * cp.square(q_col) + b * q_col + c


def adjust_Q_for_CC_feasibility(
        threshold,           # τ_j, shape (n,)
        clnt_util_params,    # list of (a,b,c) tuples
        beta_vec,            # β_j, shape (n,)
        num_strategies,      # K
        participating_clients,
        strategy_vec,
        dir_param = 0.5,
        kappa   = 1.0,       
        eps     = 1e-8):     
    """
    Phase I: find (P,Q) that minimise total slack on the chance constraints
    w_j(P,Q) ≥ τ_j + Δ_j − s_j,  s_j ≥ 0.
    Returns (P_opt, Q_opt, s_opt), where s_opt shows any remaining violations.
    """
    
    n_strat = num_strategies
    n_client = len(threshold)
    beta_vec = (1/n_client)*np.array(np.ones(n_client))
    # Δ_j = -(1/κ) log(1-β_j)
    delta = (-1.0 / kappa) * np.log(1.0 - np.asarray(beta_vec))
    #delta = 0.01*np.ones(len(beta_vec))
    #delta = -0.5+(-0.0 / kappa) * np.log(1.0 - np.asarray(beta_vec))
    
    poly_coeff_a = clnt_util_params[:,0]
    poly_coeff_b = clnt_util_params[:,1]
    poly_coeff_c = clnt_util_params[:,2]
    alpha_param = [dir_param] * n_strat
    p = np.random.dirichlet(alpha_param,size=1).squeeze()
    
    ####  optimization variables #####
    s = cp.Variable(n_client)
    Q = cp.Variable((n_strat,n_client))
    #import pdb;pdb.set_trace();
    #### objective  #####
    objective = cp.Minimize(0.5 * cp.sum_squares(s))
    ones_clnt = np.ones((n_client,1))
    ones_strat = np.ones((n_strat,1))
    #ones_strat.T @ Q == ones_clnt.T 
    ###### Costraints ######
    
    if n_strat > 1:
        constraints = [s >= 0,
                        Q >= 0,
                        Q @ ones_clnt == ones_strat,
                        ones_strat.T @ Q == (n_strat/n_client) * ones_clnt.T 
                        ]
    else:
         constraints = [s >= 0,
                        Q >= 0,
                        Q @ ones_clnt == ones_strat
                        ]


    for k in range(n_client):
        expr = cp.sum(
        cp.multiply(
            p,
            poly_coeff_a[k] * cp.square(Q[:, k]) +
            poly_coeff_b[k] * Q[:, k] +
            poly_coeff_c[k]
            )
        )   
        #constraints.append(expr >= np.minimum(threshold[k]+delta[k],1.0) - s[k]) uncomment it 
        constraints.append(expr >= threshold[k]+delta[k] - s[k])
        #constraints.append(expr >= threshold[k] - s[k])
    
    #for k in range(n_client):
    #    if participating_clients[k] == 1:
    #        constraints.append(Q[:, k] >= 1/(2*n_client))


    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    opt_wt = np.matmul(p.reshape(1,-1),Q.value).reshape(-1)
    jgap_vec = np.zeros(n_client)

    #print("opt_wt:",opt_wt)
    Q1 = Q.value
    #import pdb;pdb.set_trace();
    for k in range(n_client):
        low_val = np.sum(p * (poly_coeff_a[k] * Q1[:, k]**2 + poly_coeff_b[k] * Q1[:, k] + poly_coeff_c[k]))
        high_val = poly_coeff_a[k]*(opt_wt[k])**2 +  poly_coeff_b[k]*(opt_wt[k]) + poly_coeff_c[k]
        #print("high_val:",high_val)
        #print("low_val:",low_val)
        jgap_vec[k] = high_val - low_val
    #p_s = solve_Jensen_Gap(poly_coeff_a,Q.value,n_client,n_strat)
    return p, Q.value,jgap_vec

def solve_Jensen_Gap(coeff_a,R,n_client,n_strat):
    a =  -1*coeff_a

    R_sq = np.square(R)  # Component-wise square

    s = cp.Variable(n_strat)

    linear_term = s @ (R_sq @ a)

    quadratic_term = cp.sum(cp.multiply(a, cp.square(s @ R)))

    objective = cp.Maximize(linear_term - quadratic_term)
    
    # 5. Simplex Constraints
    constraints = [
        cp.sum(s) == 1,
        s >= 0,
    ]

    # 6. Solve
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return s.value



def check_for_feasibility(threshold, clnt_util_params,P,Q):
    cutil_params_ = clnt_util_params.copy()
    is_ut_satisfied = False
    P_  = P.copy().reshape(-1,1)
    Q_  = Q.copy()
    less_thr_clients = 0
    num_clients = len(threshold)
    for i in range(num_clients):
        weight_ass = (P_.T@Q_)[0][i]
        evaluated_thr = cutil_params_[i][0]*(weight_ass**2) + cutil_params_[i][1]*(weight_ass**1) + cutil_params_[i][2]
        if(evaluated_thr <= threshold[i]):
            less_thr_clients += 1        
    
    if less_thr_clients == num_clients:
        is_ut_satisfied = False
        eps = 1e-6

        j = int(np.random.choice(num_clients))
        a, b, c = cutil_params_[j][0], cutil_params_[j][1], cutil_params_[j][2]

        # a < 0 and threshold feasible, so b**2 - 4a(c - thr) >= 0 and the
        # smaller root is the minimal weight at which client j clears its bar.
        s  = np.sqrt(b*b - 4*a*(c - threshold[j]))
        wj = (-b + s) / (2*a) + eps          # smaller root (a<0 flips the sign), nudged over

        rest = (1.0 - wj) / (num_clients - 1) if num_clients > 1 else 0.0
        w_target = np.full(num_clients, rest, dtype=float)
        w_target[j] = wj

        Q_[:] = w_target[np.newaxis, :]      # P.T @ Q_ == w_target since columns of P sum to 1
        return  is_ut_satisfied, P_,Q_    

    else:
        is_ut_satisfied = True
        print("satisfied clients:", num_clients - less_thr_clients)
        return is_ut_satisfied, P, Q
        
 
                           
        
