# -*- coding: utf-8 -*-
import torch.nn as nn
import networkx as nx
import numpy as np
import matplotlib
import time,logging
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc. depending on your setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import numpy.matlib
from scipy.linalg import block_diag
import cvxpy as cp
import numpy as np
from numpy.linalg import solve
import scipy.sparse as sp
import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import nnls
from util_func import *
from change_detection_module import ChangeDetectionMethod
import traceback
np.random.seed(0)

MSE_FULL = 'mse_full'
F1_FULL = 'f1_full'
EIER_FULL = 'eier_full'
MSE_SPARSE = 'mse_sparse'
F1_SPARSE = 'f1_sparse'
EIER_SPARSE = 'eier_sparse'
MSE_HYBRID = 'mse_hybrid'
F1_HYBRID = 'f1_hybrid'
EIER_HYBRID = 'eier_hybrid'
MSE_ORACLE = 'mse_oracle'
F1_ORACLE = 'f1_oracle'
EIER_ORACLE = 'eier_oracle'
MSE = "mse"
F1 = "f1"
EIER = "eier"

from enum import Enum


class KFMethods(Enum):
    MSE_FULL = 'mse_full'
    F1_FULL = 'f1_full'
    MSE_SPARSE = 'mse_sparse'
    F1_SPARSE = 'f1_sparse'
    MSE_HYBRID = 'mse_hybrid'
    F1_HYBRID = 'f1_hybrid'
    MSE_ORACLE = 'mse_oracle'
    F1_ORACLE = 'f1_oracle'


def generate_q(n):
    return np.random.normal(size=(n, 1))


def generate_H(B, q):
    return np.dot(B, vector2diag(np.dot(B.T, q)))


def generate_y(H, x, C_w_sqrt):
    observation = np.dot(H, x) + np.dot(C_w_sqrt, np.random.normal(size=(H.shape[0], 1)))
    return observation


def state_evolvment(F, state, C_u):
    new_state = np.dot(F, state) + np.dot(C_u, np.random.normal(size=(state.size, 1)))
    return new_state


def calc_mse(est, true_state):
    mse = np.dot((est - true_state).T, (est - true_state)) / len(true_state)
    return mse


def calc_error_in_existing_edges(est, true_state):
    est_edge_det = set(np.where(est == 0)[0])
    true_edge_det = set(np.where(true_state == 0)[0])
    return len(est_edge_det.symmetric_difference(true_edge_det))


def calc_f1_score(est, true_state):
    est_edge_det = set(np.where(est >= 0.1)[0])
    true_edge_det = set(np.where(true_state >= 0.1)[0])
    tp = len(est_edge_det.intersection(true_edge_det))
    fp_fn = len(est_edge_det.symmetric_difference(true_edge_det))
    return 2 * tp / max(1, (2 * tp + fp_fn))


def calc_edge_identification_error_rate_score(est, true_state):
    est_edge_det = set(np.where(est >= 0.1)[0])
    true_edge_det = set(np.where(true_state >= 0.1)[0])
    edge_identification_error = len(est_edge_det.symmetric_difference(true_edge_det))
    return 100 * edge_identification_error / (2 * len(est))


def single_update_iteration(state, F, B, C_w, C_u, N, k):
    state = state_evolvment(F, state, C_u)
    q = generate_q(N)
    H = generate_H(B, q)
    observation = generate_y(H, state, C_w)
    return q, state, observation


# def get_trajectory2(trajectory_time, stateInit, F, B, C_w, C_u, N, k):
#     position = []
#     measurements_q = []
#     measurements_y = []
#     # intial state - zero position and constant velocity
#     state = stateInit
#     for t in trajectory_time:
#         q, state, observation = single_update_iteration(state, F, B, C_w, C_u, N, k)
#         measurements_q.append(q)
#         position.append(state)
#         measurements_y.append(observation)
#     return position, measurements_q, measurements_y


def truncate_matrix(x_est, thr):
    zero_idx = np.where(x_est < thr)
    x_est[zero_idx] = 0
    return x_est


def plot_state_as_graph(x, B, title_str):
    # print(z)
    L = build_L(B, x)
    # print(f"L={L}")
    Adj = np.diag(np.diag(L)) - L
    # print(Adj)
    print(Adj)
    G_t = nx.from_numpy_array(Adj)
    # G_t = nx.complete_graph(n, create_using=None)
    fig, ax = plt.subplots()
    # nx.draw(G_t, ax=ax, with_labels=True)
    # pos = nx.draw_circular(G_t)
    pos = nx.circular_layout(G_t)
    # print(pos)
    # nx.draw_networkx(G_t,pos)
    labels = nx.get_edge_attributes(G_t, 'weight')
    # print(labels)
    nx.draw(G_t, pos)
    nx.draw_networkx_edge_labels(G_t, pos, edge_labels=labels)
    # nx.draw(G, pos=pos)
    # # nx.draw_circular(G_t, with_labels=True, edge_labels=labels, width=4)
    ax.set_title(title_str)


def nnls_wls(A, y, R):
    """
    Solve   min_{x >= 0} (y - A x)^T R^{-1} (y - A x)
    by transforming into poly_c standard NNLS call.
    """
    # 1. Compute poly_c Cholesky factor of R (R = Rt^T * Rt)
    Rt = cholesky(R, lower=False)  # so R = Rt^T @ Rt

    # 2. Form B = R^{-1/2} A and z = R^{-1/2} y
    #    R^{-1/2} = inv(Rt)
    R_half_inv = np.linalg.inv(Rt)
    B = R_half_inv @ A
    z = R_half_inv @ y

    # 3. Solve NNLS:  min_{x >= 0} ||B x - z||^2
    x_nnls, residual = nnls(B, z)
    return x_nnls


def compute_Jacobian_poly_dp(L, B, q, a, edges):
    """Efficient Jacobian for f(L)=Σ a_p L^p  (sparse version)."""
    # q = q.reshape(-1, 1)

    N, E_max = B.shape
    P = len(a)
    I = np.eye(N)

    # c_p = L^p q
    c = [q]
    for _ in range(1, P):
        c.append(L @ c[-1])

    # D_p recursion
    D = [np.zeros((N,N))] * P
    D[P - 1] = np.zeros((N, N))  # D_{P-1}=0

    for p in range(P - 2, -1, -1):  # p = P-2 … 0
        D[p] = L @ D[p + 1] + a[p + 1] * I

    H = np.zeros((N, B.shape[1]), dtype=L.dtype)
    for j in range(B.shape[1]):
        (n, m) = edges[j]
        col = np.zeros((N, 1), dtype=L.dtype)
        for p in range(P):
            temp_col = (c[p][n]-c[p][m]) * (D[p][:,n]-D[p][:,m])
            col += temp_col.reshape(-1,1)
        H[:, j] = col.ravel()
    return H


def compute_Jacobian_poly_dp_sparse(L: sp.spmatrix,
                                    B: sp.spmatrix,
                                    q: np.ndarray,
                                    a: np.ndarray) -> np.ndarray:
    """Efficient Jacobian for f(L)=Σ a_p L^p  (sparse version)."""
    L = sp.csr_matrix(L)
    B = sp.csr_matrix(B)
    q = q.reshape(-1, 1)

    N, E_max = B.shape
    P = len(a)
    I = sp.identity(N, format="csr", dtype=L.dtype)

    # c_p = L^p q
    c = [q]
    for _ in range(1, P):
        c.append(L @ c[-1])

    # D_p recursion
    D = [None] * P
    D[P - 1] = sp.csr_matrix((N, N), dtype=L.dtype)  # D_{P-1}=0

    for p in range(P - 2, -1, -1):  # p = P-2 … 0
        D[p] = L @ D[p + 1] + a[p + 1] * I  # <-- note poly_c[p+1] !

    H = np.zeros((N, B.shape[1]), dtype=L.dtype)
    for j in range(B.shape[1]):
        v = B[:, j].toarray()
        col = np.zeros((N, 1), dtype=L.dtype)
        for p in range(P):
            s_p = float(v.T @ c[p])
            if s_p:
                col += s_p * (D[p] @ v)
        H[:, j] = col.ravel()
    return H


def compute_Jacobian_poly(L, B, q, a):
    """
    Compute the j-th column of the Jacobian H_t given by:
        [H_t]_j = sum_i a_i ( sum_{k=0}^{i-1} L^k (B E_j B^T) L^{i-k-1} q )

    Parameters
    ----------
    L : np.ndarray
        Laplacian matrix, shape (N, N).
    B : np.ndarray
        Incidence matrix, shape (N, E_max).
    q : np.ndarray
        Input vector q_t, shape (N,).
    a : np.ndarray
        Polynomial coefficients of f(L): f(L)=sum_i a_i L^i, shape (m+1,).

    Returns
    -------
    H_col : np.ndarray
        The j-th column of the Jacobian H_t, shape (N,).
    """
    # Dimensions
    N = L.shape[-1]
    E_max = B.shape[1]
    # Maximum power needed
    max_power = len(a) - 1  # If poly_c = [a_0, a_1, ..., a_m], need L^m

    # Precompute powers of L
    L_powers = [np.eye(N)]  # L^0 = I
    for _ in range(max_power):
        L_powers.append(np.matmul(L_powers[-1], L))
    H = np.zeros([N, E_max])
    for j in range(E_max):
        # E_j with poly_c single 1 at position (1,1), for example:
        E_j = np.zeros((E_max, E_max))
        E_j[j, j] = 1.0
        # Compute M_j = B E_j B^T
        M_j = np.matmul(B, np.matmul(E_j, B.T))

        # Initialize result vector
        H_col = np.zeros([N, 1])

        # Compute the sum for each i
        # Note: i starts from 1 since for i=0 there's no inner sum.
        # If polynomial is a_0 + a_1 L + ... + a_m L^m,
        # then i runs from 1 to m for the derivative terms.
        for i_idx in range(1, len(a)):  # i_idx corresponds to i in the formula
            ai = a[i_idx]
            if ai == 0:
                continue
            inner_sum = np.zeros([N, 1])

            # sum_{k=0}^{i-1} L^k (M_j) L^{i-k-1} q
            # i_idx - 1 is the highest exponent in the second part
            for k in range(i_idx):
                # L^k
                Lk = L_powers[k]
                # L^{i-k-1}
                L_i_k_1 = L_powers[i_idx - k - 1]

                # Compute L^k (M_j) L^{i-k-1} q
                # First compute temp = L^{i-k-1} q
                temp = np.matmul(L_i_k_1, q)
                # Then (M_j) temp
                temp = np.matmul(M_j, temp)
                # Then L^k temp
                temp = np.matmul(Lk, temp)

                inner_sum = inner_sum + temp

            # Multiply by a_i
            H_col += ai * inner_sum
        H[:, j:j + 1] = H_col

    return H


class KalmanFilt(nn.Module):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit):
        super(KalmanFilt, self).__init__()

        self.F = F
        self.F_s = self.F
        self.B = B
        self.V = C_u
        self.V_s = self.V
        self.W = C_w

        # tracked sigma
        self.Sigma = C_x
        # tracked state
        self.s = StateInit

    def predict(self):
        # Update time state
        self.s = np.dot(self.F_s, self.s)

        # Calculate error covariance
        # Sigma= F_s*Sigma*F_s' + V_s
        self.Sigma = np.dot(np.dot(self.F_s, self.Sigma), self.F_s.T) + self.V_s

    def update(self, q, y):
        # S = H*P*H'+W
        H = generate_H(self.B, q)
        S = np.dot(H, np.dot(self.Sigma, H.T)) + self.W

        # Calculate the Kalman Gain
        # K = Sigma * H'* inv(H*Sigma*H'+W)
        K = np.dot(np.dot(self.Sigma, H.T), np.linalg.inv(S))
        self.s = (self.s + np.dot(K, (y - np.dot(H, self.s))))
        I = np.eye(H.shape[1])
        self.Sigma = np.dot(np.dot((I - np.dot(K, H)), self.Sigma), (I - np.dot(K, H)).T) + np.dot(K,
                                                                                                   np.dot(self.W, K.T))

    def forward(self, q, y, *args, **kwargs):
        # First perdict next state without observation
        self.predict()
        # Then use observation to refine update
        self.update(q, y)
        self.s = np.maximum(self.s, 0)
        return self.s


class ExtendedKalmanFilter(KalmanFilt):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c):
        self.a = poly_c
        super(ExtendedKalmanFilter, self).__init__(F, B, C_u, C_w, C_x, StateInit)

    def update(self, q, y):
        L = build_L(self.B, self.s)
        # S = H*P*H'+W
        H = compute_Jacobian_poly(L, self.B, q, self.a)
        S = np.dot(H, np.dot(self.Sigma, H.T)) + self.W

        # Calculate the Kalman Gain
        # K = Sigma * H'* inv(H*Sigma*H'+W)
        K = np.dot(np.dot(self.Sigma, H.T), np.linalg.inv(S))
        self.s = (self.s + np.dot(K, (y - compute_poly(L, q, self.a))))
        I = np.eye(H.shape[1])
        self.Sigma = np.dot(np.dot((I - np.dot(K, H)), self.Sigma), (I - np.dot(K, H)).T) + np.dot(K,
                                                                                                   np.dot(self.W, K.T))

class FastExtendedKalmanFilter(KalmanFilt):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c):
        self.a = poly_c
        super(FastExtendedKalmanFilter, self).__init__(F, B, C_u, C_w, C_x, StateInit)
        self.edges = map_B_to_set(B)

    def update(self, q, y):
        L = build_L(self.B, self.s)
        # S = H*P*H'+W
        H = compute_Jacobian_poly_dp(L, self.B, q, self.a, self.edges)
        S, K, Sigma = kalman_gain_imp(H, self.Sigma, self.W)
        # S = (np.dot(H, np.dot(self.Sigma, H.T)) + np.dot(H, np.dot(self.Sigma, H.T)).T)/2 + self.W

        # Calculate the Kalman Gain
        # K = Sigma * H'* inv(H*Sigma*H'+W)
        # K = np.dot(np.dot(self.Sigma, H.T), pseudo_inv(S, 1e-3))
        self.s = (self.s + np.dot(K, (y - compute_poly(L, q, self.a))))
        # I = np.eye(H.shape[1])
        # self.Sigma = np.dot(np.dot((I - np.dot(K, H)), self.Sigma), (I - np.dot(K, H)).T) + np.dot(K, np.dot(self.W, K.T))
        self.Sigma = Sigma

class sparseKalmanFilter(FastExtendedKalmanFilter):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c, thr):
        super(sparseKalmanFilter, self).__init__(F, B, C_u, C_w, C_x, StateInit, poly_c)
        self.thr = thr

    def forward(self, q, y, *args, **kwargs):
        # First perdict next state without observation
        super(sparseKalmanFilter, self).forward(q, y)
        self.s = truncate_matrix(self.s, self.thr)
        return self.s


class sparseKalmanFilterISTA(FastExtendedKalmanFilter):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c, lambda_1):
        super(sparseKalmanFilterISTA, self).__init__(F, B, C_u, C_w, C_x, StateInit, poly_c)
        self.lambda_1 = lambda_1

    def update(self, q, y):
        L = build_L(self.B, self.s)
        Delta_Y_t = y - compute_poly(L, q, self.a)
        H = compute_Jacobian_poly_dp(L, self.B, q, self.a, self.edges)
        # Optimization variable
        x = cp.Variable((self.B.shape[1], 1), nonneg=True)
        # Objective function
        # Instead of: u.T @ inv(W) @ u
        # Use: cp.sum_squares(W^{-1/2} @ u), where W^{-1/2} is computed safely

        from scipy.linalg import sqrtm
        W_sqrt_inv = np.linalg.inv(sqrtm(self.W))
        Sigma_sqrt_inv = np.linalg.inv(sqrtm(self.Sigma))
        beta = self.lambda_1

        objective = cp.Minimize(
            cp.sum_squares(W_sqrt_inv @ (Delta_Y_t - H @ (x - self.s))) +
            cp.sum_squares(Sigma_sqrt_inv @ (x - self.s)) +
            beta * cp.norm1(x)
        )

        # Define and solve the problem
        problem = cp.Problem(objective)
        problem.solve(solver="MOSEK")


        # Solution
        self.s = x.value
        S, K, self.Sigma = kalman_gain_imp(H, self.Sigma, self.W)
        # S = np.dot(H, np.dot(self.Sigma, H.T)) + self.W
        #
        # # Calculate the Kalman Gain
        # K = np.dot(np.dot(self.Sigma, H.T), np.linalg.inv(S))
        # I = np.eye(H.shape[1])
        # self.Sigma = (
        #         np.dot(np.dot((I - np.dot(K, H)), self.Sigma), (I - np.dot(K, H)).T) + np.dot(K, np.dot(self.W, K.T)))

    def forward(self, q, y, *args, **kwargs):
        # First perdict next state without observation
        super(sparseKalmanFilterISTA, self).forward(q, y)
        # self.s = truncate_matrix(self.s, self.thr)
        return self.s


class oraclKalmanFilt_nocovariance_update(ExtendedKalmanFilter):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c):
        super(oraclKalmanFilt_nocovariance_update, self).__init__(F, B, C_u, C_w, C_x, StateInit, poly_c)

    def forward(self, q, y, *args):
        updated_connections = args[0]
        super(oraclKalmanFilt_nocovariance_update, self).predict()
        disconnections = np.setdiff1d(np.arange(self.s.shape[0]), updated_connections)
        # self.s[disconnections] = 0
        super(oraclKalmanFilt_nocovariance_update, self).update(q, y)
        self.s[disconnections] = 0
        return self.s

class oraclKalmanFilt_diagonalovariance_update(ExtendedKalmanFilter):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c):
        super(oraclKalmanFilt_diagonalovariance_update, self).__init__(F, B, C_u, C_w, C_x, StateInit, poly_c)

    def forward(self, q, y, *args):
        updated_connections = args[0]
        super(oraclKalmanFilt_diagonalovariance_update, self).predict()
        disconnections = np.setdiff1d(np.arange(self.s.shape[0]), updated_connections)
        # self.s[disconnections] = 0
        super(oraclKalmanFilt_diagonalovariance_update, self).update(q, y)
        self.s[disconnections] = 0

        self.Sigma[np.ix_(updated_connections, updated_connections)] += np.eye(len(updated_connections))
        return self.s


class oraclKalmanFilt_paper(FastExtendedKalmanFilter):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c, new_connections_uncertainty_weight):
        super(oraclKalmanFilt_paper, self).__init__(F, B, C_u, C_w, C_x, StateInit, poly_c)
        self.new_connections_uncertainty_weight = new_connections_uncertainty_weight

    def update(self, q, y, connections):
        connections_int = np.asarray(connections, dtype=int)
        old_connections = np.asarray(np.where(self.s > 0)[0], dtype=int)
        new_connections = np.setdiff1d(connections_int, old_connections)
        new_connections_uncertainty_matrix = np.zeros_like(self.Sigma)
        # new_connections_uncertainty_matrix[np.ix_(new_connections, new_connections)] = np.max(self.Sigma) # assuming that the matrix is diagonal
        self.s[new_connections] = self.new_connections_uncertainty_weight
        L = build_L(self.B, self.s)
        # S = H*P*H'+W
        H = compute_Jacobian_poly_dp(L, self.B, q, self.a, self.edges)
        H = H[:, connections]
        # Concatenate H and the identity matrix vertically
        uncertainty_sigma = self.Sigma # sigma_with_new_connections_uncertainty
        uncertainty_sigma_small = uncertainty_sigma[np.ix_(connections, connections)]
        S, K, Sigma_small = kalman_gain_imp(H, uncertainty_sigma_small, self.W)
        # S = (np.dot(H, np.dot(uncertainty_sigma_small, H.T)) + np.dot(H, np.dot(uncertainty_sigma_small, H.T)).T)/2 + self.W
        # Calculate the Kalman Gain
        # K  = np.dot(np.dot(uncertainty_sigma_small, H.T), pseudo_inv(S, 1e-3))
        self.s[connections] = (self.s[connections] + np.dot(K, (y - compute_poly(L, q, self.a))))
        # I = np.eye(H.shape[1])
        # Sigma_small = np.dot(np.dot((I - np.dot(K, H)), uncertainty_sigma_small),
        #                      (I - np.dot(K, H)).T) + np.dot(K, np.dot(self.W, K.T))
        self.Sigma = np.zeros_like(self.Sigma)
        self.Sigma[np.ix_(connections, connections)] = Sigma_small

    def forward(self, q, y, *args):
        updated_connections = args[0]
        super(oraclKalmanFilt_paper, self).predict()
        disconnections = np.setdiff1d(np.arange(self.s.shape[0]), updated_connections)
        self.s[disconnections] = 0
        self.update(q, y, updated_connections)
        self.s = np.maximum(self.s, 0)
        return self.s


class oraclKalmanFilt_paper_delayed(ExtendedKalmanFilter):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c):
        super(oraclKalmanFilt_paper_delayed, self).__init__(F, B, C_u, C_w, C_x, StateInit, poly_c)

    def update_covariance(self, q, connections):

        L = build_L(self.B, self.s)
        # S = H*P*H'+W
        H = compute_Jacobian_poly_dp(L, self.B, q, self.a)
        H = H[:, connections]
        # Concatenate H and the identity matrix vertically
        S = np.dot(H, np.dot(self.Sigma[np.ix_(connections, connections)], H.T)) + self.W
        # Calculate the Kalman Gain
        K = np.dot(np.dot(self.Sigma[np.ix_(connections, connections)], H.T), np.linalg.inv(S))
        I = np.eye(H.shape[1])
        Sigma_small = np.dot(np.dot((I - np.dot(K, H)), self.Sigma[np.ix_(connections, connections)]),
                             (I - np.dot(K, H)).T) + np.dot(K, np.dot(self.W, K.T))
        self.Sigma = np.zeros_like(self.Sigma)
        self.Sigma[np.ix_(connections, connections)] = Sigma_small

    def forward(self, q, y, *args):
        updated_connections = args[0]
        super(oraclKalmanFilt_paper_delayed, self).predict()
        super(oraclKalmanFilt_paper_delayed, self).update(q,y)
        disconnections = np.setdiff1d(np.arange(self.s.shape[0]), updated_connections)
        self.s[disconnections] = 0
        self.update_covariance(q, updated_connections)
        return self.s


class oraclKalmanFilt3(ExtendedKalmanFilter):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c):
        super(oraclKalmanFilt3, self).__init__(F, B, C_u, C_w, C_x, StateInit, poly_c)

    def forward(self, q, y, *args):
        updated_connections = args[0]
        F_s = sample_matrix(np.copy(self.F), updated_connections)
        self.F_s = F_s
        C_u_s = sample_matrix(np.copy(self.V), updated_connections)
        self.V_s = C_u_s
        super(oraclKalmanFilt3, self).forward(q, y)
        self.s = np.maximum(self.s, 0)
        return self.s


class SBL_EKF(ExtendedKalmanFilter):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c, mu):
        super(SBL_EKF, self).__init__(F, B, C_u, C_w, C_x, StateInit, poly_c)
        self.mu = mu

    def update(self, q, y):
        L = build_L(self.B, self.s)
        H = compute_Jacobian_poly_dp(L, self.B, q, self.a)
        R = self.W # measurement & noise
        x_pred = self.s
        Sigma_pred = self.Sigma # prior mean & cov
        mu = self.mu
        max_iter = 50
        eps = 1e-9
        tol = 1e-6

        n = x_pred.size
        alpha = np.full(n, 1.0 / mu)  # initialise αᵢ
        x = x_pred.copy()

        R_inv = np.linalg.inv(R)
        SigmaInv = np.linalg.inv(Sigma_pred)
        SigmaInv = 0.5 * (SigmaInv + SigmaInv.T)
        for it in range(max_iter):
            # ----- M-step : quadratic solve for x  --------------------------
            Alpha_inv = np.diag(1.0 / (alpha + eps))
            # block system  (Σ_pred⁻¹ + HᵀR⁻¹H + diag(α)⁻¹) x = rhs
            A = SigmaInv + H.T @ R_inv @ H + Alpha_inv
            b = (SigmaInv @ x_pred + H.T @ R_inv @ y)
            x_new = np.abs(solve(A, b))
            # x_new[x_new < 0] = 0
            # convergence check
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new

            # ----- E-step : update αᵢ  --------------------------------------
            alpha = np.maximum(np.abs(x) / mu, eps)

        # effective posterior covariance  (inverse Hessian)
        Sigma_eff = np.linalg.inv(A)
        self.s = x
        self.Sigma += np.diag(alpha)
        # alpha
        S = np.dot(H, np.dot(self.Sigma, H.T)) + self.W
        K = np.dot(np.dot(self.Sigma, H.T), np.linalg.inv(S))
        # self.s = (self.s + np.dot(K, (y - compute_poly(L, q, self.poly_c))))
        I = np.eye(H.shape[1])
        self.Sigma = (np.dot(np.dot((I - np.dot(K, H)), self.Sigma), (I - np.dot(K, H)).T)
                      + np.dot(K, np.dot(self.W, K.T)))

    def forward(self, q, y, *args):
        super(SBL_EKF, self).predict()
        # self.s[disconnections] = 0
        self.update(q, y)
        # self.Sigma[np.ix_(updated_connections, updated_connections)] += np.eye(len(updated_connections))
        return self.s


def sample_matrix(matrix, idx_rows, idx_columns=None):
    no_connection = np.setdiff1d(np.arange(matrix.shape[0]), idx_rows)
    if no_connection.size > 0:
        if matrix.shape[1] > 1:
            matrix[no_connection, :] = 0
            if not (idx_columns is None):
                no_connection = np.setdiff1d(np.arange(matrix.shape[1]), idx_columns)
            matrix[:, no_connection] = 0
        else:
            matrix[no_connection] = 0
    return matrix


def single_smooth_update_iteration(state, F, B, C_w, C_u, N, k):
    state = state_evolvment(F, state, C_u)
    L = build_L(B, state)
    inv_sqrt_L = inv_sqrt_singular_matrix(L)
    q = np.dot(inv_sqrt_L, generate_q(N))
    H = generate_H(B, q)
    observation = generate_y(H, state, C_w)
    return q, state, observation


def get_trajectory(trajectory_time, F, B, C_w_sqrt, C_u_sqrt, N, k, poly_c, new_edge_weight, num_edges_stateinit, delta_n):
    position = []
    measurements_q = []
    measurements_y = []
    m = num_possible_edges(N)
    idx_list = chose_indices_without_repeating(m, num_edges_stateinit)
    stateInit = np.zeros(m).reshape([m, 1])
    stateInit[idx_list] = new_edge_weight
    state = stateInit.copy()
    updated_connections_list = []
    data_len = trajectory_time.size
    for i in range(data_len):
        if (i % k == 0):
            binary_state = (state > 0).astype(float)
            updated_connections, new_binary_state = change_edge_set(np.copy(binary_state),delta_n)
            F_s = sample_matrix(np.copy(F), updated_connections)
            C_u_s_sqrt = sample_matrix(np.copy(C_u_sqrt), updated_connections)
            new_edges_indicator = ((new_binary_state - binary_state) > 0).astype(float)
            if np.sum(new_edges_indicator) > 0:
                print("new edge was added")
            state = generate_y(F_s, np.copy(state) + new_edge_weight * new_edges_indicator, C_u_s_sqrt)
        else:
            updated_connections = np.where(state > 0)[0]
            updated_connections = np.sort(updated_connections)
            C_u_s_sqrt = sample_matrix(np.copy(C_u_sqrt), updated_connections)
            F_s = sample_matrix(np.copy(F), updated_connections)
            state = generate_y(F_s, np.copy(state), C_u_s_sqrt)
        state = np.abs(state)
        q, observation = compute_measurements(build_L(B, np.copy(state)), C_w_sqrt, N, poly_c)
        updated_connections_list.append(updated_connections)
        state = np.abs(np.copy(state))
        measurements_q.append(q)
        position.append(np.copy(state))
        measurements_y.append(observation)
    return position, measurements_q, measurements_y, updated_connections_list, stateInit


def change_edge_set(state, p=1):
    # idx = np.random.choice(len(state))
    indices = np.random.choice(len(state), size=p, replace=False)
    # Change the value at the chosen index (flip between 0 and 1)
    state[indices] = 1 - state[indices]
    # Append the new vector to the list
    updated_connections = np.where(state > 0)[0]
    updated_connections = np.sort(updated_connections)
    return updated_connections, state


def compute_measurements(L, C_w_sqrt, N, a):
    q = generate_q(N)
    f = compute_poly(L, q, a)
    observation = generate_y(np.eye(N), f, C_w_sqrt)
    return q, observation


def inv_sqrt_singular_matrix(L):
    U, S, Vh = np.linalg.svd(L, full_matrices=True)
    S[S > 0.1] = np.power(S[S > 0.1], -0.5)
    return np.dot(U, np.dot(np.diag(S), Vh))


def one_method_evaluation(kf, measurements_q, measurements_y, position, updated_connections_list):
    mse_list = []
    F1_score_list = []
    EIER_list = []
    times_list = []
    for i, (q, y, true_state, updated_connections) in enumerate(
            zip(measurements_q, measurements_y, position, updated_connections_list)):
        start = time.time()
        x_est = kf(q, y, updated_connections)
        end = time.time()
        elapsed = end - start
        mse = calc_mse(x_est, true_state)
        logging.info(f"Iteration {i}: Execution time = {elapsed:.6f} seconds, mse={mse[0][0]}")
        mse_list.append(mse)
        F1_score_list.append(calc_f1_score(x_est, true_state))
        EIER_list.append(calc_edge_identification_error_rate_score(x_est, true_state))
        times_list.append(elapsed)
        if mse[0][0].item() >3:
            aa=3
        x_est_prev = x_est
    return np.concatenate(mse_list), np.array(F1_score_list).reshape([i + 1, 1]), np.array(EIER_list).reshape(
        [i + 1, 1]), times_list


def single_monte_carlo_iteration(F, B, C_u, C_w, C_x_missmatch, C_x, stateInit_missmatch, trajectory_time, stateInit, n,
                                 k, thr1, thr2, poly_coefficients, iterable):
    return single_monte_carlo(F, B, C_u, C_w, C_x_missmatch, C_x, stateInit_missmatch, trajectory_time, stateInit, n,
                              k, thr1, thr2, poly_coefficients)


def single_monte_carlo(F, B, C_u, C_w, C_x_missmatch, C_x, stateInit_missmatch, trajectory_time, stateInit, n,
                       k, thr1, thr2, poly_coefficients):
    # Generate data
    position, measurements_q, measurements_y, updated_connections_list = get_trajectory(trajectory_time, stateInit, F,
                                                                                        B, C_w, C_u, n, k, poly_coefficients)
    # Initialize methods
    kf = ExtendedKalmanFilter(F, B, C_u, C_w, C_x_missmatch, stateInit_missmatch, poly_coefficients)
    sparse_kf = sparseKalmanFilter(F, B, C_u, C_w, C_x_missmatch, stateInit_missmatch, poly_coefficients, thr1)
    hybrid_kf = sparseKalmanFilterISTA(F, B, C_u, C_w, C_x_missmatch, stateInit_missmatch, poly_coefficients, lambda_1=0.5)
    kf_oracle = oraclKalmanFilt_diagonalovariance_update(F, B, C_u, C_w, C_x, stateInit, poly_coefficients)
    # sparse_kf = oraclKalmanFilt_nocovariance_update(F, B, C_u, C_w, C_x, stateInit, poly_coefficients)
    # hybrid_kf = oraclKalmanFilt_paper(F, B, C_u, C_w, C_x, stateInit, poly_coefficients)
    # kf_oracle = oraclKalmanFilt_paper(F, B, C_u, C_w, C_x, stateInit, poly_coefficients)
    change_detection_obj = ChangeDetectionMethod(B, n, stateInit_missmatch, poly_coefficients)

    # Apply methods
    mse_full, F1_score_full, EIER_full = one_method_evaluation(kf, measurements_q, measurements_y, position,
                                                               updated_connections_list)
    mse_sparse, F1_score_sparse, EIER_sparse = one_method_evaluation(sparse_kf, measurements_q, measurements_y,
                                                                     position,
                                                                     updated_connections_list)
    mse_hybrid, F1_score_hybrid, EIER_hybrid = one_method_evaluation(hybrid_kf, measurements_q, measurements_y,
                                                                     position,
                                                                     updated_connections_list)
    mse_oracle, F1_score_oracle, EIER_oracle = one_method_evaluation(kf_oracle, measurements_q, measurements_y,
                                                                     position,
                                                                     updated_connections_list)
    # mse_oracle, F1_score_oracle, EIER_oracle = one_method_evaluation(change_detection_obj, measurements_q, measurements_y,
    #                                                                  position,
    #                                                                  updated_connections_list)
    if np.average(mse_oracle[20:40]) > 10:
        for table_name in ["position", "measurements_q", "measurements_y", "updated_connections_list"]:
            print(table_name)
            np.save(table_name + "1", np.array(eval(table_name), dtype=object), allow_pickle=True)

    # Organize output
    kwargs = {MSE_FULL: mse_full,
              F1_FULL: F1_score_full,
              EIER_FULL: EIER_full,
              MSE_SPARSE: mse_sparse,
              F1_SPARSE: F1_score_sparse,
              EIER_SPARSE: EIER_sparse,
              MSE_HYBRID: mse_hybrid,
              F1_HYBRID: F1_score_hybrid,
              EIER_HYBRID: EIER_hybrid,
              MSE_ORACLE: mse_oracle,
              F1_ORACLE: F1_score_oracle,
              EIER_ORACLE: EIER_oracle,
              }
    return kwargs  # mse_full, F1_score_full, mse_sparse, F1_score_sparse, mse_hybrid, F1_score_hybrid, mse_oracle, F1_score_oracle


def run_monte_carlo_simulation(F, B, C_u, C_w, C_x_missmatch, C_x, stateInit_missmatch, trajectory_time, stateInit,
                               n, k, thr1, thr2, num_iterations, poly_coefficients):
    errors = []
    for i in range(num_iterations):
        error = single_monte_carlo_iteration(F, B, C_u, C_w, C_x_missmatch, C_x, stateInit_missmatch,
                                             trajectory_time, stateInit, n, k, thr1, thr2, poly_coefficients, i)
        errors.append(error)
    return errors


def restore_specific_mse(list_dict, method_name, num_time_samples):
    mse = np.zeros([num_time_samples, 1])
    F1_score = np.zeros([num_time_samples, 1])
    eier = np.zeros([num_time_samples, 1])
    for i, e in enumerate(error):  # list of tuples
        mse = mse + e[f"{MSE}_{method_name}"]
        F1_score = F1_score + e[f"{F1}_{method_name}"]
        eier = eier + e[f"{EIER}_{method_name}"]
    return np.divide(mse, i + 1), np.divide(F1_score, i + 1), np.divide(eier, i + 1)


if __name__ == "__main__":
    num_time_samples = 150
    sigma_v = 0.1
    sigma_w = 0.1
    sigma_x = 0.5
    n = 10
    thr1 = 0.2
    thr2 = 0.2
    num_iterations = 10
    k = 50  # num of time samples to concat
    m = num_possible_edges(n)
    num_edges = 3 * n
    F = np.dot(1, np.eye(num_edges))
    W = np.dot(sigma_w, np.eye(n))
    C_w = W
    G = nx.complete_graph(n, create_using=None)
    B = nx.incidence_matrix(G, oriented=True).todense()
    idx_list = chose_indices_without_repeating(m, num_edges)
    trajectory_time = np.arange(0, num_time_samples)
    F = np.dot(1, np.eye(m))
    idx_list = chose_indices_without_repeating(m, num_edges)
    stateInit = np.zeros(m).reshape([m, 1])
    stateInit[idx_list] = 1
    C_u = np.dot(sigma_v ** 2, np.eye(m))
    stateInit_missmatch = np.ones(m).reshape([m, 1])
    C_x = np.dot(sigma_x ** 2, vector2diag(stateInit))
    C_x_missmatch = np.dot(sigma_x ** 2, np.eye(m))
    poly_a = np.array([1.0, 1.0, 0.1, 2.0])

    error = run_monte_carlo_simulation(F, B, C_u, C_w, C_x_missmatch, C_x, stateInit_missmatch,
                                       trajectory_time, stateInit, n, k, thr1, thr2, num_iterations, poly_coefficients=poly_a)
    # For debuging
    # loaded_data = {}
    #
    # # List of saved variable names
    # table_names = ["position", "measurements_q", "measurements_y", "updated_connections_list"]
    #
    # # Load each .npy file
    # for table_name in table_names:
    #     loaded_data[table_name] = np.load(f"{table_name}.npy", allow_pickle=True)
    # measurements_q = loaded_data["measurements_q1"]
    # measurements_y = loaded_data["measurements_y1"]
    # position = loaded_data["position1"]
    # updated_connections_list = list(loaded_data["updated_connections_list1"])
    # kf_oracle = oraclKalmanFilt3(F, B, C_u, C_w, C_x, stateInit)
    #
    # # Apply methods
    # mse_oracle, F1_score_oracle, EIER_oracle = one_method_evaluation(kf_oracle, measurements_q, measurements_y,
    #                                                                  position,
    #                                                                  updated_connections_list)

    mse_full, F1_score_full, EIER_full = restore_specific_mse(error, 'full', num_time_samples)
    mse_sparse, F1_score_sparse, EIER_sparse = restore_specific_mse(error, 'sparse', num_time_samples)
    mse_hybrid, F1_score_hybrid, EIER_hybrid = restore_specific_mse(error, 'hybrid', num_time_samples)
    mse_oracle, F1_score_oracle, EIER_oracle = restore_specific_mse(error, 'oracle', num_time_samples)

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure()
    plt.plot(trajectory_time, np.log10(mse_full), label='EKF', color='b', linewidth=2)
    plt.plot(trajectory_time, np.log10(mse_sparse), label='GSP-EKF', color='g', linestyle=':', linewidth=2)
    plt.plot(trajectory_time, np.log10(mse_hybrid), label='GSP-EKF-no propagation', color='orange', linestyle='--',
             linewidth=2)
    plt.plot(trajectory_time, np.log10(mse_oracle), label='ChangeDetection', color='r', linestyle='-.', linewidth=2)
    # plt.yscale('symlog')
    plt.legend()
    plt.ylabel("MSE [dB]")
    plt.xlabel("l [time units]")
    plt.xlim(trajectory_time[0], trajectory_time[-1])
    plt.grid()
    plt.show()

    fig = plt.figure()
    plt.plot(trajectory_time, F1_score_full, label='EKF', color='b', linewidth=2)
    plt.plot(trajectory_time, F1_score_sparse, label='GSP-EKF', color='g', linestyle=':', linewidth=2)
    plt.plot(trajectory_time, F1_score_hybrid, label='GSP-EKF-no propagation', color='orange', linestyle='--',
             linewidth=2)
    plt.plot(trajectory_time, F1_score_oracle, label='ChangeDetection', color='r', linestyle='-.', linewidth=2)
    # plt.yscale('symlog')
    plt.legend()
    plt.ylabel("F score", fontsize=18)
    plt.xlabel("l [time units]", fontsize=18)
    plt.xlim(trajectory_time[0], trajectory_time[-1])
    plt.ylim(-0.005, 1.005)
    plt.grid()
    plt.show()

    fig = plt.figure()
    plt.plot(trajectory_time, EIER_full, label='EKF', color='b', linewidth=2)
    plt.plot(trajectory_time, EIER_sparse, label='GSP-EKF', color='g', linestyle=':', linewidth=2)
    plt.plot(trajectory_time, EIER_hybrid, label='GSP-EKF-no propagation', color='orange', linestyle='--', linewidth=2)
    plt.plot(trajectory_time, EIER_oracle, label='ChangeDetection', color='r', linestyle='-.', linewidth=2)
    # plt.yscale('symlog')
    plt.legend()
    plt.ylabel("EIER [%]", fontsize=18)
    plt.xlabel("l [time units]", fontsize=18)
    plt.xlim(trajectory_time[0], trajectory_time[-1])
    # plt.ylim(-0.005, 1.005)
    plt.grid()
    plt.show()

    import multiprocessing
    from functools import partial

    # # Aid function for parallel computing of the non-linear experiment function
    # pool_obj = multiprocessing.Pool()
    # func = partial(single_monte_carlo_iteration, F, B, C_u, C_w, C_x_missmatch, C_x, stateInit_missmatch, trajectory_time,
    #                stateInit, n, k, thr1, thr2)
    # error = pool_obj.map(func, np.arange(num_iterations))
    # # error = single_monte_carlo_iteration(F, B, C_u, C_w, C_x_missmatch, C_x, stateInit_missmatch, trajectory_time, stateInit, n,
    # #                              k, thr1, thr2)
    with multiprocessing.Pool() as pool:
        func = partial(single_monte_carlo_iteration, F, B, C_u, C_w, C_x_missmatch, C_x, stateInit_missmatch,
                       trajectory_time,
                       stateInit, n, k, thr1, thr2, poly_a)
        error = pool.map(func, np.arange(num_iterations))
    #

    mse_full, F1_score_full, EIER_full = restore_specific_mse(error, 'full', num_time_samples)
    mse_sparse, F1_score_sparse, EIER_sparse = restore_specific_mse(error, 'sparse', num_time_samples)
    mse_hybrid, F1_score_hybrid, EIER_hybrid = restore_specific_mse(error, 'hybrid', num_time_samples)
    mse_oracle, F1_score_oracle, EIER_oracle = restore_specific_mse(error, 'oracle', num_time_samples)

    plt.rcParams.update({'font.size': 14})
    fig = plt.figure()
    plt.plot(trajectory_time, np.log10(mse_full), label='EKF', color='b', linewidth=2)
    plt.plot(trajectory_time, np.log10(mse_sparse), label='GSP-EKF', color='g', linestyle=':', linewidth=2)
    plt.plot(trajectory_time, np.log10(mse_hybrid), label='GSP-EKF-no propagation', color='orange', linestyle='--',
             linewidth=2)
    plt.plot(trajectory_time, np.log10(mse_oracle), label='ChangeDetection', color='r', linestyle='-.', linewidth=2)
    # plt.yscale('symlog')
    plt.legend()
    plt.ylabel("MSE [dB]")
    plt.xlabel("l [time units]")
    plt.xlim(trajectory_time[0], trajectory_time[-1])
    plt.grid()
    plt.show()

    fig = plt.figure()
    plt.plot(trajectory_time, F1_score_full, label='EKF', color='b', linewidth=2)
    plt.plot(trajectory_time, F1_score_sparse, label='GSP-EKF', color='g', linestyle=':', linewidth=2)
    plt.plot(trajectory_time, F1_score_hybrid, label='GSP-EKF-no propagation', color='orange', linestyle='--',
             linewidth=2)
    plt.plot(trajectory_time, F1_score_oracle, label='ChangeDetection', color='r', linestyle='-.', linewidth=2)
    # plt.yscale('symlog')
    plt.legend()
    plt.ylabel("F score", fontsize=18)
    plt.xlabel("l [time units]", fontsize=18)
    plt.xlim(trajectory_time[0], trajectory_time[-1])
    plt.ylim(-0.005, 1.005)
    plt.grid()
    plt.show()

    fig = plt.figure()
    plt.plot(trajectory_time, EIER_full, label='EKF', color='b', linewidth=2)
    plt.plot(trajectory_time, EIER_sparse, label='GSP-EKF', color='g', linestyle=':', linewidth=2)
    plt.plot(trajectory_time, EIER_hybrid, label='GSP-EKF-no propagation', color='orange', linestyle='--', linewidth=2)
    plt.plot(trajectory_time, EIER_oracle, label='ChangeDetection', color='r', linestyle='-.', linewidth=2)
    # plt.yscale('symlog')
    plt.legend()
    plt.ylabel("EIER [%]", fontsize=18)
    plt.xlabel("l [time units]", fontsize=18)
    plt.xlim(trajectory_time[0], trajectory_time[-1])
    # plt.ylim(-0.005, 1.005)
    plt.grid()
    plt.show()
    ccc = 5
