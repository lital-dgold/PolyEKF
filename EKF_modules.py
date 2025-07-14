# -*- coding: utf-8 -*-
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc. depending on your setup
import cvxpy as cp
from util_func import *
np.random.seed(0)


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
        H = compute_Jacobian_poly_dp(L, self.B, q, self.a, self.edges)
        S, K, Sigma = kalman_gain_imp(H, self.Sigma, self.W)
        self.s = (self.s + np.dot(K, (y - compute_poly(L, q, self.a))))
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
        x = cp.Variable((self.B.shape[1], 1), nonneg=True)

        from scipy.linalg import sqrtm
        W_sqrt_inv = np.linalg.inv(sqrtm(self.W))
        Sigma_sqrt_inv = np.linalg.inv(sqrtm(self.Sigma))
        beta = self.lambda_1

        objective = cp.Minimize(
            cp.sum_squares(W_sqrt_inv @ (Delta_Y_t - H @ (x - self.s))) +
            cp.sum_squares(Sigma_sqrt_inv @ (x - self.s)) +
            beta * cp.norm1(x)
        )

        problem = cp.Problem(objective)
        problem.solve(solver="MOSEK")


        # Solution
        self.s = x.value
        S, K, self.Sigma = kalman_gain_imp(H, self.Sigma, self.W)


    def forward(self, q, y, *args, **kwargs):
        # First perdict next state without observation
        super(sparseKalmanFilterISTA, self).forward(q, y)
        # self.s = truncate_matrix(self.s, self.thr)
        return self.s


class oraclKalmanFilt_paper(FastExtendedKalmanFilter):
    def __init__(self, F, B, C_u, C_w, C_x, StateInit, poly_c, new_connections_uncertainty_weight):
        super(oraclKalmanFilt_paper, self).__init__(F, B, C_u, C_w, C_x, StateInit, poly_c)
        self.new_connections_uncertainty_weight = new_connections_uncertainty_weight

    def update(self, q, y, connections):
        connections_int = np.asarray(connections, dtype=int)
        old_connections = np.asarray(np.where(self.s > 0)[0], dtype=int)
        new_connections = np.setdiff1d(connections_int, old_connections)
        self.s[new_connections] = self.new_connections_uncertainty_weight
        L = build_L(self.B, self.s)
        H = compute_Jacobian_poly_dp(L, self.B, q, self.a, self.edges)
        H = H[:, connections]
        uncertainty_sigma = self.Sigma
        uncertainty_sigma_small = uncertainty_sigma[np.ix_(connections, connections)]
        S, K, Sigma_small = kalman_gain_imp(H, uncertainty_sigma_small, self.W)
        self.s[connections] = (self.s[connections] + np.dot(K, (y - compute_poly(L, q, self.a))))
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

