# -*- coding: utf-8 -*-
import torch.nn as nn
import networkx as nx
import numpy as np
import matplotlib
import time,logging

from torch.backends.mkl import verbose

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
import cvxpy as cp
import scipy.sparse as sp
from util_func import *
import cvxpy as cp


def compute_delta_Y_poly_cvx(L, delta_s, q, a):
    """
    CVXPY-compatible version of compute_delta_Y_poly.
    delta_s is poly_c CVXPY variable.
    """
    N = L.shape[0]
    max_power = len(a) - 1

    # Precompute powers of L
    L_powers = [np.eye(N)]
    for _ in range(max_power):
        L_powers.append(L_powers[-1] @ L)

    H_col = 0  # CVXPY expression

    for i_idx in range(1, len(a)):
        ai = a[i_idx]
        if ai == 0:
            continue

        inner_sum = 0  # CVXPY expression

        for k in range(i_idx):
            Lk = L_powers[k]
            L_i_k_1 = L_powers[i_idx - k - 1]

            temp = L_i_k_1 @ q
            temp = delta_s @ temp  # CVXPY matrix product
            temp = Lk @ temp

            inner_sum += temp

        H_col += ai * inner_sum

    return H_col


def precompute_P(L, q, a, B):
    """
    Build P such that  ΔY' = P · Δs
    for poly_c single input column q.

    Parameters
    ----------
    L : (N,N) ndarray or sparse
        Nominal Laplacian.
    q : (N,1) ndarray
        Current input vector.
    a : 1-D array_like
        Polynomial coefficients  [a0, a1, …, aK].
    B : (N,E) csr_matrix
        Signed incidence matrix.

    Returns
    -------
    P : (N,E) csr_matrix
    """
    B = sp.csr_matrix(B)
    L = sp.csr_matrix(L)
    N, E   = B.shape
    K      = len(a) - 1
    Lpow   = [sp.identity(N, format="csr")]
    for _ in range(K):
        Lpow.append(Lpow[-1] @ L)        # sparse powers

    # accumulate columns
    data, rows, cols = [], [], []
    for e in range(E):
        b_e = B[:, e]                    # (N,1) sparse column
        for k in range(1, K+1):
            ak = a[k]
            if ak == 0:
                continue
            inner = sp.csr_matrix((N, 1))    # zero column
            for i in range(k):
                term = Lpow[i] @ (b_e.multiply(1))      # S^i b_e
                term = term.multiply(b_e.T @ Lpow[k-1-i] @ q)  # scalar * vector
                inner += term
            # add column e contribution
            rows.extend(inner.nonzero()[0])
            cols.extend([e]*inner.nnz)
            data.extend((ak * inner.data).tolist())

    P = sp.csr_matrix((data, (rows, cols)), shape=(N, E))
    return P


class ChangeDetectionMethod(nn.Module):
    def __init__(self, B, m, StateInit, poly_c, lambda_1=10, lambda_2=0):
        super(ChangeDetectionMethod, self).__init__()

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.prev_y = np.empty((min(np.shape(B)), 0))
        self.prev_q = np.empty((min(np.shape(B)), 0))
        self.B = B
        self.M = m
        self.s = np.array(StateInit, dtype=np.float64)
        self.a = poly_c

    @staticmethod
    def build_L_cvx(B, x):
        """
        Constructs the Laplacian L = B * diag(x) * B.T in poly_c CVXPY-compatible way.

        Parameters
        ----------
        B : np.ndarray
            Incidence matrix, shape (N, E)
        x : cp.Variable or cp.Parameter
            Edge weights, shape (E,)

        Returns
        -------
        L : cp.Expression
            Laplacian matrix expression, shape (N, N)
        """
        gamma = cp.diag(x)  # Make poly_c diagonal matrix from vector x
        return B @ gamma @ B.T  # CVXPY uses @ for matrix multiplication

    def onlineDirectedGraphEstimtion(self, Delta_Y_t, Xt, L):
        # Delta_Y_t = Yt - compute_poly(L, Xt, self.poly_c)
        beta1 = self.lambda_1
        beta2 = self.lambda_2
        # --- 1. data-dependent scaling factors --------------------------------------
        sigma_Y = np.linalg.norm(Delta_Y_t, 'fro') + 1e-12  # avoid 0-division
        sigma_L = 1#np.linalg.norm(L, 'fro') + 1e-12  # rough size of ΔL

        # renormalise the weights
        beta1_scaled = beta1 * sigma_L / (sigma_Y ** 2)
        beta2_scaled = beta2 * sigma_L / (sigma_Y ** 2)

        # --- 2. CVXPY model ----------------------------------------------------------
        Delta_S_t = cp.Variable((self.B.shape[1], 1))

        # scaled decision variable   ΔL̃ = ΔL / σ_L  → norm1/nuc are O(1)
        Delta_L_tilde = self.build_L_cvx(self.B, Delta_S_t) / sigma_L

        # forward model already divided by σ_Y
        Delta_Y_t_prime = compute_delta_Y_poly_cvx(
            L,  # original Laplacian
            Delta_L_tilde * sigma_L,  # undo scaling inside the model call
            Xt, self.a
        ) / sigma_Y

        objective = cp.Minimize(
            cp.sum_squares((Delta_Y_t / sigma_Y) - Delta_Y_t_prime)  # ‖·‖_F² / σ_Y²
            + beta1_scaled * cp.norm1(Delta_L_tilde)  # β₁‖ΔL̃‖₁
            + beta2_scaled * cp.normNuc(Delta_L_tilde)  # β₂‖ΔL̃‖_*
        )

        prob = cp.Problem(objective)

        # # factor = 0.8
        # # max_retry = 5
        # # for attempt in range(max_retry):
        # # Optimization variable
        # Delta_S_t = cp.Variable((self.B.shape[1], 1))
        # Delta_L = self.build_L_cvx(self.B, Delta_S_t)
        # Delta_Y_t_prime = compute_delta_Y_poly_cvx(L, Delta_L, Xt, self.a)
        # # Objective function
        # objective = cp.Minimize(
        #     cp.norm(Delta_Y_t - Delta_Y_t_prime, 'fro') ** 2 +
        #     beta1 * cp.norm1(Delta_L) +
        #     beta2 * cp.normNuc(Delta_L)
        # )
        #
        # # Problem definition and solving
        # prob = cp.Problem(objective)
        solvers = ["MOSEK", "CLARABEL", "SCS"]  # order matters
        for s in solvers:
            try:
                logging.info(f"Trying solver: {s}")
                prob.solve(solver=s, verbose=False)
                if prob.status == cp.OPTIMAL and Delta_S_t.value is not None:
                    return Delta_S_t.value.astype(float)
            except cp.SolverError as e:
                logging.warning(f"{s} failed ({e}).")

            # # ----- 4.  fail → adapt & retry -----------------------------------
            # beta2 *= factor
            # beta1 *= (factor**-1)
            # logging.warning(f"delta_s solve failed (status={prob.status}); "
            #                 f"increasing beta1 to {beta1} beta2 to {beta2}")

        logging.error("delta_s could not be found – skipping update this step")
        return np.zeros_like(self.s)


    def forward(self, q, y, *args, **kwargs):
        L = build_L(self.B, self.s)
        if self.prev_y.shape[1] > 0:
            self.prev_y = self.prev_y[:, -self.M:]
        self.prev_y = np.hstack((self.prev_y, y))

        if self.prev_q.shape[1] > 0:
            self.prev_q = self.prev_q[:, -self.M:]
        self.prev_q = np.hstack((self.prev_q, q))
        Delta_Y_t = self.prev_y - compute_poly(L, self.prev_q, self.a)

        delta_s = self.onlineDirectedGraphEstimtion(Delta_Y_t, self.prev_q, L)
        self.s += delta_s
        self.s = np.maximum(self.s, 0)

        return self.s
