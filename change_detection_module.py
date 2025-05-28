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
    delta_s is a CVXPY variable.
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
    for a single input column q.

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
        Constructs the Laplacian L = B * diag(x) * B.T in a CVXPY-compatible way.

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
        gamma = cp.diag(x)  # Make a diagonal matrix from vector x
        return B @ gamma @ B.T  # CVXPY uses @ for matrix multiplication

    def onlineDirectedGraphEstimtion(self, Delta_Y_t, Xt, L):
        # Delta_Y_t = Yt - compute_poly(L, Xt, self.a)
        beta1 = self.lambda_1
        beta2 = self.lambda_2
        factor = 2
        max_retry = 10
        for attempt in range(max_retry):
            # Optimization variable
            Delta_S_t = cp.Variable((self.B.shape[1], 1), nonneg=True)
            Delta_L = self.build_L_cvx(self.B, Delta_S_t)
            Delta_Y_t_prime = compute_delta_Y_poly_cvx(L, Delta_L, Xt, self.a)
            # Objective function
            objective = cp.Minimize(
                cp.norm(Delta_Y_t - Delta_Y_t_prime, 'fro') ** 2 +
                beta1 * cp.norm1(Delta_L) +
                beta2 * cp.normNuc(Delta_L)
            )

            # Problem definition and solving
            prob = cp.Problem(objective)
            prob.solve()
            # ----- 3.  success? -----------------------------------------------
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and Delta_S_t.value is not None:
                return Delta_S_t.value.astype(float)

            # ----- 4.  fail → adapt & retry -----------------------------------
            logging.warning(f"delta_s solve failed (status={prob.status}); "
                            f"increasing beta to {beta1 * factor}")
            beta1 *= factor
            beta2 *= factor

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

class FastChangeDetectionMethod(ChangeDetectionMethod):
        def __init__(self, B, m, StateInit, poly_c, lambda_1=5, lambda_2=0):
            super(FastChangeDetectionMethod, self).__init__(B, m, StateInit, poly_c, lambda_1=lambda_1, lambda_2=lambda_2)
            self.N = B.shape[0]
            self.M1 = B.shape[1]
            self.Y_param = cp.Parameter((self.N, 1))
            self.Delta_S = cp.Variable((self.M1, 1), nonneg=True)  # optimisation variable
            self.P = cp.Parameter((self.N, self.M1))
            resid = self.Y_param - self.P @ self.Delta_S
            if self.lambda_2 > 0: #cp.sum_squares(resid) \
                obj = cp.norm(resid, 'fro') ** 2 \
                  + self.lambda_1 * cp.norm1(self.Delta_S)  \
                  + self.lambda_2 * cp.normNuc(self.Delta_L)
            else:#cp.sum_squares(resid) \
                obj = cp.norm(resid, 'fro') ** 2 \
                  + self.lambda_1 * cp.norm1(self.B @ cp.diag(self.Delta_S) @ self.B.T)

            self.prob = cp.Problem(cp.Minimize(obj))

        def onlineDirectedGraphEstimtion(self, Delta_Y_t, Xt, L):
            Delta_S_t = cp.Variable((self.B.shape[1], 1))
            # Delta_L = self.build_L_cvx(self.B, Delta_S_t)
            # Delta_Y_t_prime = compute_delta_Y_poly_cvx(L, Delta_L, Xt, self.a)
            # # Objective function
            # objective = cp.Minimize(
            #     cp.norm(Delta_Y_t - Delta_Y_t_prime, 'fro') ** 2 +
            #     self.lambda_1 * cp.norm1(Delta_L) +
            #     self.lambda_2 * cp.normNuc(Delta_L)
            # )
            #
            # # Problem definition and solving
            # problem = cp.Problem(objective)
            # problem.solve(solver="CLARABEL", verbose=True)
            #######

            # if (self.Y_param.value is not None) and (self.Y_param.value.shape != Delta_Y_t.shape):
            #     self.Y_param = cp.Parameter((Delta_Y_t.shape[0] * Delta_Y_t.shape[1], 1))
            #     self.P = cp.Parameter((Delta_Y_t.shape[0] * Delta_Y_t.shape[1], self.M1))
            #
            # self.Y_param.value = Delta_Y_t.reshape(-1, 1, order="F")
            # once per time-step (or whenever L changes)
            P_blocks = []
            for j in range(Xt.shape[1]):
                Pj = precompute_P(L, Xt[:, [j]], self.a, self.B).toarray()
                P_blocks.append(Pj)  # N×E                     # N×1
            # self.P.value = np.vstack(P_blocks)
            # self.prob.solve(verbose=True)  #warm_start=True , solver=cp.OSQP)  # or MOSEK/SCS
            objective = cp.Minimize(
                cp.norm(Delta_Y_t.reshape(-1, 1, order="F") - np.vstack(P_blocks) @ Delta_S_t , 'fro') ** 2 +
                self.lambda_1 * cp.norm1(self.B @ cp.diag(Delta_S_t) @ self.B.T)
                 + self.lambda_2 * cp.normNuc(self.B @ cp.diag(Delta_S_t) @ self.B.T)
            )

            # Problem definition and solving
            problem = cp.Problem(objective)
            problem.solve(solver="CLARABEL", verbose=True)
            # zzzz= np.linalg.norm(self.Delta_S.value - Delta_S_t.value)
            return Delta_S_t.value#self.Delta_S.value



if __name__ == "__main__":
    # --- imports the file already has ---------------------------
    import numpy as np
    import scipy.sparse as sp
    from numpy.random import default_rng

    rng = default_rng(0)

    # ---------- test parameters ---------------------------------
    N = 8                   # number of nodes
    M = num_possible_edges(N)
    E = 12                  # number of edges  (must be ≥ N-1 for connectivity)
    r = 3                   # window length  (number of q / Y columns)
    Kdeg = 3                # polynomial degree
    trials = 5              # repeat to be sure
    tol = 1e-9              # numeric tolerance for equality

    # ---------- run several random checks -----------------------
    for t in range(trials):
        G = nx.complete_graph(N, create_using=None)
        B = nx.incidence_matrix(G, oriented=True).todense()
        new_edge_weight = 1
        idx_list = chose_indices_without_repeating(M, E)
        s0 = np.zeros(M).reshape([M, 1])
        s0[idx_list] = new_edge_weight
        # B = random_incidence(N, E, rng)
        # s0 = rng.uniform(0.5, 1.5, size=(E, 1))          # nominal weights
        L0 = B @ sp.diags(s0.ravel()) @ B.T  # Laplacian

        # polynomial coefficients  a0 … aK  (a0 unused in derivative)
        a = rng.standard_normal(Kdeg + 1)

        # random window of inputs  X ∈ ℝ^{N×r}
        X = rng.standard_normal((N, r))

        # random perturbation  Δs
        Δs = np.zeros((M,1))
        Δs[idx_list] = rng.standard_normal((E, 1))

        # ---- slow path: build ΔL, call compute_delta_Y_poly_cvx ---
        ΔL = B @ sp.diags(Δs.ravel()) @ B.T
        # slow_cols = []
        # for j in range(r):
        slow_cols = compute_delta_Y_poly_cvx(L0, ΔL, X, a)
            # slow_cols.append(col)
        ΔYprime_slow = ΔYprime_slow.reshape(-1, 1, order="F")
        # ---- fast path: use P_j blocks ----------------------------
        P_blocks = []
        for j in range(r):
            Pj = precompute_P(L0, X[:, [j]], a, B).toarray()
            P_blocks.append(Pj)# N×E                     # N×1
        Big_P = np.vstack(P_blocks)
        ΔYprime_fast = Big_P @ Δs# N×r

        # ---- compare ---------------------------------------------
        err = np.linalg.norm(ΔYprime_fast - ΔYprime_slow, ord='fro')
        print(f"trial {t}:  ‖fast − slow‖_F = {err:.3e}")
        assert err < tol, "Mismatch!  Something is wrong."

    print("\nAll trials matched within the numerical tolerance.")



    a=2