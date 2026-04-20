# -*- coding: utf-8 -*-
"""
Baseline: Graph Recursive Least Squares (GRLS) Filter
=======================================================

Signal model
------------
  f[k] = Σ_{i=1}^{M}  R_i · f[k-i]  +  noise

where each filter  R_i = Σ_j h_{ij} · W^j  is a polynomial in the
graph shift operator W (weighted adjacency matrix).

Algorithm summary
-----------------
  For each new f[k]:

  Step 1 – BCD+RLS to update filter matrices {R_i}
    Outer τ iterations (BCD), inner t iterations (gradient descent).
    For each filter i:
      • Residual:  x_{-i}[k] = f[k] - Σ_{l≠i} R_l · f[k-l]
      • V_k = f[k-i]^T ⊗ I_N            (N × N²)
      • Q_i ← β·Q_i + V_k^T·V_k         (RLS covariance accumulation)
      • q_i ← β·q_i + V_k^T·x_{-i}[k]  (RLS cross-corr accumulation)
      • Γ-sum = Σ_{l≠i} Γ(R_l)^T·Γ(R_l)    commutator penalty
        where  Γ(R) = R^T⊗I − I⊗R
      • Gradient-descent on  r_i = vec(R_i):
          ∇J = (Q_i + λ_3·Γ-sum)·r_i − q_i + λ_r·sign(r_i)
          r_i ← r_i − α·∇J

  Step 2 – Recover W (graph shift operator)
      min_W  ‖R̂_1 − W‖²_F
           + λ_1 · ‖vec(W)‖_1
           + λ_3 · Σ_{i=2}^{M} ‖[W, R̂_i]‖²_F
    Solved via cvxpy (vectorised LASSO + quadratic penalty).

  Step 3 – Recover polynomial coefficients ĥ_i  (optional)
      min_{h_i}  0.5·‖vec(R̂_1) − Υ_i·h_i‖²  +  λ_2·‖h_i‖_1
    where  Υ_i = [vec(I) | vec(W) | … | vec(W^i)].

Integration with evaluation harness
-------------------------------------
    x_est = method(q, y, updated_connections)

  * q  – unused excitation (interface compatibility)
  * y  – observed graph signal f[k]  (N×1)
  * x_est – estimated edge weights from Ŵ  (m×1, ≥ 0)
"""

import numpy as np
import cvxpy as cp
from collections import deque

from util_func import map_B_to_set, extract_x_to_match_B_from_L, build_L


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _commutator_matrix(R: np.ndarray, N: int) -> np.ndarray:
    """
    Build the N²×N² matrix  Γ(R)  such that

        Γ(R) · vec(X)  =  vec([X, R])  =  vec(X·R − R·X)

    Using the Kronecker identities:
        vec(X·R) = (R^T ⊗ I_N) · vec(X)
        vec(R·X) = (I_N ⊗ R)   · vec(X)
    ⟹  Γ(R) = R^T ⊗ I_N  −  I_N ⊗ R
    """
    I = np.eye(N)
    return np.kron(R.T, I) - np.kron(I, R)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class GRLS:
    """
    Graph Recursive Least Squares (GRLS) Filter.

    Parameters
    ----------
    B         : (N, m) ndarray – incidence matrix of the complete graph
    M         : int            – filter order (number of lags)
    beta      : float          – RLS forgetting factor ∈ (0, 1]
    alpha     : float          – step size for inner gradient descent
    lambda_r  : float          – L1 sparsity weight on r_i (filter vec)
    lambda_1  : float          – L1 sparsity weight on vec(W)
    lambda_2  : float          – L1 sparsity weight on h_i (poly coeffs)
    lambda_3  : float          – commutativity penalty weight
    tau_max   : int            – BCD outer iterations per time step
    t_max     : int            – gradient-descent iterations per filter
    """

    def __init__(self, B,
                 M: int = 1,
                 beta: float = 0.99,
                 alpha: float = 0.01,
                 lambda_r: float = 0.01,
                 lambda_1: float = 0.1,
                 lambda_2: float = 0.01,
                 lambda_3: float = 0.1,
                 tau_max: int = 1,
                 t_max: int = 10):

        B = np.asarray(B, dtype=float)
        N = B.shape[0]

        self.B         = B
        self.N         = N
        self.M         = M
        self.beta      = beta
        self.alpha     = alpha
        self.lambda_r  = lambda_r
        self.lambda_1  = lambda_1
        self.lambda_2  = lambda_2
        self.lambda_3  = lambda_3
        self.tau_max   = tau_max
        self.t_max     = t_max

        self.edge_pairs = map_B_to_set(B)

        N2 = N * N

        # Filter matrices R[i] ∈ ℝ^{N×N},  lag = i+1  (i = 0 … M-1)
        self.R = [np.zeros((N, N)) for _ in range(M)]

        # RLS sufficient statistics — one (Q_i, q_i) pair per filter
        self.Q     = [np.zeros((N2, N2)) for _ in range(M)]   # (N²×N²)
        self.q_vec = [np.zeros((N2, 1))  for _ in range(M)]   # (N²×1)

        # Circular signal buffer: buffer[0]=f[k], buffer[l]=f[k-l]
        self.buffer = deque([np.zeros((N, 1))] * (M + 1), maxlen=M + 1)

        # Current graph-weight-matrix estimate
        self.W_hat = np.zeros((N, N))

    # ------------------------------------------------------------------
    def __call__(self, q_exc, y, *args, **kwargs) -> np.ndarray:
        """
        Process one time step.

        Parameters
        ----------
        q_exc : (N,1) – excitation (unused, kept for interface compatibility)
        y     : (N,1) – observed graph signal  f[k]

        Returns
        -------
        x_est : (m,1) – estimated edge weights (≥ 0)
        """
        # Push new signal; appendleft shifts so buffer[0]=f[k], buffer[l]=f[k-l]
        self.buffer.appendleft(y.reshape(-1, 1))

        # Step 1 – update filter matrices via BCD + gradient descent
        self._update_filters()

        # Step 2 – recover graph weight matrix W
        self._recover_W()

        return np.maximum(self._extract_weights(), 0.0)

    # ------------------------------------------------------------------
    def _update_filters(self):
        """BCD outer iterations; inner gradient descent on each r_i = vec(R_i)."""
        N   = self.N
        M   = self.M
        buf = list(self.buffer)    # buf[0]=f[k], buf[l]=f[k-l]
        f_k = buf[0]

        for _tau in range(self.tau_max):
            for i in range(M):

                lag = i + 1
                if lag >= len(buf):
                    continue
                f_lag = buf[lag]      # f[k-(i+1)]

                # ── Residual: remove contribution of all other filters ──
                x_minus_i = f_k.copy()
                for l in range(M):
                    if l == i:
                        continue
                    ll = l + 1
                    if ll >= len(buf):
                        continue
                    x_minus_i -= self.R[l] @ buf[ll]

                # ── V_k = f[k-i]^T ⊗ I_N  →  shape (N, N²) ───────────
                V_k = np.kron(f_lag.T, np.eye(N))

                # ── RLS sufficient-statistic updates ──────────────────
                self.Q[i]     = self.beta * self.Q[i]     + V_k.T @ V_k
                self.q_vec[i] = self.beta * self.q_vec[i] + V_k.T @ x_minus_i

                # ── Commutator penalty:  Σ_{l≠i} Γ_l^T Γ_l ──────────
                Gamma_sum = np.zeros((N * N, N * N))
                for l in range(M):
                    if l != i:
                        Gamma_l    = _commutator_matrix(self.R[l], N)
                        Gamma_sum += Gamma_l.T @ Gamma_l

                # A = Q_i + λ_3 · Γ-sum  (curvature for gradient step)
                A = self.Q[i] + self.lambda_3 * Gamma_sum

                # ── Inner gradient descent on r_i = vec(R_i) ──────────
                # vec uses column-major ('F') to match Kronecker convention
                r_i = self.R[i].ravel(order='F').reshape(-1, 1)

                for _t in range(self.t_max):
                    grad = A @ r_i - self.q_vec[i] + self.lambda_r * np.sign(r_i)
                    r_i  = r_i - self.alpha * grad

                self.R[i] = r_i.reshape(N, N, order='F')

    # ------------------------------------------------------------------
    def _recover_W(self):
        """
        Recover W by solving (vectorised form, w = vec(W)):

            min_w  ‖r̂_1 − w‖²
                 + λ_3 · ‖Γ_stack · w‖²
                 + λ_1 · ‖w‖_1

        where  Γ_stack = [Γ(R̂_2); Γ(R̂_3); …; Γ(R̂_M)]  (stacked N²-row blocks).
        """
        N   = self.N
        N2  = N * N
        r1  = self.R[0].ravel(order='F')           # vec(R̂_1)

        # Build stacked commutator matrix for i = 2..M (indices 1..M-1)
        if self.M > 1 and self.lambda_3 > 0.0:
            Gamma_stack = np.vstack([
                _commutator_matrix(self.R[i], N) for i in range(1, self.M)
            ])                                     # ((M-1)N², N²)
        else:
            Gamma_stack = np.zeros((1, N2))

        w = cp.Variable(N2)
        objective = cp.Minimize(
            cp.sum_squares(r1 - w)
            + self.lambda_3 * cp.sum_squares(Gamma_stack @ w)
            + self.lambda_1 * cp.norm1(w)
        )
        cp.Problem(objective).solve(verbose=False)

        if w.value is not None:
            W = w.value.reshape(N, N, order='F')
            W = (W + W.T) / 2.0           # symmetrize (undirected graph)
            np.fill_diagonal(W, 0.0)      # no self-loops
            self.W_hat = W

    # ------------------------------------------------------------------
    def _recover_h(self):
        """
        Recover polynomial filter coefficients via LASSO (optional):

            min_{h_i}  0.5·‖vec(R̂_1) − Υ_i·h_i‖²  +  λ_2·‖h_i‖_1

        where  Υ_i = [vec(I) | vec(W) | vec(W²) | … | vec(W^i)]  ∈ ℝ^{N²×(i+2)}.
        """
        N  = self.N
        r1 = self.R[0].ravel(order='F')

        # Precompute W^0 = I, W^1, …, W^M
        W_pow = [np.eye(N)]
        for _ in range(self.M):
            W_pow.append(W_pow[-1] @ self.W_hat)

        for i in range(self.M):
            Upsilon_i = np.column_stack([
                W_pow[j].ravel(order='F') for j in range(i + 2)
            ])                                     # (N², i+2)

            h_i = cp.Variable(i + 2)
            objective = cp.Minimize(
                0.5 * cp.sum_squares(r1 - Upsilon_i @ h_i)
                + self.lambda_2 * cp.norm1(h_i)
            )
            cp.Problem(objective).solve(verbose=False)

    # ------------------------------------------------------------------
    def _extract_weights(self) -> np.ndarray:
        """Read symmetrised off-diagonal W_hat entries for each edge in B."""
        m = len(self.edge_pairs)
        x = np.zeros((m, 1))
        for k, (i, j) in enumerate(self.edge_pairs):
            x[k, 0] = (self.W_hat[i, j] + self.W_hat[j, i]) / 2.0
        return x


# ---------------------------------------------------------------------------
# Laplacian projection and eigendecomposition-based L recovery
# ---------------------------------------------------------------------------
def project_to_laplacian(L_raw: np.ndarray) -> np.ndarray:
    """
    Project an arbitrary symmetric matrix onto the cone of valid
    combinatorial Laplacians:

      1. Symmetrise:             L = (L_raw + L_raw^T) / 2
      2. Clip off-diagonal:      L_ij = min(L_ij, 0)  for i ≠ j
      3. Fix diagonal (row-sum zero):  L_ii = -Σ_{j≠i} L_ij

    Returns
    -------
    L : (N, N) ndarray – valid Laplacian (PSD, zero row-sums, non-positive off-diag)
    """
    L = (L_raw + L_raw.T) / 2.0

    # Force off-diagonal entries to be non-positive
    mask = ~np.eye(L.shape[0], dtype=bool)
    L[mask] = np.minimum(L[mask], 0.0)

    # Fix diagonal so every row sums to zero
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(L, -L.sum(axis=1))

    return L


def extract_L_from_R(R: np.ndarray, poly_c) -> np.ndarray:
    """
    Recover the Laplacian L from the estimated filter matrix
    R = Σ_j h_j · L^j,  using the fact that R and L share eigenvectors.

    Algorithm
    ---------
    1. Eigendecompose R  →  R = U · Λ_R · U^T
    2. For each eigenvalue λ_R_i, solve the scalar polynomial
           h_0 + h_1·λ_L + … + h_M·λ_L^M  =  λ_R_i
       and pick the real, non-negative root (valid Laplacian eigenvalue).
    3. Reconstruct  L_raw = U · diag(λ_L) · U^T
    4. Project onto the Laplacian cone via project_to_laplacian().

    Parameters
    ----------
    R      : (N, N) ndarray – estimated filter matrix
    poly_c : array-like     – polynomial coefficients [h_0, h_1, …, h_M]
                              (same convention as the EKF modules)

    Returns
    -------
    L : (N, N) ndarray – valid Laplacian estimate
    """
    N = R.shape[0]
    h = np.asarray(poly_c, dtype=float)     # [h_0, h_1, …, h_M]

    # ── Degree-1 shortcut: R = h_0·I + h_1·L  →  L = (R − h_0·I) / h_1 ──
    if len(h) == 2:
        h0, h1 = float(h[0]), float(h[1])
        if abs(h1) < 1e-12:
            return np.zeros((N, N))
        L_raw = (R - h0 * np.eye(N)) / h1
        return project_to_laplacian(L_raw)

    # ── Step 1: eigendecompose (exploit symmetry for stability) ──────
    R_sym = (R + R.T) / 2.0
    eigvals_R, U = np.linalg.eigh(R_sym)    # R = U Λ_R U^T, real eigvals

    # ── Step 2: invert polynomial per eigenvalue ──────────────────────
    # numpy.roots convention: highest-degree first → reverse h
    h_desc = h[::-1].copy()                 # [h_M, h_{M-1}, …, h_1, h_0]

    eigvals_L = np.zeros(N)
    for i, lam_R in enumerate(eigvals_R):
        # Polynomial equation: h(λ_L) - λ_R = 0
        coeffs = h_desc.copy()
        coeffs[-1] -= lam_R                 # subtract λ_R from constant term

        if len(coeffs) < 2:
            # Degree-0 case: degenerate, keep 0
            eigvals_L[i] = 0.0
            continue

        roots = np.roots(coeffs)

        # Keep only roots with negligible imaginary part
        real_mask  = np.abs(roots.imag) < 1e-6
        real_roots = roots[real_mask].real

        # Among real roots, prefer non-negative ones (valid Laplacian eigenvalues)
        nonneg = real_roots[real_roots >= -1e-6]

        if nonneg.size > 0:
            eigvals_L[i] = nonneg.min()     # smallest non-negative root
        elif real_roots.size > 0:
            eigvals_L[i] = max(0.0, real_roots[np.argmin(np.abs(real_roots))])
        else:
            # All roots are complex: fall back to 0
            eigvals_L[i] = 0.0

    # ── Step 3: reconstruct L ─────────────────────────────────────────
    L_raw = (U * eigvals_L) @ U.T           # U @ diag(eigvals_L) @ U.T

    # ── Step 4: project onto valid Laplacian cone ─────────────────────
    return project_to_laplacian(L_raw)


# ---------------------------------------------------------------------------
# Simplified GRLS — single polynomial filter, no W-recovery step
# ---------------------------------------------------------------------------
class GRLSSimple:
    """
    Simplified GRLS Filter (single filter, no BCD, no W-recovery).

    Models  f[k] ≈ R · f[k-M]  where  R = H(L) = Σ_j h_j · L^j
    is a polynomial in the Laplacian L with known coefficients h = poly_c.

    Because there is only one filter term, BCD reduces to a single
    gradient-descent update and no commutator penalty is needed.

    The harness passes:
        q  →  f[k-M]   (lagged signal, M steps ago)
        y  →  f[k]     (current observed signal)

    Laplacian recovery pipeline (per time step):
        R  →  extract_L_from_R(R, poly_c)   (eigendecompose + poly inversion)
           →  project_to_laplacian(L_raw)
           →  extract_x_to_match_B_from_L(L, B)

    Parameters
    ----------
    B        : (N, m) ndarray – incidence matrix of the complete graph
    poly_c   : array-like     – polynomial coefficients [h_0, h_1, …, h_M]
                                (same convention as the EKF modules)
    beta     : float          – RLS forgetting factor ∈ (0, 1]
    alpha    : float          – gradient-descent step size
    lambda_r : float          – L1 sparsity weight on r = vec(R)
    t_max    : int            – gradient-descent iterations per time step
    """

    def __init__(self, B, poly_c, cov_init, state_init_weight,
                 beta, alpha, lambda_r, t_max):

        B = np.asarray(B, dtype=float)
        N = B.shape[0]
        N2 = N * N

        self.B        = B
        self.N        = N
        self.poly_c   = np.asarray(poly_c, dtype=float)
        self.alpha    = alpha
        self.beta     = beta
        self.lambda_r = lambda_r
        self.t_max    = t_max

        self.edge_pairs = map_B_to_set(B)

        e_comp = state_init_weight * np.ones(B.shape[1]).reshape([B.shape[1], 1])
        state_init =  build_L(B, e_comp).reshape(-1, 1)
        # Filter vector  r = vec(R),  R ∈ ℝ^{N×N}
        self.r = state_init

        # RLS sufficient statistics
        self.Q     = cov_init * np.eye(N2)  # (N²×N²) covariance
        self.q_vec = state_init    # (N²×1)  cross-correlation

    # ------------------------------------------------------------------
    def __call__(self, q, y, *args, **kwargs) -> np.ndarray:
        """
        Process one time step.

        Parameters
        ----------
        q : (N,1) – lagged signal f[k-M]
        y : (N,1) – current signal f[k]

        Returns
        -------
        x_est : (m,1) – estimated edge weights (≥ 0)
        """
        f_lag = q.reshape(-1, 1)    # f[k-M]
        f_k   = y.reshape(-1, 1)    # f[k]

        # ── V_k = f[k-M]^T ⊗ I_N  →  shape (N, N²) ──────────────────
        V_k = np.kron(f_lag.T, np.eye(self.N))

        # ── RLS sufficient-statistic updates ──────────────────────────
        self.Q     = self.beta * self.Q     + V_k.T @ V_k
        self.q_vec = self.beta * self.q_vec + V_k.T @ f_k

        # ── Gradient descent on r = vec(R) ────────────────────────────
        #   ∇J(r) = Q_K · r  −  q_K  +  λ_r · sign(r)
        for _t in range(self.t_max):
            grad   = self.Q @ self.r - self.q_vec + self.lambda_r * np.sign(self.r)
            self.r = self.r - self.alpha * grad

        # ── Recover R, then L, then edge weights ──────────────────────
        R = self.r.reshape(self.N, self.N, order='F')

        L = extract_L_from_R(R, self.poly_c)   # eigendecomp + poly inversion + projection
        x_est = extract_x_to_match_B_from_L(L, self.B)

        return np.maximum(x_est, 0.0)


# ---------------------------------------------------------------------------
# Quick standalone example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import pickle
    import logging

    from constants import cfg_linear
    from util_func import get_trajectory, one_method_evaluation, plot_metric

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    data_folder_name = "Results"
    os.makedirs(data_folder_name, exist_ok=True)

    cfg = cfg_linear.copy()

    # ── Single (non-parallel) Monte-Carlo run ─────────────────────────────
    runs_grls = []
    for _ in range(cfg["num_iterations"]):
        pos, q_meas, y_meas, conn, stateInit = get_trajectory(
            cfg["trajectory_time"], cfg["F"], cfg["B"],
            cfg["C_w_sqrt"], cfg["C_u_sqrt"], cfg["n"], cfg["k"],
            cfg["poly_coefficients"], cfg["new_edge_weight"],
            cfg["num_edges_stateinit"], cfg["delta_n"],
        )
        method = GRLSSimple(
            cfg["B"], cfg["poly_coefficients"],
            cfg["cov_init"], cfg["stateInit_weight_grls"],
            beta=cfg["beta"],
            alpha=cfg["alpha"],
            lambda_r=cfg["lambda_r"],
            t_max=cfg["t_max"],
        )
        mse, nmse, f1, eier, neier, times = one_method_evaluation(
            method, q_meas, y_meas, pos, conn
        )
        runs_grls.append(dict(grls=dict(mse=mse, normalized_mse=nmse, f1=f1,
                                        eier=eier, normalized_eier=neier,
                                        times=times)))

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(data_folder_name, "runs_grls_linear.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(runs_grls, f)
    logging.info(f"Results saved to {out_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    grls_labels  = {"grls": "GRLS"}
    grls_methods = ["grls"]

    plot_metric(cfg["trajectory_time"], runs_grls, "mse",
                labels=grls_labels, methods_to_plot=grls_methods,
                log_format=True, to_save=True,
                folder_name=data_folder_name, suffix="grls_linear")
    plot_metric(cfg["trajectory_time"], runs_grls, "f1",
                labels=grls_labels, methods_to_plot=grls_methods,
                to_save=True, folder_name=data_folder_name, suffix="grls_linear")
    plot_metric(cfg["trajectory_time"], runs_grls, "eier",
                labels=grls_labels, methods_to_plot=grls_methods,
                to_save=True, folder_name=data_folder_name, suffix="grls_linear")
    plot_metric(cfg["trajectory_time"], runs_grls, "times",
                labels=grls_labels, methods_to_plot=grls_methods,
                log_format=True, to_save=True,
                folder_name=data_folder_name, suffix="grls_linear")