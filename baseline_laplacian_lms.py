# -*- coding: utf-8 -*-
"""
Baseline: Laplacian LMS (unconstrained)
========================================
Reference algorithm
-------------------
  center(s_i):
      s̄_i = s_i − (1/N) · 11ᵀ · s_i

  getLaplacian(W̄_i, T):
      W_i = W̄_i + (1/N) · 11ᵀ
      L̂   = −(1/T) · logm(W_i)      # matrix logarithm

  Algorithm 1 — Laplacian LMS (unconstrained)
  input:  streaming centered signals {s̄_i},  step-size μ
  output: sequence of estimates {W̄_i}

  W̄_0 = zeros(N, N)

  for i = 1, 2, 3, … :
      s̄_i   = center(s_i)
      s̄_i₋₁ = center(s_{i-1})
      e_i   = s̄_i − W̄_{i-1} · s̄_{i-1}
      W̄_i   = W̄_{i-1} + μ · e_i · s̄_{i-1}ᵀ
      L̂_i   = getLaplacian(W̄_i, T)

Signal model assumed by LMS
----------------------------
  s_i = W · s_{i-1} + noise,  where  W = exp(−L · T)

Integration with evaluation harness
-------------------------------------
The ``__call__`` method matches the interface expected by
``one_method_evaluation`` in util_func.py:

    x_est = method(q, y, updated_connections)

  * q                 – excitation signal (N×1), not used by LMS
  * y                 – observed node signal s_i (N×1)
  * updated_connections – active edge indices, not used by LMS
  * x_est             – estimated edge weights (m×1), extracted from L̂
"""

import numpy as np
from scipy.linalg import logm

from util_func import map_B_to_set


# ---------------------------------------------------------------------------
# Helper – extract edge weights from a (possibly slightly asymmetric) L
# ---------------------------------------------------------------------------
def _extract_edge_weights(L_hat, B, edge_pairs):
    """
    For every edge (i, j) encoded in B, read −(L[i,j] + L[j,i])/2 as the
    estimated weight (symmetric average guards against small numerical drift
    introduced by the matrix logarithm).

    Parameters
    ----------
    L_hat : (N, N) ndarray  – estimated Laplacian (real part taken)
    B     : (N, m) ndarray  – incidence matrix
    edge_pairs : list of (i, j) tuples from map_B_to_set(B)

    Returns
    -------
    x_est : (m, 1) ndarray  – non-negative edge weights
    """
    m = B.shape[1]
    x_est = np.zeros((m, 1))
    for k, (i, j) in enumerate(edge_pairs):
        w_ij = -(L_hat[i, j] + L_hat[j, i]) / 2.0
        x_est[k, 0] = w_ij
    return x_est


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class LaplacianLMS:
    """
    Online Laplacian estimation via the unconstrained LMS rule.

    Parameters
    ----------
    B   : (N, m) ndarray  – incidence matrix of the complete graph
    mu  : float           – LMS step-size  (default 1.0)
    T   : float           – time constant in  W = exp(−L·T)  (default 1.0)
    """

    def __init__(self, B, mu: float = 1.0, T: float = 1.0):
        B = np.asarray(B, dtype=float)
        N = B.shape[0]

        self.B = B
        self.N = N
        self.mu = mu
        self.T = T

        self.edge_pairs = map_B_to_set(B)   # list of (i, j) per edge

        # Centering projector  (1/N) · 11ᵀ
        ones = np.ones((N, 1))
        self.dc_proj = (1.0 / N) * (ones @ ones.T)   # (N, N)

        # Algorithm state
        self.W_bar = np.zeros((N, N))   # W̄_0
        self.prev_s_bar = None          # s̄_{i-1}, initialised on first call

    # ------------------------------------------------------------------
    def _center(self, s: np.ndarray) -> np.ndarray:
        """Remove the graph-mean component:  s̄ = s − (1/N)·11ᵀ·s."""
        s = s.reshape(-1, 1)
        return s - self.dc_proj @ s     # (N, 1)

    # ------------------------------------------------------------------
    def _get_laplacian(self) -> np.ndarray:
        """
        Recover Laplacian from the current W̄:

            W   = W̄ + (1/N)·11ᵀ
            L̂   = −(1/T) · logm(W)
        """
        W = self.W_bar + self.dc_proj
        L_hat = -(1.0 / self.T) * logm(W)
        return L_hat.real               # discard tiny imaginary residuals

    # ------------------------------------------------------------------
    def __call__(self, q, y, *args, **kwargs) -> np.ndarray:
        """
        Process one time step.

        Parameters
        ----------
        q    : (N, 1) ndarray – excitation signal (unused by LMS, kept for
                                interface compatibility)
        y    : (N, 1) ndarray – observed graph signal  s_i
        *args / **kwargs      – accept (and ignore) updated_connections, etc.

        Returns
        -------
        x_est : (m, 1) ndarray – estimated edge weights (clamped ≥ 0)
        """
        prev_s_bar = self._center(q)
        s_bar_i = self._center(y)

        # # First iteration: no previous sample yet – store and return zeros
        # if self.prev_s_bar is None:
        #     self.prev_s_bar = s_bar_i
        #     return np.zeros((self.B.shape[1], 1))

        # ── 2. Prediction error ──────────────────────────────────────
        e_i = s_bar_i - self.W_bar @ prev_s_bar   # (N, 1)

        # ── 3. LMS update ────────────────────────────────────────────
        self.W_bar = self.W_bar + self.mu * (e_i @ prev_s_bar.T)

        # ── 4. Recover Laplacian and extract edge weights ────────────
        L_hat = self._get_laplacian()
        x_est = _extract_edge_weights(L_hat, self.B, self.edge_pairs)
        x_est = np.maximum(x_est, 0.0)   # edge weights are non-negative

        # self.prev_s_bar = s_bar_i
        return x_est


# ---------------------------------------------------------------------------
# Grid search over mu — T solved analytically
# ---------------------------------------------------------------------------
def grid_search_lms(B, q_meas, y_meas, pos, mu_grid,
                    lms_class=None) -> dict:
    """
    Find the best step-size μ and the matching optimal T for a given LMS class.

    Key insight
    -----------
    With T = 1, the extractor gives a base estimate  x_base_i  at each step.
    With a general T the estimate is simply scaled:

        x_est_i = (1/T) · x_base_i  =  s · x_base_i,   s = 1/T

    So, given the true states {x_true_i}, the optimal scalar s (minimising
    the sum of squared errors over the whole trajectory) has a closed form:

        s*  =  Σ_i  x_base_i · x_true_i
               ─────────────────────────
               Σ_i  ‖x_base_i‖²

        T*  =  1 / s*

    The function therefore:
      1. Loops over μ values in mu_grid.
      2. For each μ, runs the LMS algorithm with T = 1 to collect {x_base_i}.
      3. Solves for T* analytically.
      4. Computes the normalised MSE trajectory at (μ, T*).
      5. Returns the best (μ, T*) and the full results table.

    Parameters
    ----------
    B         : (N, m) ndarray – incidence matrix
    q_meas    : list of (N,1) arrays – excitation signals  (s_{i-1} in LMS)
    y_meas    : list of (N,1) arrays – observed graph signals  (s_i in LMS)
    pos       : list of (m,1) arrays – true edge-weight states
    mu_grid   : 1-D array-like       – μ values to search over
    lms_class : class, optional      – LaplacianLMS or LaplacianLMSConstrained
                                       (defaults to LaplacianLMSConstrained)

    Returns
    -------
    best_mu   : float
    best_T    : float
    results   : dict  {mu: {'T_opt': float, 'nmse': (T_steps,) array,
                             'avg_nmse': float}}
    """
    if lms_class is None:
        lms_class = LaplacianLMSConstrained

    results = {}

    for mu in mu_grid:
        method = lms_class(B, mu=float(mu), T=1.0)   # T=1 → x_est = x_base

        x_base_list = []
        x_true_list = []

        for q, y, x_true in zip(q_meas, y_meas, pos):
            x_base = method(q, y)                     # (m, 1), T=1 estimate
            x_base_list.append(x_base.ravel())
            x_true_list.append(x_true.ravel())

        x_base_arr = np.array(x_base_list)            # (T_steps, m)
        x_true_arr = np.array(x_true_list)            # (T_steps, m)

        # ── Analytical optimal T ─────────────────────────────────────
        #   s* = Σ(x_base · x_true) / Σ‖x_base‖²  with s = 1/T
        numer = float(np.sum(x_base_arr * x_true_arr))
        denom = float(np.sum(x_base_arr ** 2))

        if denom < 1e-12 or numer <= 0:
            T_opt = 1.0
        else:
            T_opt = denom / numer                     # 1/s* = Σ‖x_base‖² / Σ(x_base·x_true)

        # ── Rescale and compute normalised MSE ───────────────────────
        x_est_arr = np.maximum(x_base_arr / T_opt, 0.0)

        diff        = x_est_arr - x_true_arr           # (T_steps, m)
        sq_err      = np.sum(diff ** 2, axis=1)        # (T_steps,)
        sq_true     = np.sum(x_true_arr ** 2, axis=1)  # (T_steps,)
        nmse        = sq_err / np.maximum(sq_true, 1e-12)

        results[float(mu)] = {
            'T_opt':    T_opt,
            'nmse':     nmse,
            'avg_nmse': float(nmse.mean()),
        }

    best_mu = min(results, key=lambda m: results[m]['avg_nmse'])
    best_T  = results[best_mu]['T_opt']
    return best_mu, best_T, results


# ---------------------------------------------------------------------------
# Constrained variant (Algorithm 2)
# ---------------------------------------------------------------------------
class LaplacianLMSConstrained(LaplacianLMS):
    """
    Constrained Laplacian LMS.

    After the unconstrained gradient step the estimate W̄′ is projected
    onto the feasible set in three sequential steps:

      1. Element-wise non-negativity of W = W̄ + (1/N)·11ᵀ:
             [W̄′]_{pq} = max([W̄′]_{pq}, −1/N)

      2. Null-space / row-sum-zero projection  (ensures W̄·1 = 0,
         i.e. each row of W sums to 1):
             W̄″ = W̄′ − (1/N) · W̄′ · 11ᵀ

      3. Symmetry projection:
             W̄‴ = (W̄″ + W̄″ᵀ) / 2

      4. Spectral projection: clip eigenvalues to [0, 1]
             V, Λ = eigh(W̄‴)          # W̄‴ symmetric → real eigenvalues
             W̄   = V · diag(clamp(Λ, 0, 1)) · Vᵀ

    Parameters are identical to LaplacianLMS.
    """

    def __call__(self, q, y, *args, **kwargs) -> np.ndarray:
        prev_s_bar = self._center(q)
        s_bar_i    = self._center(y)

        # ── 1. Gradient step (same as unconstrained) ─────────────────
        e_i = s_bar_i - self.W_bar @ prev_s_bar          # (N, 1)
        W_prime = self.W_bar + self.mu * (e_i @ prev_s_bar.T)

        # ── 2. Element-wise non-negativity:  W̄′_{pq} ≥ −1/N ─────────
        W_prime = np.maximum(W_prime, -1.0 / self.N)

        # ── 3. Row-sum-zero projection:  W̄″ = W̄′ − (1/N)·W̄′·11ᵀ ───
        #   Subtracts each row's mean so that W̄″·1 = 0
        W_double_prime = W_prime - (1.0 / self.N) * (W_prime @ np.ones((self.N, self.N)))

        # ── 4. Symmetry projection ────────────────────────────────────
        W_triple_prime = 0.5 * (W_double_prime + W_double_prime.T)

        # ── 5. Spectral projection: clamp eigenvalues to [0, 1] ──────
        #   eigh exploits symmetry → real eigenvalues, orthonormal V
        eigvals, V = np.linalg.eigh(W_triple_prime)
        eigvals_clamped = np.clip(eigvals, 0.0, 1.0)
        self.W_bar = (V * eigvals_clamped) @ V.T

        # ── 6. Recover Laplacian and extract edge weights ─────────────
        L_hat = self._get_laplacian()
        x_est = _extract_edge_weights(L_hat, self.B, self.edge_pairs)
        x_est = np.maximum(x_est, 0.0)
        return x_est


# ---------------------------------------------------------------------------
# Quick standalone example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import pickle
    import logging

    from constants import cfg_linear, LABELS, METHODS_ORDER
    from util_func import get_trajectory, one_method_evaluation, plot_metric

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_folder_name = "Results"
    os.makedirs(data_folder_name, exist_ok=True)

    cfg = cfg_linear.copy()

    # ── Grid search: find best mu and optimal T ────────────────────────────
    mu_grid = np.logspace(-3, 1, 20)   # 20 values from 0.001 to 10

    pos, q_meas, y_meas, conn, stateInit = get_trajectory(
        cfg["trajectory_time"], cfg["F"], cfg["B"],
        cfg["C_w_sqrt"], cfg["C_u_sqrt"], cfg["n"], cfg["k"],
        cfg["poly_coefficients"], cfg["new_edge_weight"],
        cfg["num_edges_stateinit"], cfg["delta_n"],
    )

    best_mu, best_T, gs_results = grid_search_lms(
        cfg["B"], q_meas, y_meas, pos, mu_grid,
        lms_class=LaplacianLMSConstrained,
    )
    logging.info(f"Grid search done → best μ = {best_mu:.4g},  optimal T = {best_T:.4g}")
    for mu_val, res in gs_results.items():
        logging.info(f"  μ={mu_val:.4g}  T*={res['T_opt']:.4g}  avg-NMSE={res['avg_nmse']:.4g}")

    cfg["mu"]    = best_mu
    cfg["lms_T"] = best_T

    # ── Single (non-parallel) Monte-Carlo run with tuned parameters ────────
    runs_lms = []
    for _ in range(cfg["num_iterations"]):
        pos, q_meas, y_meas, conn, stateInit = get_trajectory(
            cfg["trajectory_time"], cfg["F"], cfg["B"],
            cfg["C_w_sqrt"], cfg["C_u_sqrt"], cfg["n"], cfg["k"],
            cfg["poly_coefficients"], cfg["new_edge_weight"],
            cfg["num_edges_stateinit"], cfg["delta_n"],
        )
        method = LaplacianLMSConstrained(cfg["B"], mu=cfg.get("mu", 1.0), T=cfg.get("lms_T", 1.0))
        mse, nmse, f1, eier, neier, times = one_method_evaluation(
            method, q_meas, y_meas, pos, conn
        )
        runs_lms.append(dict(lms=dict(mse=mse, normalized_mse=nmse, f1=f1,
                                      eier=eier, normalized_eier=neier, times=times)))

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(data_folder_name, "runs_lms_linear.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(runs_lms, f)
    logging.info(f"Results saved to {out_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    lms_labels = {"lms": "Lap-LMS"}
    lms_methods = ["lms"]

    plot_metric(cfg["trajectory_time"], runs_lms, "mse",
                labels=lms_labels, methods_to_plot=lms_methods,
                log_format=True, to_save=True, folder_name=data_folder_name, suffix="lms_linear")
    plot_metric(cfg["trajectory_time"], runs_lms, "f1",
                labels=lms_labels, methods_to_plot=lms_methods,
                to_save=True, folder_name=data_folder_name, suffix="lms_linear")
    plot_metric(cfg["trajectory_time"], runs_lms, "eier",
                labels=lms_labels, methods_to_plot=lms_methods,
                to_save=True, folder_name=data_folder_name, suffix="lms_linear")
    plot_metric(cfg["trajectory_time"], runs_lms, "times",
                labels=lms_labels, methods_to_plot=lms_methods,
                log_format=True, to_save=True, folder_name=data_folder_name, suffix="lms_linear")