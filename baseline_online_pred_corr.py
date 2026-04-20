# -*- coding: utf-8 -*-
"""
Baseline: Online Predictor-Corrector for Precision Matrix Tracking
===================================================================

Tracks a symmetric positive definite precision matrix Ŝ ∈ S_N++ via an
online predict-correct gradient-descent scheme.

The per-step objective is the negative Gaussian log-likelihood:

    f(S) = -log|S| + tr(Σ̂_t · S)

with gradient and Hessian w.r.t. vech(S):

    ∇f   = Dᵀ · vec(Σ̂_t − S⁻¹)
    ∇²f  = Dᵀ · (S⁻¹ ⊗ S⁻¹) · D
    ∇_t f = Dᵀ · vec(Σ̂_t − Σ̂_{t-1})          (temporal change of gradient)

where D is the N² × N(N+1)/2 duplication matrix (column-major convention).

Algorithm
---------
REQUIRE: Ŝ₀ ∈ S_N++, forgetting factor γ, step sizes αₜ, βₜ,
         prediction steps P, correction steps C, temporal weight h

FOR t = 0, 1, 2, ...

    PRECOMPUTE (at current Ŝₜ, Σ̂ₜ):
        ∇f     ← Dᵀ · vec(Σ̂ₜ − Ŝₜ⁻¹)
        ∇²f    ← Dᵀ · (Ŝₜ⁻¹ ⊗ Ŝₜ⁻¹) · D
        ∇ₜf    ← Dᵀ · vec(Σ̂ₜ − Σ̂ₜ₋₁)

    PREDICTION (P gradient steps using linearised gradient):
        ŝ⁰ ← ŝₜ
        for p = 0 … P-1:
            d        ← ∇f + ∇²f·(ŝᵖ − ŝₜ) + h·∇ₜf
            ŝ^{p+1} ← prox( ŝᵖ − 2αₜ·d )
        ŝ_{t+1|t} ← ŝᴾ

    UPDATE COVARIANCE (EWMA, new observation x_t available):
        Σ̂_{t+1} ← γ·Σ̂ₜ + (1−γ)·xₜxₜᵀ

    CORRECTION (C true-gradient steps at t+1):
        ŝ⁰ ← ŝ_{t+1|t}
        for c = 0 … C-1:
            ∇f_new  ← Dᵀ · vec(Σ̂_{t+1} − (Ŝᶜ)⁻¹)
            ŝ^{c+1} ← prox( ŝᶜ − βₜ·∇f_new )
        ŝ_{t+1} ← ŝᶜ

Two proximal operator options (selected via prox_type):
    Option A  'psd' : prox(v) = P_{S_N++}(v)
                      PSD projection — clamp eigenvalues of devech(v) to ≥ eps.
    Option B  'l1'  : prox(v) = P_{S_N++}( soft_thresh(v, λ) )
                      ℓ₁ soft-threshold on vech entries (promotes sparsity in
                      the precision matrix), then PSD projection to restore
                      positive definiteness.

    Ŝ_{t+1} ← devech(ŝ_{t+1})

END FOR

Integration with evaluation harness
-------------------------------------
    x_est = method(q, y, updated_connections)

  * q     – excitation (unused, kept for interface compatibility)
  * y     – observed graph signal x_t  (N×1)
  * x_est – estimated edge weights (m×1, ≥ 0)

Edge weights are recovered by projecting Ŝ onto the Laplacian cone
(valid for GMRF models where the precision matrix ≈ graph Laplacian).
"""

import numpy as np

from util_func import map_B_to_set, extract_x_to_match_B_from_L
from baseline_grls import project_to_laplacian, extract_L_from_R


# ---------------------------------------------------------------------------
# vech / devech / duplication matrix  (all use column-major convention)
# ---------------------------------------------------------------------------

def _build_duplication_matrix(N: int) -> np.ndarray:
    """
    Build the N² × N(N+1)/2 duplication matrix D such that

        vec(S) = D · vech(S)

    for any symmetric N×N matrix S, using column-major vec ordering
    (consistent with np.ndarray.ravel(order='F')).
    """
    n_vech = N * (N + 1) // 2
    D = np.zeros((N * N, n_vech))
    col = 0
    for j in range(N):            # outer: column index
        for i in range(j, N):     # inner: row index (lower triangle)
            D[j * N + i, col] = 1.0        # position (i, j) in col-major vec
            if i != j:
                D[i * N + j, col] = 1.0    # symmetric position (j, i)
            col += 1
    return D


def _vech(S: np.ndarray) -> np.ndarray:
    """
    Column-major half-vectorization of symmetric N×N matrix S.
    Returns 1-D array of length N(N+1)/2.
    """
    N = S.shape[0]
    n_vech = N * (N + 1) // 2
    s = np.empty(n_vech)
    col = 0
    for j in range(N):
        for i in range(j, N):
            s[col] = S[i, j]
            col += 1
    return s


def _devech(s: np.ndarray, N: int) -> np.ndarray:
    """
    Reconstruct symmetric N×N matrix from column-major vech vector s.
    """
    S = np.zeros((N, N))
    col = 0
    for j in range(N):
        for i in range(j, N):
            S[i, j] = s[col]
            S[j, i] = s[col]   # symmetrize (diagonal written once, fine)
            col += 1
    return S


def _project_vech_to_spd(s: np.ndarray, N: int, eps: float = 1e-6) -> np.ndarray:
    """
    Project a vech vector onto S_N++ by:
        1. Reconstruct S = devech(s)
        2. Symmetrize: S ← (S + Sᵀ) / 2
        3. Clamp eigenvalues to ≥ eps
        4. Return vech of result
    """
    S = _devech(s, N)
    S_sym = (S + S.T) / 2.0
    eigvals, V = np.linalg.eigh(S_sym)
    S_proj = (V * np.maximum(eigvals, eps)) @ V.T
    return _vech(S_proj)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OnlinePredCorrEstimator:
    """
    Online Predictor-Corrector for precision matrix tracking (Ŝ ∈ S_N++).

    Parameters
    ----------
    B           : (N, m) ndarray  – incidence matrix of the complete graph
    gamma       : float           – EWMA forgetting factor ∈ (0, 1)
    alpha       : float or callable(t) -> float  – prediction step size αₜ
    beta        : float or callable(t) -> float  – correction step size βₜ
    h           : float           – temporal gradient weight
    P           : int             – prediction gradient steps per time step
    C           : int             – correction gradient steps per time step
    prox_type   : str             – proximal operator: 'psd' (Option A) or
                                    'l1' (Option B)
    lambda_prox : float           – ℓ₁ soft-threshold parameter (Option B only)
    S0          : (N, N) ndarray or None  – initial Ŝ (default: identity)
    Sigma0      : (N, N) ndarray or None  – initial EWMA covariance (default: identity)
    eps_psd     : float           – minimum eigenvalue for PSD projection
    """

    def __init__(self, B, poly_c,
                 gamma: float = 0.99,
                 alpha=0.01,
                 beta=0.01,
                 h: float = 1.0,
                 P: int = 1,
                 C: int = 1,
                 prox_type: str = 'psd',
                 lambda_prox: float = 0.01,
                 S0=None,
                 Sigma0=None,
                 eps_psd: float = 1e-6):

        B = np.asarray(B, dtype=float)
        N = B.shape[0]

        self.B        = B
        self.N        = N
        self.poly_c   = np.asarray(np.convolve(poly_c, poly_c), dtype=float)
        self.gamma    = gamma
        self._alpha   = alpha    # scalar or callable(t)
        self._beta    = beta     # scalar or callable(t)
        self.h        = h
        self.P            = P
        self.C            = C
        self.prox_type    = prox_type
        self.lambda_prox  = lambda_prox
        self.eps_psd      = eps_psd

        self.edge_pairs = map_B_to_set(B)

        # Duplication matrix D and its transpose (cached, shape-invariant)
        self.D  = _build_duplication_matrix(N)   # (N², N(N+1)/2)
        self.Dt = self.D.T                        # (N(N+1)/2, N²)

        # Ŝₜ: current precision matrix estimate
        self.S_hat = np.eye(N) if S0 is None else np.asarray(S0, dtype=float).copy()

        # Σ̂ₜ: EWMA sample covariance
        self.Sigma      = np.eye(N) if Sigma0 is None else np.asarray(Sigma0, dtype=float).copy()
        self.Sigma_prev = self.Sigma.copy()   # Σ̂ₜ₋₁ for temporal gradient

        self.t = 0   # iteration counter

    # ------------------------------------------------------------------
    # Step-size accessors
    # ------------------------------------------------------------------
    def _get_alpha(self) -> float:
        return self._alpha(self.t) if callable(self._alpha) else float(self._alpha)

    def _get_beta(self) -> float:
        return self._beta(self.t) if callable(self._beta) else float(self._beta)

    # ------------------------------------------------------------------
    # Gradient / Hessian helpers
    # ------------------------------------------------------------------
    def _gradient(self, S_inv: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """
        ∇f = Dᵀ · vec(Σ̂ − S⁻¹)   →   (N(N+1)/2,) array.
        vec uses column-major ordering (order='F').
        """
        diff = Sigma - S_inv
        return self.Dt @ diff.ravel(order='F')

    def _hessian(self, S_inv: np.ndarray) -> np.ndarray:
        """
        ∇²f = Dᵀ · (S⁻¹ ⊗ S⁻¹) · D   →   (N(N+1)/2, N(N+1)/2) matrix.
        """
        return self.Dt @ np.kron(S_inv, S_inv) @ self.D

    def _temporal_gradient(self, Sigma: np.ndarray, Sigma_prev: np.ndarray) -> np.ndarray:
        """
        ∇ₜf = Dᵀ · vec(Σ̂ₜ − Σ̂ₜ₋₁)   →   (N(N+1)/2,) array.
        """
        return self.Dt @ (Sigma - Sigma_prev).ravel(order='F')

    def _prox(self, v: np.ndarray) -> np.ndarray:
        """
        Proximal operator applied to a vech vector v.

        Option A  'psd':  P_{S_N++}(v)
            PSD projection — clamp eigenvalues of devech(v) to ≥ eps_psd.

        Option B  'l1':   P_{S_N++}( soft_thresh(v, λ) )
            ℓ₁ soft-threshold (promotes sparsity in the precision matrix
            entries), followed by PSD projection to restore S_N++.

            soft_thresh(v, λ)_i = sign(v_i) · max(|v_i| − λ, 0)
        """
        # if self.prox_type == 'l1':
        #     v = np.sign(v) * np.maximum(np.abs(v) - self.lambda_prox, 0.0)
        return _project_vech_to_spd(v, self.N, self.eps_psd)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def __call__(self, q, y, *args, **kwargs) -> np.ndarray:
        """
        Process one time step.

        Parameters
        ----------
        q : (N, 1) – excitation signal (unused, kept for interface compatibility)
        y : (N, 1) – observed graph signal x_t

        Returns
        -------
        x_est : (m, 1) – estimated edge weights (≥ 0)
        """
        x_t   = y.reshape(-1, 1)
        alpha = self._get_alpha()
        beta  = self._get_beta()

        # ── Precompute at current Ŝₜ and Σ̂ₜ ────────────────────────
        S_inv  = np.linalg.inv(self.S_hat)
        grad_f = self._gradient(S_inv, self.Sigma)           # (n_vech,)
        hess_f = self._hessian(S_inv)                        # (n_vech, n_vech)
        grad_t = self._temporal_gradient(self.Sigma, self.Sigma_prev)  # (n_vech,)

        s_t = _vech(self.S_hat)   # current iterate ŝₜ

        # ── Prediction: P linearised gradient steps ──────────────────
        s_p = s_t.copy()
        for _ in range(self.P):
            d   = grad_f + hess_f @ (s_p - s_t) + self.h * grad_t
            s_p = self._prox(s_p - 2.0 * alpha * d)
        s_pred = s_p   # ŝ_{t+1|t}

        # ── EWMA covariance update (x_t now available) ───────────────
        self.Sigma_prev = self.Sigma.copy()
        if self.t == 0:
            self.Sigma = (x_t @ x_t.T)
        elif self.t > self.N:
            self.Sigma = self.gamma * self.Sigma + (1.0 - self.gamma) * (x_t @ x_t.T)
        else:
            self.Sigma = self.t/(self.t+1) * self.Sigma + 1/(self.t+1) * (x_t @ x_t.T)


        # ── Correction: C true-gradient steps with updated Σ̂_{t+1} ──
        s_c = s_pred.copy()
        for _ in range(self.C):
            S_c      = _devech(s_c, self.N)
            S_c_inv  = np.linalg.inv(S_c)
            grad_new = self._gradient(S_c_inv, self.Sigma)
            s_c = self._prox(s_c - beta * grad_new)

        # ── Store updated estimate ────────────────────────────────────
        self.S_hat = _devech(s_c, self.N)
        self.t += 1

        # ── Extract edge weights (precision ≈ scaled Laplacian) ──────
        L = extract_L_from_R(np.linalg.inv(self.S_hat), self.poly_c)
        L     = project_to_laplacian(L)
        x_est = extract_x_to_match_B_from_L(L, self.B)
        return np.maximum(x_est, 0.0)


# ---------------------------------------------------------------------------
# Quick standalone example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import pickle
    import logging

    from constants import cfg_linear, LABELS, METHODS_ORDER
    from util_func import get_trajectory, one_method_evaluation, plot_metric

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    data_folder_name = "Results"
    os.makedirs(data_folder_name, exist_ok=True)

    cfg = cfg_linear.copy()

    runs = []
    for _ in range(min(cfg["num_iterations"], 10)):   # small run for testing
        pos, q_meas, y_meas, conn, stateInit = get_trajectory(
            cfg["trajectory_time"], cfg["F"], cfg["B"],
            cfg["C_w_sqrt"], cfg["C_u_sqrt"], cfg["n"], cfg["k"],
            cfg["poly_coefficients"], cfg["new_edge_weight"],
            cfg["num_edges_stateinit"], cfg["delta_n"],
        )
        method = OnlinePredCorrEstimator(
            cfg["B"], cfg["poly_coefficients"],
            gamma=0.9,
            alpha=0.1,
            beta=0.1,
            h=1.0,
            P=1,
            C=1,
        )
        mse, nmse, f1, eier, neier, times = one_method_evaluation(
            method, q_meas, y_meas, pos, conn
        )
        runs.append(dict(pred_corr=dict(mse=mse, normalized_mse=nmse, f1=f1,
                                        eier=eier, normalized_eier=neier,
                                        times=times)))

    out_path = os.path.join(data_folder_name, "runs_pred_corr.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(runs, f)
    logging.info(f"Saved to {out_path}")

    labels  = {"pred_corr": "PredCorr"}
    methods = ["pred_corr"]

    plot_metric(cfg["trajectory_time"], runs, "mse",
                labels=labels, methods_to_plot=methods,
                log_format=True, to_save=True,
                folder_name=data_folder_name, suffix="pred_corr")
    plot_metric(cfg["trajectory_time"], runs, "f1",
                labels=labels, methods_to_plot=methods,
                to_save=True, folder_name=data_folder_name, suffix="pred_corr")
    plot_metric(cfg["trajectory_time"], runs, "eier",
                labels=labels, methods_to_plot=methods,
                to_save=True, folder_name=data_folder_name, suffix="pred_corr")