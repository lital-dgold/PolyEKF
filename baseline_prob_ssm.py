# -*- coding: utf-8 -*-
"""
Baseline: Probabilistic State-Space Model (ProbSSM)
====================================================

Signal model (simulation)
--------------------------
    y_t  =  H_t · x_t  +  noise_t
    H_t  =  B · diag(Bᵀ · q_t)          (N × m)
    x_t  =  edge weights  (m × 1)

Binary-edge approximation
--------------------------
Each edge e is assumed to be either active (weight = w_ref) or
inactive (weight = 0).  The estimator tracks a per-node belief over
which of its incident edges are active.

For node n the local state is a binary vector c_n ∈ {0,1}^{deg(n)}
where deg(n) is the number of edges incident to n.  There are
2^{deg(n)} possible states.  ⚠ Complexity is exponential in degree;
for a complete N-node graph deg(n) = N-1, giving 2^(N-1) states per
node.  Practical only for N ≲ 12.

Per-node predict-update cycle (mirrors AirportDataExperiment.ipynb)
--------------------------------------------------------------------
Predict:
    b_n ← (1 − p_e) · b_n  +  p_e / K_n          (K_n = 2^{deg(n)})

Update (Bayesian, Gaussian likelihood on y_n):
    ŷ_n(s) = H_t[n, inc(n)] · (s · w_ref)          for each state s
    likelihood(s) = N(y_n ; ŷ_n(s), σ²)
    b_n ← b_n ⊙ likelihood  (then normalise)

Edge weight reconstruction
--------------------------
For each edge e = (i, j) the posterior probability that it is active
is estimated from both endpoint beliefs and averaged:

    P(x_e = w_ref) ≈ (p_i(e) + p_j(e)) / 2

Final weight: x̂_e = w_ref · P(x_e = w_ref).

Corresponds to 'prob' method in AirportDataExperiment.ipynb.
"""

import numpy as np
from itertools import product
from util_func import map_B_to_set


class ProbSSMBaseline:
    """
    Per-node Bayesian binary-state SSM for graph-weight tracking.

    Parameters
    ----------
    B       : (N, m) ndarray – incidence matrix of the complete graph
    p_e     : float          – probability of an edge changing state per step
    sigma   : float          – observation noise std (for Gaussian likelihood)
    w_ref   : float          – reference weight when an edge is "active"
    state_est : str          – 'avg'  → weighted average of states
                               'map'  → MAP (most probable state)
    """

    def __init__(self, B, poly_c,
                 p_e: float = 0.05,
                 sigma: float = 0.25,
                 w_ref: float = 1.0,
                 state_est: str = 'avg'):

        B = np.asarray(B, dtype=float)
        self.B         = B
        self.N         = B.shape[0]
        self.poly_c = np.asarray(poly_c, dtype=float)
        self.m         = B.shape[1]
        self.p_e       = p_e
        self.sigma     = sigma
        self.w_ref     = w_ref
        self.state_est = state_est

        self.edge_pairs = map_B_to_set(B)   # list of (i, j) per edge

        # For each node n: indices of its incident edges, and state table
        self._incident   = []   # incident[n] = sorted list of edge indices
        self._states     = []   # states[n]   = (K_n, deg_n) binary matrix
        self._beliefs    = []   # beliefs[n]  = (K_n,) probability vector

        for n in range(self.N):
            inc = [e for e, (i, j) in enumerate(self.edge_pairs)
                   if i == n or j == n]
            deg = len(inc)
            # All 2^deg binary states as (K, deg) array
            states_n = np.array(list(product([0, 1], repeat=deg)), dtype=float)
            K_n = states_n.shape[0]
            self._incident.append(inc)
            self._states.append(states_n)
            self._beliefs.append(np.full(K_n, 1.0 / K_n))  # uniform init

    # ------------------------------------------------------------------
    def _gaussian_pdf(self, x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # ------------------------------------------------------------------
    def __call__(self, q, y, *args, **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        q : (N, 1) – excitation signal (defines H_t)
        y : (N, 1) – observation

        Returns
        -------
        x_est : (m, 1) – estimated edge weights
        """
        q = q.reshape(-1)
        y = y.reshape(-1)

        # H_t = B · diag(Bᵀ q),  shape (N, m)
        Bt_q = self.B.T @ q
        H_t  = self.B * Bt_q[np.newaxis, :]

        # --- Predict + Update for each node ---------------------------
        for n in range(self.N):
            inc      = self._incident[n]
            states_n = self._states[n]    # (K_n, deg_n)
            K_n      = states_n.shape[0]

            # Predict: mix toward uniform with probability p_e
            b = (1.0 - self.p_e) * self._beliefs[n] + self.p_e / K_n

            # Expected observation y_n for each state:
            # ŷ_n(s) = Σ_e  H_t[n, e] · s_e · w_ref
            H_n_inc  = H_t[n, inc]                         # (deg_n,)
            y_hat    = states_n @ H_n_inc * self.w_ref      # (K_n,)

            # Gaussian likelihood
            likelihood = self._gaussian_pdf(y[n], y_hat, self.sigma)

            b = b * likelihood
            b_sum = b.sum()
            if b_sum > 0:
                b /= b_sum
            else:
                b = np.full(K_n, 1.0 / K_n)

            self._beliefs[n] = b

        # --- Reconstruct edge weights ----------------------------------
        # For edge e=(i,j): average the marginal P(e active) from both endpoints
        x_est = np.zeros((self.m, 1))
        for e, (i, j) in enumerate(self.edge_pairs):
            p_i = self._marginal(i, e)
            p_j = self._marginal(j, e)
            x_est[e, 0] = self.w_ref * (p_i + p_j) / 2.0

        return x_est

    # ------------------------------------------------------------------
    def _marginal(self, n: int, e: int) -> float:
        """
        Marginal probability P(edge e is active) from node n's belief.
        """
        inc      = self._incident[n]
        if e not in inc:
            return 0.0
        local_idx = inc.index(e)
        states_n  = self._states[n]       # (K_n, deg_n)
        b         = self._beliefs[n]      # (K_n,)

        if self.state_est == 'map':
            best_state = states_n[np.argmax(b)]
            return float(best_state[local_idx])
        else:  # 'avg'
            return float(b @ states_n[:, local_idx])


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
    for _ in range(min(cfg["num_iterations"],10)):
        pos, q_meas, y_meas, conn, stateInit = get_trajectory(
            cfg["trajectory_time"], cfg["F"], cfg["B"],
            cfg["C_w_sqrt"], cfg["C_u_sqrt"], cfg["n"], cfg["k"],
            cfg["poly_coefficients"], cfg["new_edge_weight"],
            cfg["num_edges_stateinit"], cfg["delta_n"],
        )
        method = ProbSSMBaseline(
            cfg["B"], cfg["poly_coefficients"],
        )
        mse, nmse, f1, eier, neier, times = one_method_evaluation(
            method, q_meas, y_meas, pos, conn
        )
        runs_grls.append(dict(grls=dict(mse=mse, normalized_mse=nmse, f1=f1,
                                        eier=eier, normalized_eier=neier,
                                        times=times)))

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(data_folder_name, "runs_ProbSSM_linear.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(runs_grls, f)
    logging.info(f"Results saved to {out_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    grls_labels  = {"ProbSSM": "ProbSSM"}
    grls_methods = ["ProbSSM"]

    plot_metric(cfg["trajectory_time"], runs_grls, "mse",
                labels=grls_labels, methods_to_plot=grls_methods,
                log_format=True, to_save=True,
                folder_name=data_folder_name, suffix="ProbSSM_linear")
    plot_metric(cfg["trajectory_time"], runs_grls, "f1",
                labels=grls_labels, methods_to_plot=grls_methods,
                to_save=True, folder_name=data_folder_name, suffix="ProbSSM_linear")
    plot_metric(cfg["trajectory_time"], runs_grls, "eier",
                labels=grls_labels, methods_to_plot=grls_methods,
                to_save=True, folder_name=data_folder_name, suffix="ProbSSM_linear")
    plot_metric(cfg["trajectory_time"], runs_grls, "times",
                labels=grls_labels, methods_to_plot=grls_methods,
                log_format=True, to_save=True,
                folder_name=data_folder_name, suffix="ProbSSM_linear")