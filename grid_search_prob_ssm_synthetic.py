"""
grid_search_prob_ssm_synthetic.py
-----------------------------------
Grid search over p_e, sigma, w_ref, and state_est for ProbSSMBaseline
on synthetic data (vs-time simulation).

⚠  COMPUTATIONAL WARNING ⚠
ProbSSMBaseline has complexity exponential in node degree: 2^(N-1) states per
node. For a complete graph:
  N=10  →  2^9  = 512   states/node  (feasible, ~seconds per run)
  N=20  →  2^19 = 524k  states/node  (infeasible)

cfg_linear uses a 20-node graph — if you intend to run this, switch to a
smaller cfg (e.g. cfg_non_linear_case2 which is N=10) or reduce the graph.

Experiment types (set EXPERIMENT_TYPE below):
  "linear"          — cfg_linear          (20-node, [0, 1.0]         — ⚠ infeasible)
  "nonlinear_case2" — cfg_non_linear_case2 (10-node, highly nonlinear — feasible)

Ranking metric:
  score = avg_mse + VARIANCE_WEIGHT * avg_std

Outputs (saved to SAVE_DIR):
  traj_rank<N>_*.png                   — top N_PLOT_BEST trajectory plots
  heatmap_<metric>_<p1>_vs_<p2>.png   — 2D heat-maps for every parameter pair
  grid_results.pkl                     — raw results dict
  summary.txt                          — all combos ranked by score
"""

import copy
import itertools
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from util_func import get_trajectory, one_method_evaluation, pick_worker_count
from baseline_prob_ssm import ProbSSMBaseline
from constants import cfg_linear, cfg_non_linear_case2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

METHOD_TO_OPT = "prob_ssm"

# ─── Experiment selector ──────────────────────────────────────────────────────
# "linear"          : 20-node linear model  (cfg_linear)       ⚠ likely infeasible
# "nonlinear_case2" : 10-node highly nonlinear (cfg_non_linear_case2) ← recommended
EXPERIMENT_TYPE = "linear"

# ─── Grid search configuration ────────────────────────────────────────────────
P_E_GRID       = [0.01, 0.05, 0.1, 0.2]        # edge-change probability per step
SIGMA_GRID     = [0.01, 0.05, 0.1, 0.25, 0.5]  # observation noise std
W_REF_GRID     = [0.5, 1.0, 2.0]               # reference active-edge weight
STATE_EST_GRID = ["avg", "map"]                 # posterior readout method

N_MC_PER_COMBO  = 10    # Monte-Carlo runs per parameter combo
WINDOW_A        = (0, 40)
N_PLOT_BEST     = 10
VARIANCE_WEIGHT = 0.5   # score = avg_mse + VARIANCE_WEIGHT * avg_std

# ─── Base configs ─────────────────────────────────────────────────────────────
_CFGS = {
    "linear":          copy.deepcopy(cfg_linear),
    "nonlinear_case2": copy.deepcopy(cfg_non_linear_case2),
}

SAVE_DIR = f"Results/grid_search_{METHOD_TO_OPT}_synthetic_{EXPERIMENT_TYPE}"
os.makedirs(SAVE_DIR, exist_ok=True)


# ─── Worker ───────────────────────────────────────────────────────────────────

def _run_mc_task(args):
    """Top-level wrapper for ProcessPoolExecutor (pickling on Windows)."""
    p_e, sigma, w_ref, state_est, _mc_idx, experiment_type = args
    base_cfg = _CFGS[experiment_type]
    cfg_run  = copy.deepcopy(base_cfg)

    pos, q_meas, y_meas, conn, _ = get_trajectory(
        cfg_run["trajectory_time"], cfg_run["F"], cfg_run["B"],
        cfg_run["C_w_sqrt"], cfg_run["C_u_sqrt"], cfg_run["n"],
        cfg_run["k"], cfg_run["poly_coefficients"],
        cfg_run["new_edge_weight"], cfg_run["num_edges_stateinit"],
        cfg_run["delta_n"],
    )

    filt = ProbSSMBaseline(
        cfg_run["B"], cfg_run["poly_coefficients"],
        p_e=p_e, sigma=sigma, w_ref=w_ref, state_est=state_est,
    )
    mse, *_ = one_method_evaluation(filt, q_meas, y_meas, pos, conn)
    return p_e, sigma, w_ref, state_est, mse.ravel()   # (T,)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _gap_metric(trajectories, window):
    start, end = window
    values = [t[start:end].max() - t[start:end].min()
              for t in trajectories if len(t) > start]
    return float(np.mean(values)) if values else float("nan")


# ─── Run ──────────────────────────────────────────────────────────────────────

def _run_grid(param_grid, n_workers):
    tasks = [(p_e, sigma, w_ref, state_est, mc_idx, EXPERIMENT_TYPE)
             for p_e, sigma, w_ref, state_est in param_grid
             for mc_idx in range(N_MC_PER_COMBO)]

    logging.info(f"[{EXPERIMENT_TYPE}]  Combos: {len(param_grid)}  MC/combo: {N_MC_PER_COMBO}  "
                 f"Tasks: {len(tasks)}  Workers: {n_workers}")

    raw: dict = {}
    t_wall = time.time()
    done   = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_mc_task, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            p_e, sigma, w_ref, state_est, mc_idx, _ = futures[fut]
            try:
                p_e_, sigma_, w_ref_, state_est_, traj = fut.result()
                raw.setdefault((p_e_, sigma_, w_ref_, state_est_), []).append(traj)
            except Exception as exc:
                logging.warning(f"  [{done}/{len(tasks)}] Skipped "
                                f"p_e={p_e} σ={sigma} w_ref={w_ref} "
                                f"state_est={state_est} mc={mc_idx}: {exc}")
                continue
            logging.info(f"  [{done}/{len(tasks)}]  p_e={p_e}  σ={sigma}  "
                         f"w_ref={w_ref}  state_est={state_est}  mc={mc_idx}  "
                         f"({time.time()-t_wall:.1f}s)")

    logging.info(f"All tasks done in {time.time()-t_wall:.1f}s")

    results_grid = {}
    for params, trajectories in raw.items():
        per_run_mse = [t.mean() for t in trajectories]
        avg_mse = float(np.mean(per_run_mse))
        avg_std = float(np.std(per_run_mse))
        gap     = _gap_metric(trajectories, WINDOW_A)
        score   = avg_mse + VARIANCE_WEIGHT * avg_std
        results_grid[params] = dict(
            avg_mse=avg_mse,
            avg_std=avg_std,
            score=score,
            gap_a=gap,
            trajectories=trajectories,
        )

    return results_grid


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_trajectories(results_grid):
    ranked = sorted(results_grid.items(), key=lambda kv: kv[1]["score"])

    for rank, (params, res) in enumerate(ranked[:N_PLOT_BEST]):
        p_e, sigma, w_ref, state_est = params
        traj_mat  = np.stack(res["trajectories"])
        mean_traj = traj_mat.mean(axis=0)
        std_traj  = traj_mat.std(axis=0)
        T = mean_traj.shape[0]
        t = np.arange(T)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t, mean_traj, color="steelblue", linewidth=2, label="mean MSE")
        ax.fill_between(t, mean_traj - std_traj, mean_traj + std_traj,
                        alpha=0.25, color="steelblue", label="±1 std")
        wa_end = min(WINDOW_A[1], T)
        ax.axvspan(WINDOW_A[0], wa_end, alpha=0.10, color="green",
                   label=f"Window [{WINDOW_A[0]},{wa_end})  gap={res['gap_a']:.4f}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("MSE")
        ax.set_title(
            f"ProbSSM ({EXPERIMENT_TYPE})  p_e={p_e}  σ={sigma}  "
            f"w_ref={w_ref}  state_est={state_est}\n"
            f"avg MSE={res['avg_mse']:.4f}   std={res['avg_std']:.4f}   "
            f"score={res['score']:.4f}   gap={res['gap_a']:.4f}   "
            f"({len(res['trajectories'])} MC runs)"
        )
        ax.legend(fontsize=9, loc="upper right")
        plt.tight_layout()
        fname = f"traj_rank{rank+1:02d}_pe{p_e}_s{sigma}_wr{w_ref}_{state_est}.png"
        plt.savefig(os.path.join(SAVE_DIR, fname), dpi=100)
        plt.close(fig)
        logging.info(f"  Saved → {fname}")

    return ranked


def _save_heatmap_2d(results_grid, metric_key, param_i, param_j,
                     grid_i, grid_j, param_names):
    """2D heatmap for param_i vs param_j; projects over remaining dims by min."""
    mat = np.full((len(grid_i), len(grid_j)), np.nan)

    for ri, vi in enumerate(grid_i):
        for rj, vj in enumerate(grid_j):
            candidates = [res[metric_key] for key, res in results_grid.items()
                          if key[param_i] == vi and key[param_j] == vj
                          and not np.isnan(res[metric_key])]
            if candidates:
                mat[ri, rj] = min(candidates)

    if np.all(np.isnan(mat)):
        return

    xlabel = param_names[param_j]
    ylabel = param_names[param_i]
    other_dims = [d for d in range(len(param_names)) if d not in (param_i, param_j)]
    other_names = " & ".join(param_names[d] for d in other_dims)
    metric_labels = {"avg_mse": "Avg MSE", "avg_std": "Avg Std", "score": "Score", "gap_a": "Gap"}
    metric_label = metric_labels.get(metric_key, metric_key)

    fig, ax = plt.subplots(figsize=(max(6, len(grid_j) * 1.4), max(4, len(grid_i) * 1.1)))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(grid_j))); ax.set_xticklabels(grid_j, fontsize=9)
    ax.set_yticks(range(len(grid_i))); ax.set_yticklabels(grid_i, fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{metric_label} — ProbSSM ({EXPERIMENT_TYPE})  (min over {other_names})")
    plt.colorbar(im, ax=ax)

    for ri in range(len(grid_i)):
        for rj in range(len(grid_j)):
            if not np.isnan(mat[ri, rj]):
                ax.text(rj, ri, f"{mat[ri, rj]:.3f}", ha="center", va="center",
                        fontsize=8, color="white")

    if (~np.isnan(mat)).any():
        best = np.unravel_index(np.nanargmin(mat), mat.shape)
        ax.add_patch(plt.Rectangle(
            (best[1] - 0.5, best[0] - 0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=2.5, label="best"
        ))
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"heatmap_{metric_key}_{ylabel}_vs_{xlabel}.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    logging.info(f"  Saved heatmap → {path}")


def _write_summary(ranked, summary_path):
    with open(summary_path, "w") as fh:
        header = (f"{'rank':>4}  {'p_e':>6}  {'sigma':>7}  {'w_ref':>7}  "
                  f"{'state_est':>10}  {'avg_mse':>10}  {'avg_std':>10}  "
                  f"{'score':>10}  {'gap':>10}\n")
        fh.write(header)
        fh.write("-" * len(header) + "\n")
        for rank, (params, res) in enumerate(ranked):
            p_e, sigma, w_ref, state_est = params
            fh.write(
                f"{rank+1:>4}  {p_e:>6}  {sigma:>7}  {w_ref:>7}  "
                f"{state_est:>10}  {res['avg_mse']:>10.4f}  {res['avg_std']:>10.4f}  "
                f"{res['score']:>10.4f}  {res['gap_a']:>10.4f}\n"
            )
    logging.info(f"Saved summary → {summary_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os as _os
    for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        _os.environ[_var] = "1"

    cfg = _CFGS[EXPERIMENT_TYPE]
    n_nodes = cfg["n"]
    max_states_per_node = 2 ** (n_nodes - 1)
    if max_states_per_node > 10_000:
        logging.warning(
            f"N={n_nodes} → 2^{n_nodes-1} = {max_states_per_node:,} states/node. "
            f"ProbSSM will be very slow. Consider switching to EXPERIMENT_TYPE='nonlinear_case2' (N=10)."
        )

    param_grid  = list(itertools.product(P_E_GRID, SIGMA_GRID, W_REF_GRID, STATE_EST_GRID))
    n_workers   = pick_worker_count()
    param_names = ["p_e", "sigma", "w_ref", "state_est"]
    all_grids   = [P_E_GRID, SIGMA_GRID, W_REF_GRID, STATE_EST_GRID]

    # ── Run grid search ────────────────────────────────────────────────────────
    results_grid = _run_grid(param_grid, n_workers)
    ranked       = _plot_trajectories(results_grid)

    # ── Save raw results ───────────────────────────────────────────────────────
    pkl_path = os.path.join(SAVE_DIR, "grid_results.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(results_grid, fh)
    logging.info(f"Saved raw results → {pkl_path}")

    # ── 2D heatmaps for every parameter pair ──────────────────────────────────
    for (pi, pj) in itertools.combinations(range(len(param_names)), 2):
        for metric_key in ("avg_mse", "score"):
            _save_heatmap_2d(results_grid, metric_key, pi, pj,
                             all_grids[pi], all_grids[pj], param_names)

    # ── Summary ───────────────────────────────────────────────────────────────
    _write_summary(ranked, os.path.join(SAVE_DIR, "summary.txt"))

    # ── Print top 5 ───────────────────────────────────────────────────────────
    print(f"\n══════════════ ProbSSM Synthetic ({EXPERIMENT_TYPE}) — Top 5 by score ══════════════")
    for rank, (params, res) in enumerate(ranked[:5]):
        p_e, sigma, w_ref, state_est = params
        print(f"  #{rank+1}  p_e={p_e:<5}  σ={sigma:<6}  w_ref={w_ref:<5}  state_est={state_est:<4}  →  "
              f"avg_mse={res['avg_mse']:.4f}  std={res['avg_std']:.4f}  score={res['score']:.4f}")
    print(f"\nAll results saved to {SAVE_DIR}/")