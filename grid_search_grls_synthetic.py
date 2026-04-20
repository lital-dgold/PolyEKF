"""
grid_search_grls_synthetic.py
------------------------------
Grid search over beta, alpha, lambda_r, t_max, and cov_init for GRLSSimple
on synthetic data (vs-time simulation).

Three experiment types (set EXPERIMENT_TYPE below):
  "linear"          — 20-node graph, linear model       (cfg_linear)
  "nonlinear_case1" — 20-node graph, nonlinear case 1   (cfg_non_linear_case1)
  "nonlinear_case2" — 10-node graph, highly nonlinear   (cfg_non_linear_case2)

Ranking metric:
  score = avg_mse + VARIANCE_WEIGHT * avg_std
  where avg_std is the std of per-run mean MSE across MC trials.

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
from constants import (
    METHOD_REGISTRY,
    cfg_linear,
    cfg_non_linear_case1,
    cfg_non_linear_case2,
    cfg_non_linear_vs_filter_order,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

METHOD_TO_OPT = "grls"

# ─── Experiment selector ──────────────────────────────────────────────────────
# "linear"            : 20-node linear model           (cfg_linear)
# "nonlinear_case1"   : 20-node nonlinear model         (cfg_non_linear_case1)
# "nonlinear_case2"   : 10-nodeplotting highly nonlinear model  (cfg_non_linear_case2)
# "vs_poly_order"     : 10-node, sweep polynomial order (cfg_non_linear_vs_filter_order)
# "vs_poly_order_p0"  : 10-node, fixed at p=p_list[0]  — optimise for the first (lowest) poly order
EXPERIMENT_TYPE = "vs_poly_order_p0"
# ─── Grid search configuration ────────────────────────────────────────────────
BETA_GRID     = [0.8, 0.9, 0.95, 0.99]       # RLS forgetting factor
ALPHA_GRID    = [0.001, 0.01, 0.05, 0.1]     # gradient-descent step size
LAMBDA_R_GRID = [0.001, 0.01, 0.1, 1.0]      # L1 sparsity on vec(R)
T_MAX_GRID    = [1, 5, 10, 20]               # gradient-descent iters per step
COV_INIT_GRID = [0.0, 1.0, 10.0]             # initial RLS covariance scale

N_MC_PER_COMBO  = 10    # Monte-Carlo runs per parameter combo
WINDOW_A        = (0, 40)
N_PLOT_BEST     = 10
VARIANCE_WEIGHT = 0.5   # score = avg_mse + VARIANCE_WEIGHT * avg_std

# stateInit_weight_grls is fixed at 0.0 (same as the base configs)
INIT_WEIGHT = 0.0

SAVE_DIR = f"Results/grid_search_{METHOD_TO_OPT}_synthetic_{EXPERIMENT_TYPE}"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Base configs ─────────────────────────────────────────────────────────────
_cfg_vs_poly_order = copy.deepcopy(cfg_non_linear_vs_filter_order)

# Build fixed-p0 config: poly_coefficients set to the first (lowest) p in p_list
_p0 = int(_cfg_vs_poly_order["p_list"][0])
_poly_p0 = 1.0 / np.geomspace(1, 2 ** _p0, num=_p0 + 1)
_cfg_vs_poly_order_p0 = copy.deepcopy(_cfg_vs_poly_order)
_cfg_vs_poly_order_p0["poly_coefficients"] = _poly_p0

_CFGS = {
    "linear":           copy.deepcopy(cfg_linear),
    "nonlinear_case1":  copy.deepcopy(cfg_non_linear_case1),
    "nonlinear_case2":  copy.deepcopy(cfg_non_linear_case2),
    "vs_poly_order":    _cfg_vs_poly_order,
    "vs_poly_order_p0": _cfg_vs_poly_order_p0,   # fixed at p=p_list[0] (p={_p0})
}


# ─── Worker ───────────────────────────────────────────────────────────────────

def _run_one_mc(beta, alpha, lambda_r, t_max, cov_init, base_cfg):
    """Run GRLSSimple on one synthetic MC trial. Returns MSE trajectory (T,)."""
    cfg_run = copy.deepcopy(base_cfg)
    cfg_run["beta"]                 = beta
    cfg_run["alpha"]                = alpha
    cfg_run["lambda_r"]             = lambda_r
    cfg_run["t_max"]                = t_max
    cfg_run["cov_init"]             = cov_init
    cfg_run["stateInit_weight_grls"] = INIT_WEIGHT

    pos, q_meas, y_meas, conn, _ = get_trajectory(
        cfg_run["trajectory_time"], cfg_run["F"], cfg_run["B"],
        cfg_run["C_w_sqrt"], cfg_run["C_u_sqrt"], cfg_run["n"],
        cfg_run["k"], cfg_run["poly_coefficients"],
        cfg_run["new_edge_weight"], cfg_run["num_edges_stateinit"],
        cfg_run["delta_n"],
    )

    filt = METHOD_REGISTRY[METHOD_TO_OPT](cfg_run)
    mse, *_ = one_method_evaluation(filt, q_meas, y_meas, pos, conn)
    return mse.ravel()   # (T,)


def _run_mc_task(args):
    """Top-level wrapper for ProcessPoolExecutor (pickling on Windows)."""
    beta, alpha, lambda_r, t_max, cov_init, _mc_idx, experiment_type = args
    traj = _run_one_mc(beta, alpha, lambda_r, t_max, cov_init, _CFGS[experiment_type])
    return beta, alpha, lambda_r, t_max, cov_init, traj


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _gap_metric(trajectories, window):
    start, end = window
    values = [t[start:end].max() - t[start:end].min()
              for t in trajectories if len(t) > start]
    return float(np.mean(values)) if values else float("nan")


# ─── Run ──────────────────────────────────────────────────────────────────────

def _run_grid(param_grid, n_workers):
    tasks = [(beta, alpha, lambda_r, t_max, cov_init, mc_idx, EXPERIMENT_TYPE)
             for beta, alpha, lambda_r, t_max, cov_init in param_grid
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
            beta, alpha, lambda_r, t_max, cov_init, mc_idx, _ = futures[fut]
            try:
                beta_, alpha_, lambda_r_, t_max_, cov_init_, traj = fut.result()
                raw.setdefault((beta_, alpha_, lambda_r_, t_max_, cov_init_), []).append(traj)
            except Exception as exc:
                logging.warning(f"  [{done}/{len(tasks)}] Skipped "
                                f"β={beta} α={alpha} λ_r={lambda_r} t_max={t_max} "
                                f"cov_init={cov_init} mc={mc_idx}: {exc}")
                continue
            logging.info(f"  [{done}/{len(tasks)}]  β={beta}  α={alpha}  λ_r={lambda_r}  "
                         f"t_max={t_max}  cov_init={cov_init}  mc={mc_idx}  "
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
        beta, alpha, lambda_r, t_max, cov_init = params
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
            f"GRLS ({EXPERIMENT_TYPE})  β={beta}  α={alpha}  λ_r={lambda_r}  "
            f"t_max={t_max}  cov_init={cov_init}\n"
            f"avg MSE={res['avg_mse']:.4f}   std={res['avg_std']:.4f}   "
            f"score={res['score']:.4f}   gap={res['gap_a']:.4f}   "
            f"({len(res['trajectories'])} MC runs)"
        )
        ax.legend(fontsize=9, loc="upper right")
        plt.tight_layout()
        fname = (f"traj_rank{rank+1:02d}_b{beta}_a{alpha}_"
                 f"lr{lambda_r}_tm{t_max}_ci{cov_init}.png")
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
    metric_labels = {"avg_mse": "Avg MSE", "avg_std": "Avg Std", "score": "Score", "gap_a": f"Gap"}
    metric_label = metric_labels.get(metric_key, metric_key)

    fig, ax = plt.subplots(figsize=(max(6, len(grid_j) * 1.4), max(4, len(grid_i) * 1.1)))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(grid_j))); ax.set_xticklabels(grid_j, fontsize=9)
    ax.set_yticks(range(len(grid_i))); ax.set_yticklabels(grid_i, fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{metric_label} — GRLS ({EXPERIMENT_TYPE})  (min over {other_names})")
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
        header = (f"{'rank':>4}  {'beta':>6}  {'alpha':>7}  {'lambda_r':>9}  "
                  f"{'t_max':>6}  {'cov_init':>9}  {'avg_mse':>10}  "
                  f"{'avg_std':>10}  {'score':>10}  {'gap':>10}\n")
        fh.write(header)
        fh.write("-" * len(header) + "\n")
        for rank, (params, res) in enumerate(ranked):
            beta, alpha, lambda_r, t_max, cov_init = params
            fh.write(
                f"{rank+1:>4}  {beta:>6}  {alpha:>7}  {lambda_r:>9}  "
                f"{t_max:>6}  {cov_init:>9}  {res['avg_mse']:>10.4f}  "
                f"{res['avg_std']:>10.4f}  {res['score']:>10.4f}  {res['gap_a']:>10.4f}\n"
            )
    logging.info(f"Saved summary → {summary_path}")


# ─── vs_poly_order experiment ────────────────────────────────────────────────

def _run_mc_task_vs_poly_order(args):
    """Worker: one MC run for a given (combo, p) pair."""
    beta, alpha, lambda_r, t_max, cov_init, p, _mc_idx = args
    base_cfg = _CFGS["vs_poly_order"]
    cfg_run = copy.deepcopy(base_cfg)
    cfg_run["beta"]                  = beta
    cfg_run["alpha"]                 = alpha
    cfg_run["lambda_r"]              = lambda_r
    cfg_run["t_max"]                 = t_max
    cfg_run["cov_init"]              = cov_init
    cfg_run["stateInit_weight_grls"] = INIT_WEIGHT

    poly_coefficients = np.geomspace(1, 2 ** p, num=p + 1)
    poly_coefficients = 1 / poly_coefficients
    cfg_run["poly_coefficients"] = poly_coefficients

    pos, q_meas, y_meas, conn, _ = get_trajectory(
        cfg_run["trajectory_time"], cfg_run["F"], cfg_run["B"],
        cfg_run["C_w_sqrt"], cfg_run["C_u_sqrt"], cfg_run["n"],
        cfg_run["k"], cfg_run["poly_coefficients"],
        cfg_run["new_edge_weight"], cfg_run["num_edges_stateinit"],
        cfg_run["delta_n"],
    )

    filt = METHOD_REGISTRY[METHOD_TO_OPT](cfg_run)
    mse, *_ = one_method_evaluation(filt, q_meas, y_meas, pos, conn)
    return beta, alpha, lambda_r, t_max, cov_init, p, mse.ravel()   # (T,)


def _run_grid_vs_poly_order(param_grid, p_list, n_workers):
    tasks = [(beta, alpha, lambda_r, t_max, cov_init, int(p), mc_idx)
             for beta, alpha, lambda_r, t_max, cov_init in param_grid
             for p in p_list
             for mc_idx in range(N_MC_PER_COMBO)]

    logging.info(f"[vs_poly_order]  Combos: {len(param_grid)}  p pts: {len(p_list)}  "
                 f"MC/point: {N_MC_PER_COMBO}  Tasks: {len(tasks)}  Workers: {n_workers}")

    raw: dict = {}
    t_wall = time.time()
    done   = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_mc_task_vs_poly_order, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            beta, alpha, lambda_r, t_max, cov_init, p, mc_idx = futures[fut]
            try:
                beta_, alpha_, lambda_r_, t_max_, cov_init_, p_, traj = fut.result()
                raw.setdefault((beta_, alpha_, lambda_r_, t_max_, cov_init_, p_), []).append(traj)
            except Exception as exc:
                logging.warning(f"  [{done}/{len(tasks)}] Skipped "
                                f"β={beta} α={alpha} λ_r={lambda_r} t_max={t_max} "
                                f"cov_init={cov_init} p={p} mc={mc_idx}: {exc}")
                continue
            logging.info(f"  [{done}/{len(tasks)}]  β={beta}  α={alpha}  λ_r={lambda_r}  "
                         f"t_max={t_max}  cov_init={cov_init}  p={p}  mc={mc_idx}  "
                         f"({time.time()-t_wall:.1f}s)")

    logging.info(f"All tasks done in {time.time()-t_wall:.1f}s")

    results_grid = {}
    for combo in param_grid:
        beta, alpha, lambda_r, t_max, cov_init = combo
        mse_per_p = []
        std_per_p = []
        for p in p_list:
            key = (beta, alpha, lambda_r, t_max, cov_init, int(p))
            trajs = raw.get(key, [])
            if trajs:
                per_run = [t.mean() for t in trajs]
                mse_per_p.append(float(np.mean(per_run)))
                std_per_p.append(float(np.std(per_run)))
            else:
                mse_per_p.append(float("nan"))
                std_per_p.append(float("nan"))
        mse_curve = np.array(mse_per_p)
        std_curve = np.array(std_per_p)
        avg_mse   = float(np.nanmean(mse_curve))
        avg_std   = float(np.nanmean(std_curve))
        score     = avg_mse + VARIANCE_WEIGHT * avg_std
        results_grid[combo] = dict(
            avg_mse=avg_mse,
            avg_std=avg_std,
            score=score,
            gap_a=float("nan"),
            mse_curve=mse_curve,
            std_curve=std_curve,
            raw=raw,
        )

    return results_grid


def _plot_vs_poly_order(results_grid, p_list):
    ranked = sorted(results_grid.items(), key=lambda kv: kv[1]["score"])
    p_arr  = np.array(p_list)

    for rank, (params, res) in enumerate(ranked[:N_PLOT_BEST]):
        beta, alpha, lambda_r, t_max, cov_init = params
        mse_curve = res["mse_curve"]
        std_curve = res["std_curve"]

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(p_arr, mse_curve, color="steelblue", linewidth=2,
                marker="o", label="mean MSE")
        ax.fill_between(p_arr, mse_curve - std_curve, mse_curve + std_curve,
                        alpha=0.25, color="steelblue", label="±1 std")
        ax.set_xlabel("Polynomial Order")
        ax.set_ylabel("MSE (mean over time)")
        ax.set_title(
            f"GRLS (vs poly order)  β={beta}  α={alpha}  λ_r={lambda_r}  "
            f"t_max={t_max}  cov_init={cov_init}\n"
            f"avg MSE={res['avg_mse']:.4f}   std={res['avg_std']:.4f}   "
            f"score={res['score']:.4f}   ({N_MC_PER_COMBO} MC runs/point)"
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True)
        plt.tight_layout()
        fname = (f"curve_rank{rank+1:02d}_b{beta}_a{alpha}_"
                 f"lr{lambda_r}_tm{t_max}_ci{cov_init}.png")
        plt.savefig(os.path.join(SAVE_DIR, fname), dpi=100)
        plt.close(fig)
        logging.info(f"  Saved → {fname}")

    return ranked


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os as _os
    for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        _os.environ[_var] = "1"

    param_grid  = list(itertools.product(BETA_GRID, ALPHA_GRID, LAMBDA_R_GRID, T_MAX_GRID, COV_INIT_GRID))
    n_workers   = pick_worker_count()
    param_names = ["beta", "alpha", "lambda_r", "t_max", "cov_init"]
    all_grids   = [BETA_GRID, ALPHA_GRID, LAMBDA_R_GRID, T_MAX_GRID, COV_INIT_GRID]

    # ── Run grid search ────────────────────────────────────────────────────────
    if EXPERIMENT_TYPE == "vs_poly_order":
        p_list       = list(_CFGS["vs_poly_order"]["p_list"])
        results_grid = _run_grid_vs_poly_order(param_grid, p_list, n_workers)
        ranked       = _plot_vs_poly_order(results_grid, p_list)
    else:
        # "linear", "nonlinear_case1", "nonlinear_case2", "vs_poly_order_p0"
        # all use the standard vs-time path
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
    print(f"\n══════════════ GRLS Synthetic ({EXPERIMENT_TYPE}) — Top 5 by score ══════════════")
    for rank, (params, res) in enumerate(ranked[:5]):
        beta, alpha, lambda_r, t_max, cov_init = params
        print(f"  #{rank+1}  β={beta:<5}  α={alpha:<6}  λ_r={lambda_r:<6}  "
              f"t_max={t_max:<4}  cov_init={cov_init:<5}  →  "
              f"avg_mse={res['avg_mse']:.4f}  std={res['avg_std']:.4f}  score={res['score']:.4f}")
    print(f"\nAll results saved to {SAVE_DIR}/")