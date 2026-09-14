"""
grid_search_gsp_ekf.py
----------------------
Grid search over sigma_v (C_u), sigma_w (C_w), sigma_x_miss (C_x_missmatch),
and thr1 (hard-threshold / "mu") for gsp-ekf (sparseKalmanFilter) on IEEE57 data.

Metrics per combo:
  avg_mse  — mean MSE across time steps and MC runs
  gap      — mean (max - min) of MSE inside WINDOW_A

Outputs (saved to SAVE_DIR):
  traj_rank<N>_*.png                   — top N_PLOT_BEST trajectory plots
  heatmap_<metric>_<p1>_vs_<p2>.png   — 2D heat-maps for every parameter pair
  grid_results.pkl                     — raw results dict keyed by (sv, sw, sx, thr)
  summary.txt                          — all combos ranked by avg_mse
"""

import copy
import glob
import itertools
import json
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from util_func import (
    vector2diag, extract_x_to_match_B_from_L,
    one_method_evaluation, pick_worker_count,
)
from constants import METHOD_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

METHOD_TO_OPT = "gsp-ekf"
# ─── Grid search configuration ────────────────────────────────────────────────
SIGMA_V_GRID    = [0.001, 0.01, 0.1, 1.0, 5]     # process noise     → C_u = σ_v² I_m
SIGMA_W_GRID    = [0.001, 0.01, 0.1, 1.0, 5]     # measurement noise → C_w = σ_w² I_n
SIGMA_X_GRID    = [1.0, 5.0, 10.0]            # init uncertainty  → C_x_miss = σ_x² I_m
THR_GRID        = [0.1, 0.5, 1.0, 2.0, 5.0]  # hard threshold (thr1 / "mu")

T_MAX        = 249
WINDOW_A     = (0, 100)
N_PLOT_BEST  = 10

DATASET_PATTERN = "Power_data/ieee57_dataset*"
N_DATASETS_MAX  = 3    # None = all

SAVE_DIR = "Power_data/grid_search_" + METHOD_TO_OPT
os.makedirs(SAVE_DIR, exist_ok=True)

v_degree = False

# ─── One-time setup ───────────────────────────────────────────────────────────
_ref_folder = "Power_data/ieee57_dataset0"
_line_ids   = np.load(f"{_ref_folder}/line_ids.npy")
_Y_mat      = np.load(f"{_ref_folder}/Y_mat.npy")
with open(f"{_ref_folder}/metadata.json") as _fh:
    _metadata = json.load(_fh)

_n = _metadata["N_buses"]
_B = nx.incidence_matrix(
    nx.complete_graph(_n, create_using=None), oriented=True
).todense()
_L_ref = (
    -np.imag(_Y_mat)
    - vector2diag(-np.imag(_Y_mat) @ np.ones([_Y_mat.shape[0], 1]))
)
_full_weights    = extract_x_to_match_B_from_L(_L_ref, _B)
_mean_edge_weight = float(np.mean(_full_weights[_full_weights > 0]))

_edge_to_col = {e: i for i, e in enumerate(nx.complete_graph(_n).edges())}
_M_complete  = len(_edge_to_col)

_cfg_base = {
    "n": _metadata["N_buses"],
}
_cfg_base.update({
    "B": _B,
    "k": _metadata["change_period"],
})
_cfg_base.update({
    "m": _cfg_base["B"].shape[1],
    "F": np.eye(_cfg_base["B"].shape[1]),
})
_cfg_base.update({"poly_coefficients": np.array([0, 1.0])})


# ─── Worker ───────────────────────────────────────────────────────────────────

def _run_dataset(ds_folder, sigma_v, sigma_w, sigma_x, thr):
    """Run gsp-ekf on one dataset; return MSE trajectory (T,)."""
    ds_folder = ds_folder.replace("\\", "/")
    bv = np.load(f"{ds_folder}/bus_voltages.npy")
    bl = np.load(f"{ds_folder}/bus_loads.npy")
    lt = np.load(f"{ds_folder}/line_topology.npy")

    va_mc     = bv[:T_MAX, :, 1]
    if v_degree:
        va_mc = np.deg2rad(va_mc)
    P_load_mc = bl[:T_MAX, :, 0]
    lt        = lt[:T_MAX]

    conn_mc = np.zeros((lt.shape[0], _M_complete))
    for k in range(_line_ids.shape[0]):
        u, v = int(_line_ids[k, 0]), int(_line_ids[k, 1])
        conn_mc[:, _edge_to_col[(min(u, v), max(u, v))]] = lt[:, k]

    pos_mc                 = (conn_mc.T * _full_weights).T
    updated_connections_mc = [np.sort(np.where(row > 0)[0]) for row in conn_mc]

    # gsp-ekf uses uniform init (stateInit_missmatch), not topology-aware
    stateInit_miss = _mean_edge_weight * np.ones(_cfg_base["m"]).reshape(-1, 1)

    cfg_run = copy.deepcopy(_cfg_base)
    cfg_run["stateInit_missmatch"] = stateInit_miss
    cfg_run["C_u"]          = (sigma_v ** 2) * np.eye(cfg_run["m"])
    cfg_run["C_w"]          = (sigma_w ** 2) * np.eye(cfg_run["n"])
    cfg_run["C_x_missmatch"] = (sigma_x ** 2) * np.eye(cfg_run["m"])
    cfg_run["thr1"]          = thr

    filt = METHOD_REGISTRY[METHOD_TO_OPT](cfg_run)
    mse, *_ = one_method_evaluation(
        filt, va_mc, P_load_mc, pos_mc, updated_connections_mc, mse_threshold=1e2
    )
    return mse.ravel()   # (T,)


def _run_dataset_task(args):
    """Top-level wrapper — required for pickling on Windows spawn."""
    ds_folder, sigma_v, sigma_w, sigma_x, thr = args
    traj = _run_dataset(ds_folder, sigma_v, sigma_w, sigma_x, thr)
    return sigma_v, sigma_w, sigma_x, thr, traj


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _gap_metric(trajectories, window):
    start, end = window
    values = [t[start:end].max() - t[start:end].min()
              for t in trajectories if len(t) > start]
    return float(np.mean(values)) if values else float("nan")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_trajectory(trajectories, params, avg_mse, gap, save_path):
    sigma_v, sigma_w, sigma_x, thr = params
    traj_mat  = np.stack(trajectories)
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
               label=f"Window [{WINDOW_A[0]},{wa_end})  gap={gap:.4f}")

    ax.set_xlabel("Time step")
    ax.set_ylabel("MSE")
    ax.set_title(
        f"GSP-EKF  σ_v={sigma_v}  σ_w={sigma_w}  σ_x={sigma_x}  thr={thr}\n"
        f"avg MSE={avg_mse:.4f}   gap={gap:.4f}   ({len(trajectories)} MC runs)"
    )
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    logging.info(f"  Saved → {save_path}")


def _save_heatmap_2d(results_grid, metric_key, param_i, param_j,
                     grid_i, grid_j, param_names):
    """2D heatmap for param_i vs param_j; projects over remaining dims by min."""
    other_dims = [d for d in range(len(param_names)) if d not in (param_i, param_j)]
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
    other_names = " & ".join(param_names[d] for d in other_dims)
    metric_label = "Avg MSE" if metric_key == "avg_mse" else f"Gap [{WINDOW_A[0]},{WINDOW_A[1]})"

    fig, ax = plt.subplots(figsize=(max(6, len(grid_j) * 1.4), max(4, len(grid_i) * 1.1)))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(grid_j))); ax.set_xticklabels(grid_j, fontsize=9)
    ax.set_yticks(range(len(grid_i))); ax.set_yticklabels(grid_i, fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{metric_label} — GSP-EKF  (min over {other_names})")
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


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os as _os
    for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        _os.environ[_var] = "1"

    dataset_dirs = sorted(glob.glob(DATASET_PATTERN))
    dataset_dirs = [d.replace("\\", "/") for d in dataset_dirs]
    if N_DATASETS_MAX is not None:
        dataset_dirs = dataset_dirs[:N_DATASETS_MAX]
    param_grid = list(itertools.product(SIGMA_V_GRID, SIGMA_W_GRID, SIGMA_X_GRID, THR_GRID))
    tasks = [(ds, sv, sw, sx, thr)
             for sv, sw, sx, thr in param_grid
             for ds in dataset_dirs]


    n_combos  = len(param_grid)
    n_tasks   = n_combos * len(dataset_dirs)
    n_workers = pick_worker_count()
    logging.info(f"Datasets: {len(dataset_dirs)}  |  Combos: {n_combos}  |  "
                 f"Tasks: {n_tasks}  |  Workers: {n_workers}")



    raw: dict = {}
    t_wall = time.time()
    done   = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_dataset_task, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            ds, sv, sw, sx, thr = futures[fut]
            try:
                sv_, sw_, sx_, thr_, traj = fut.result()
                raw.setdefault((sv_, sw_, sx_, thr_), []).append(traj)
            except Exception as exc:
                logging.warning(f"  [{done}/{n_tasks}] Skipped {ds} "
                                f"σ_v={sv} σ_w={sw} σ_x={sx} thr={thr}: {exc}")
                continue
            logging.info(f"  [{done}/{n_tasks}]  σ_v={sv}  σ_w={sw}  σ_x={sx}  thr={thr}"
                         f"  ({time.time()-t_wall:.1f}s)")

    logging.info(f"All tasks done in {time.time()-t_wall:.1f}s")

    # ── Metrics ────────────────────────────────────────────────────────────────
    results_grid = {}
    for params, trajectories in raw.items():
        avg_mse = float(np.mean([t.mean() for t in trajectories]))
        gap     = _gap_metric(trajectories, WINDOW_A)
        results_grid[params] = dict(avg_mse=avg_mse, gap_a=gap, trajectories=trajectories)

    # ── Save raw results ───────────────────────────────────────────────────────
    pkl_path = os.path.join(SAVE_DIR, "grid_results.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(results_grid, fh)
    logging.info(f"Saved raw results → {pkl_path}")

    # ── Trajectory plots — top N_PLOT_BEST ────────────────────────────────────
    ranked = sorted(results_grid.items(), key=lambda kv: kv[1]["avg_mse"])
    for rank, (params, res) in enumerate(ranked[:N_PLOT_BEST]):
        sv, sw, sx, thr = params
        fname = f"traj_rank{rank+1:02d}_sv{sv}_sw{sw}_sx{sx}_thr{thr}.png"
        _plot_trajectory(res["trajectories"], params,
                         res["avg_mse"], res["gap_a"],
                         os.path.join(SAVE_DIR, fname))

    # ── 2D heatmaps for every parameter pair ──────────────────────────────────
    param_names = ["sigma_v", "sigma_w", "sigma_x", "thr"]
    for (pi, pj) in itertools.combinations(range(4), 2):
        all_grids = [SIGMA_V_GRID, SIGMA_W_GRID, SIGMA_X_GRID, THR_GRID]
        for metric_key in ("avg_mse", "gap_a"):
            _save_heatmap_2d(results_grid, metric_key, pi, pj,
                             all_grids[pi], all_grids[pj], param_names)

    # ── Summary table ──────────────────────────────────────────────────────────
    summary_path = os.path.join(SAVE_DIR, "summary.txt")
    with open(summary_path, "w") as fh:
        header = f"{'rank':>4}  {'sigma_v':>8}  {'sigma_w':>8}  {'sigma_x':>8}  {'thr':>6}  {'avg_mse':>10}  {'gap':>10}\n"
        fh.write(header)
        fh.write("-" * len(header) + "\n")
        for rank, (params, res) in enumerate(ranked):
            sv, sw, sx, thr = params
            fh.write(f"{rank+1:>4}  {sv:>8}  {sw:>8}  {sx:>8}  {thr:>6}  "
                     f"{res['avg_mse']:>10.4f}  {res['gap_a']:>10.4f}\n")
    logging.info(f"Saved summary → {summary_path}")

    # ── Print top 5 ────────────────────────────────────────────────────────────
    print("\n══════════════ GSP-EKF Grid Search — Top 5 by avg MSE ══════════════")
    for rank, (params, res) in enumerate(ranked[:5]):
        sv, sw, sx, thr = params
        print(f"  #{rank+1}  σ_v={sv:<6}  σ_w={sw:<6}  σ_x={sx:<5}  thr={thr:<5}"
              f"  →  avg_mse={res['avg_mse']:.4f}   gap={res['gap_a']:.4f}")
    print(f"\nAll results saved to {SAVE_DIR}/")