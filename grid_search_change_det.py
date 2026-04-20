"""
grid_search_change_det.py
-------------------------
Grid search over lambda_1 and lambda_2 for ChangeDetectionMethod on IEEE57 data.

For each (lambda_1, lambda_2) combination:
  - Runs the method on all (or a capped number of) IEEE57 MC datasets
  - Saves a trajectory plot (mean ± std MSE) with the evaluation window shaded
  - Computes two summary metrics:
      avg_mse  : mean MSE across time and MC runs
      gap_A    : mean (max - min) of MSE in window WINDOW_A

  gap_A captures how much the error fluctuates inside the window.
  A large gap means the method reacts strongly (good if MSE drops after a change).
  A small gap means the method is stable (good if MSE stays low throughout).

Outputs (saved to SAVE_DIR):
  traj_l1_<v>_l2_<v>.png   — per-combo trajectory plot
  heatmap_avg_mse.png       — grid heat-map of avg MSE
  heatmap_gap_a.png         — grid heat-map of gap in window A
  grid_results.pkl          — raw results dict keyed by (lambda_1, lambda_2)
"""

import copy
import glob
import json
import logging
import os
import pickle
import time

import matplotlib
matplotlib.use('Agg')  # non-interactive — safe for saving without a display
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from util_func import (
    build_L, vector2diag, extract_x_to_match_B_from_L,
    one_method_evaluation,
)
from constants import METHOD_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── Grid search configuration ───────────────────────────────────────────────
INIT_WEIGHT_GRID = [0.0, 1.0]
LAMBDA_1_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]#0.1, 1, 10, 100,     # L1 / sparsity penalty
LAMBDA_2_GRID = [0, 1e-8, 1e-6]#, 100]       # nuclear-norm penalty

T_MAX = 248             # only run on the first T_MAX time steps of each trajectory
MSE_THRESHOLD = 1e2     # abort a run early if MSE exceeds this

WINDOW_A = (0, 100)     # [start, end)  — evaluation window

DATASET_PATTERN = "Power_data/ieee57_dataset*"
N_DATASETS_MAX  = 2   # set to e.g. 5 to limit MC runs for a quick test; None = all

SAVE_DIR = "Power_data/grid_search_change_det_win2"
os.makedirs(SAVE_DIR, exist_ok=True)

v_degree = False   # keep in sync with power_system_tracking.py

# ─── One-time setup from reference dataset ────────────────────────────────────
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
_full_weights = extract_x_to_match_B_from_L(_L_ref, _B)

_edge_to_col = {e: i for i, e in enumerate(nx.complete_graph(_n).edges())}
_M_complete  = len(_edge_to_col)

# Base config — mirrors the cfg_power_graph block in power_system_tracking.py
_cfg_base = {
    "n":               _metadata["N_buses"],
    "new_edge_weight": float(np.mean(_full_weights[_full_weights > 0])),
    "trajectory_time": np.arange(0, _metadata["T"]),
    "thr1": 2, "thr2": 2, "mu": 1,
    "lambda_1": 100, "lambda_2": 1,         # overwritten per combo
    "delta_n":   _metadata["n_varying_edges"],
    "sigma_v": 1, "sigma_w": 0.1, "sigma_x": 5, "sigma_x_miss": 5,
}
_cfg_base.update({
    "B":                  _B,
    "k":                  _metadata["change_period"],
    "num_edges_stateinit": int(3 * _cfg_base["n"]),
    "window_len":          int(_cfg_base["n"] / 1),
})
_cfg_base.update({
    "m":        _cfg_base["B"].shape[1],
    "C_w_sqrt": np.dot(_cfg_base["sigma_w"], np.eye(_cfg_base["n"])),
})
_cfg_base.update({
    "F":                np.eye(_cfg_base["m"]),
    "C_w":              _cfg_base["C_w_sqrt"] @ _cfg_base["C_w_sqrt"],
    "C_u_sqrt":         np.dot(_cfg_base["sigma_v"], np.eye(_cfg_base["m"])),
    "C_x_missmatch":    np.dot(_cfg_base["sigma_x_miss"] ** 2, np.eye(_cfg_base["m"])),
    "stateInit_missmatch": _cfg_base["new_edge_weight"] * np.ones(
        _cfg_base["m"]).reshape([_cfg_base["m"], 1]),
})
_cfg_base.update({"C_u": _cfg_base["C_u_sqrt"] @ _cfg_base["C_u_sqrt"]})
_cfg_base.update({"poly_coefficients": np.array([0, 1.0])})


# ─── Core helpers ─────────────────────────────────────────────────────────────

def _run_dataset(ds_folder, lambda_1, lambda_2, init_weight):
    """Run ChangeDetectionMethod on one dataset; return MSE trajectory (T,)."""
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

    pos_mc                  = (conn_mc.T * _full_weights).T
    updated_connections_mc  = [np.sort(np.where(row > 0)[0]) for row in conn_mc]
    # stateInit_mc            =     conn_mc[0, :].reshape(-1, 1) * _full_weights +np.random.normal(0, 5, size=_full_weights.shape)
    #_cfg_base["new_edge_weight"] * conn_mc[0, :].reshape(-1, 1)
    cfg_run = copy.deepcopy(_cfg_base)
    # cfg_run["stateInit"]          = stateInit_mc
    # cfg_run["C_x"]                = np.dot(cfg_run["sigma_x"] ** 2, vector2diag(stateInit_mc))
    cfg_run["stateInit_missmatch"] = init_weight * np.ones(
        cfg_run["m"]).reshape([cfg_run["m"], 1])
    cfg_run["lambda_1"] = lambda_1
    cfg_run["lambda_2"] = lambda_2

    filt = METHOD_REGISTRY["change-det"](cfg_run)
    mse, *_ = one_method_evaluation(
        filt, va_mc, P_load_mc, pos_mc, updated_connections_mc, mse_threshold=MSE_THRESHOLD
    )
    return mse.ravel()   # shape (T,)


def _run_dataset_task(args):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable on Windows)."""
    ds_folder, lambda_1, lambda_2, init_weight1 = args
    traj = _run_dataset(ds_folder, lambda_1, lambda_2, init_weight1)
    return lambda_1, lambda_2, init_weight1, traj


def _gap_metric(trajectories, window):
    """
    Mean across MC runs of (max - min) of MSE inside *window* = (start, end).
    Captures how much the error fluctuates within the window.
    """
    start, end = window
    values = [t[start:end].max() - t[start:end].min()
              for t in trajectories if len(t) > start]
    return float(np.mean(values)) if values else float("nan")


def _plot_trajectory(trajectories, lambda_1, lambda_2, init_weight_0, avg_mse, gap_a, save_path):
    traj_mat  = np.stack(trajectories)          # (R, T)
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
               label=f"Window [{WINDOW_A[0]},{wa_end})  gap={gap_a:.4f}")

    ax.set_xlabel("Time step")
    ax.set_ylabel("MSE")
    ax.set_title(
        f"ChangeDetection  λ₁={lambda_1}  λ₂={lambda_2} init_weight={init_weight_0} \n"
        f"avg MSE={avg_mse:.4f}   gap={gap_a:.4f}   ({len(trajectories)} MC runs)"
    )
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    logging.info(f"  Saved trajectory plot → {save_path}")


def _save_heatmap(results_grid, metric_key, metric_label, l1_vals, l2_vals):
    mat = np.full((len(l1_vals), len(l2_vals)), np.nan)
    for i, l1 in enumerate(l1_vals):
        for j, l2 in enumerate(l2_vals):
            candidates = [res[metric_key] for key, res in results_grid.items()
                          if key[0] == l1 and key[1] == l2
                          and not np.isnan(res[metric_key])]
            if candidates:
                mat[i, j] = min(candidates)

    fig, ax = plt.subplots(figsize=(max(6, len(l2_vals) * 1.3), max(4, len(l1_vals) * 1.0)))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(l2_vals))); ax.set_xticklabels(l2_vals)
    ax.set_yticks(range(len(l1_vals))); ax.set_yticklabels(l1_vals)
    ax.set_xlabel("lambda_2 (nuclear-norm)")
    ax.set_ylabel("lambda_1 (L1 / sparsity)")
    ax.set_title(f"{metric_label} — IEEE57 grid search")
    plt.colorbar(im, ax=ax)

    for i in range(len(l1_vals)):
        for j in range(len(l2_vals)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="white")

    # Mark the best cell
    valid = ~np.isnan(mat)
    if valid.any():
        best_idx = np.unravel_index(np.nanargmin(mat), mat.shape)
        ax.add_patch(plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=2.5, label="best"
        ))
        ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, f"heatmap_{metric_key}.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    logging.info(f"  Saved heatmap → {path}")


# ─── Grid search ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os as _os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from util_func import pick_worker_count

    # Prevent BLAS/MKL thread oversubscription inside workers
    for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        _os.environ[_var] = "1"

    dataset_dirs = sorted(glob.glob(DATASET_PATTERN))
    dataset_dirs = [d.replace("\\", "/") for d in dataset_dirs]
    if N_DATASETS_MAX is not None:
        dataset_dirs = dataset_dirs[:N_DATASETS_MAX]

    n_combos  = len(INIT_WEIGHT_GRID) * len(LAMBDA_1_GRID) * len(LAMBDA_2_GRID)
    n_tasks   = n_combos * len(dataset_dirs)
    n_workers = pick_worker_count()
    logging.info(f"Datasets: {len(dataset_dirs)}  |  "
                 f"Grid: {len(INIT_WEIGHT_GRID)} × {len(LAMBDA_1_GRID)} × {len(LAMBDA_2_GRID)} = {n_combos} combos  |  "
                 f"{n_tasks} total tasks  |  {n_workers} workers")

    # Build all (ds, lambda_1, lambda_2) tasks
    tasks = [
        (ds, l1, l2, init_weight1)
        for init_weight1 in INIT_WEIGHT_GRID
        for l1 in LAMBDA_1_GRID
        for l2 in LAMBDA_2_GRID
        for ds in dataset_dirs
    ]

    # Collect raw trajectories grouped by (lambda_1, lambda_2)
    raw: dict = {}   # (lambda_1, lambda_2) -> list of trajectories
    t_wall = time.time()
    done   = 0

    for t in tasks:
        done += 1
        ds, l1, l2, init_weight1 = t
        try:
            _, _, _, traj = _run_dataset_task(t)
            raw.setdefault((l1, l2, init_weight1), []).append(traj)
        except Exception as exc:
            logging.warning(f"  [{done}/{n_tasks}] Skipped {ds} λ₁={l1} λ₂={l2} init_weight={init_weight1}: {exc}")
            continue
        logging.info(f"  [{done}/{n_tasks}] done  λ₁={l1}  λ₂={l2} init_weight={init_weight1} ({time.time()-t_wall:.1f}s)")

    # with ProcessPoolExecutor(max_workers=n_workers) as pool:
    #     futures = {pool.submit(_run_dataset_task, t): t for t in tasks}
    #     for fut in as_completed(futures):
    #         done += 1
    #         ds, l1, l2 = futures[fut]
    #         try:
    #             _, _, traj = fut.result()
    #             raw.setdefault((l1, l2), []).append(traj)
    #         except Exception as exc:
    #             logging.warning(f"  [{done}/{n_tasks}] Skipped {ds} λ₁={l1} λ₂={l2}: {exc}")
    #             continue
    #         logging.info(f"  [{done}/{n_tasks}] done  λ₁={l1}  λ₂={l2}  ({time.time() - t_wall:.1f}s)")

    logging.info(f"All tasks done in {time.time()-t_wall:.1f}s")

    # Compute metrics and plots per combo
    results_grid = {}
    for (lambda_1, lambda_2, init_weight_0), trajectories in raw.items():
        avg_mse = float(np.mean([t.mean() for t in trajectories]))
        gap_a   = _gap_metric(trajectories, WINDOW_A)

        results_grid[(lambda_1, lambda_2, init_weight_0)] = dict(
            avg_mse=avg_mse, gap_a=gap_a,
            trajectories=trajectories,
        )
        logging.info(f"  λ₁={lambda_1}  λ₂={lambda_2} init_weight={init_weight_0} avg_mse={avg_mse:.4f}  gap={gap_a:.4f}")

        plot_name = f"traj_l1_{lambda_1}_l2_{lambda_2}_init_weight_{init_weight_0}.png"
        _plot_trajectory(trajectories, lambda_1, lambda_2, init_weight_0,
                         avg_mse, gap_a,
                         os.path.join(SAVE_DIR, plot_name))

    logging.info(f"Grid search done in {time.time() - t_wall:.1f}s")

    # ── Save raw results ───────────────────────────────────────────────────────
    pkl_path = os.path.join(SAVE_DIR, "grid_results.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(results_grid, fh)
    logging.info(f"Saved raw results → {pkl_path}")

    # ── Heatmaps ───────────────────────────────────────────────────────────────
    l1_vals = sorted(set(k[0] for k in results_grid))
    l2_vals = sorted(set(k[1] for k in results_grid))

    _save_heatmap(results_grid, "avg_mse", "Avg MSE",                           l1_vals, l2_vals)
    _save_heatmap(results_grid, "gap_a",   f"Gap [{WINDOW_A[0]},{WINDOW_A[1]})", l1_vals, l2_vals)

    # ── Print best parameters ──────────────────────────────────────────────────
    best_mse   = min(results_grid, key=lambda k: results_grid[k]["avg_mse"])
    best_gap_a = min(results_grid, key=lambda k: results_grid[k]["gap_a"])

    print("\n══════════════ Grid Search Summary ══════════════")
    print(f"Best avg MSE  : λ₁={best_mse[0]:<6}  λ₂={best_mse[1]:<6}  init_weight={best_mse[2]:<6}"
          f"  →  {results_grid[best_mse]['avg_mse']:.4f}")
    print(f"Best gap      : λ₁={best_gap_a[0]:<6}  λ₂={best_gap_a[1]:<6}  init_weight={best_gap_a[2]:<6}"
          f"  →  {results_grid[best_gap_a]['gap_a']:.4f}")
    print(f"All results saved to {SAVE_DIR}/")