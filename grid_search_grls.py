"""
grid_search_grls.py
-------------------
Grid search over beta, alpha, lambda_r, and t_max for GRLSSimple
on IEEE57 power data.

Metrics per combo:
  avg_mse  — mean MSE across time steps and MC runs
  gap      — mean (max - min) of MSE inside WINDOW_A

Outputs (saved to SAVE_DIR):
  traj_rank<N>_*.png                  — top N_PLOT_BEST trajectory plots
  heatmap_<metric>_<p1>_vs_<p2>.png  — 2D heat-maps for every parameter pair
  grid_results.pkl                    — raw results dict keyed by (beta, alpha, lambda_r, t_max)
  summary.txt                         — all combos ranked by avg_mse
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
    one_method_evaluation, pick_worker_count, build_L,
)
from constants import METHOD_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

METHOD_TO_OPT = "grls"


def load_cfg():
    # ─── Grid search configuration ────────────────────────────────────────────────
    BETA_GRID = [0.8, 0.9, 0.95]          # RLS forgetting factor
    ALPHA_GRID = [0.001, 0.01, 0.1]         # gradient-descent step size
    LAMBDA_R_GRID = [0.1, 1.0]        # L1 sparsity on vec(R)
    T_MAX_GRID = [1, 5, 10, 20.0]                 # gradient-descent iterations per step
    INIT_WEIGHT_GRID = [10.0]     # edge weight for full-graph Laplacian init
    COV_INIT_GRID = [10.0, 20.0]

    MSE_THRESHOLD = 1e2                      # abort a run early if MSE exceeds this

    T_MAX_DATA = 249
    WINDOW_A = (0, 100)
    N_PLOT_BEST = 10

    DATASET_PATTERN = "Power_data/ieee57_dataset*"
    N_DATASETS_MAX = 4  # None = all

    SAVE_DIR = "Power_data/grid_search_" + METHOD_TO_OPT
    os.makedirs(SAVE_DIR, exist_ok=True)

    v_degree = False

    # ─── One-time setup ───────────────────────────────────────────────────────────
    _ref_folder = "Power_data/ieee57_dataset0"
    _line_ids = np.load(f"{_ref_folder}/line_ids.npy")
    _Y_mat = np.load(f"{_ref_folder}/Y_mat.npy")
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
    _mean_edge_weight = float(np.mean(_full_weights[_full_weights > 0]))

    _edge_to_col = {e: i for i, e in enumerate(nx.complete_graph(_n).edges())}
    _M_complete = len(_edge_to_col)

    _cfg_base = {
        "n": _metadata["N_buses"],
        "B": _B,
        "k": _metadata["change_period"],
        "poly_coefficients": np.array([0, 1.0]),
    }
    _cfg_base.update({
        "m": _cfg_base["B"].shape[1],
        "new_edge_weight": _mean_edge_weight,
    })
    return {
        "cfg_base": _cfg_base,
        "BETA_GRID": BETA_GRID,
        "ALPHA_GRID": ALPHA_GRID,
        "LAMBDA_R_GRID": LAMBDA_R_GRID,
        "T_MAX_GRID": T_MAX_GRID,
        "INIT_WEIGHT_GRID": INIT_WEIGHT_GRID,
        "COV_INIT_GRID": COV_INIT_GRID,
        "MSE_THRESHOLD": MSE_THRESHOLD,
        "T_MAX_DATA": T_MAX_DATA,
        "WINDOW_A": WINDOW_A,
        "N_PLOT_BEST": N_PLOT_BEST,
        "DATASET_PATTERN": DATASET_PATTERN,
        "N_DATASETS_MAX": N_DATASETS_MAX,
        "SAVE_DIR": SAVE_DIR,
        "v_degree": v_degree,
        "line_ids": _line_ids,
        "full_weights": _full_weights,
        "edge_to_col": _edge_to_col,
        "M_complete": _M_complete,
    }


# ─── Worker ───────────────────────────────────────────────────────────────────

def _run_dataset(ds_folder, beta, alpha, lambda_r, t_max, init_weight, cov_init, cfg):
    """Run GRLSSimple on one dataset; return MSE trajectory (T,)."""
    ds_folder = ds_folder.replace("\\", "/")
    bv = np.load(f"{ds_folder}/bus_voltages.npy")
    bl = np.load(f"{ds_folder}/bus_loads.npy")
    lt = np.load(f"{ds_folder}/line_topology.npy")

    T_MAX_DATA = cfg["T_MAX_DATA"]
    v_degree = cfg["v_degree"]
    line_ids = cfg["line_ids"]
    full_weights = cfg["full_weights"]
    edge_to_col = cfg["edge_to_col"]
    M_complete = cfg["M_complete"]
    cfg_base = cfg["cfg_base"]
    MSE_THRESHOLD = cfg["MSE_THRESHOLD"]

    va_mc = bv[:T_MAX_DATA, :, 1]
    if v_degree:
        va_mc = np.deg2rad(va_mc)
    P_load_mc = bl[:T_MAX_DATA, :, 0]
    lt = lt[:T_MAX_DATA]

    conn_mc = np.zeros((lt.shape[0], M_complete))
    for k in range(line_ids.shape[0]):
        u, v = int(line_ids[k, 0]), int(line_ids[k, 1])
        conn_mc[:, edge_to_col[(min(u, v), max(u, v))]] = lt[:, k]

    pos_mc = (conn_mc.T * full_weights).T
    updated_connections_mc = [np.sort(np.where(row > 0)[0]) for row in conn_mc]

    cfg_run = copy.deepcopy(cfg_base)
    cfg_run["beta"] = beta
    cfg_run["alpha"] = alpha
    cfg_run["lambda_r"] = lambda_r
    cfg_run["t_max"] = t_max
    cfg_run["cov_init"] = cov_init
    cfg_run["stateInit_weight_grls"] = init_weight

    filt = METHOD_REGISTRY["grls"](cfg_run)
    mse, nmse, f1, eier, *_ = one_method_evaluation(
        filt, va_mc, P_load_mc, pos_mc, updated_connections_mc,
        mse_threshold=MSE_THRESHOLD,
    )
    return mse.ravel(), eier.ravel()  # (T,)


def _run_dataset_task(args):
    """Top-level wrapper — required for pickling on Windows spawn."""
    ds_folder, beta, alpha, lambda_r, t_max, init_weight, cov_init, cfg = args
    traj_mse, traj_eier = _run_dataset(ds_folder, beta, alpha, lambda_r, t_max, init_weight, cov_init, cfg)
    return beta, alpha, lambda_r, t_max, init_weight, cov_init, traj_mse, traj_eier


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _gap_metric(trajectories, window):
    start, end = window
    values = [t[start:end].max() - t[start:end].min()
              for t in trajectories if len(t) > start]
    return float(np.mean(values)) if values else float("nan")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_trajectory(trajectories, params, avg_mse, avg_eier, combined, gap, save_path, window_a):
    beta, alpha, lambda_r, t_max, init_weight, cov_init = params
    traj_mat = np.stack(trajectories)
    mean_traj = traj_mat.mean(axis=0)
    std_traj = traj_mat.std(axis=0)
    T = mean_traj.shape[0]
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, mean_traj, color="steelblue", linewidth=2, label="mean MSE")
    ax.fill_between(t, mean_traj - std_traj, mean_traj + std_traj,
                    alpha=0.25, color="steelblue", label="±1 std")

    wa_end = min(window_a[1], T)
    ax.axvspan(window_a[0], wa_end, alpha=0.10, color="green",
               label=f"Window [{window_a[0]},{wa_end})  gap={gap:.4f}")

    ax.set_xlabel("Time step")
    ax.set_ylabel("MSE")
    ax.set_title(
        f"GRLS  β={beta}  α={alpha}  λ_r={lambda_r}  t_max={t_max}  w_init={init_weight}  cov_init={cov_init}\n"
        f"avg MSE={avg_mse:.4f}   avg EIER={avg_eier:.4f}   combined={combined:.4f}   gap={gap:.4f}   ({len(trajectories)} MC runs)"
    )
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    logging.info(f"  Saved → {save_path}")


def _save_heatmap_2d(results_grid, metric_key, param_i, param_j,
                     grid_i, grid_j, param_names, window_a, save_dir):
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
    metric_labels = {
        "avg_mse": "Avg MSE",
        "avg_eier": "Avg EIER",
        "combined": "Combined (MSE × (1+EIER))",
        "gap_a": f"Gap [{window_a[0]},{window_a[1]})",
    }
    metric_label = metric_labels.get(metric_key, metric_key)

    fig, ax = plt.subplots(figsize=(max(6, len(grid_j) * 1.4), max(4, len(grid_i) * 1.1)))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(grid_j)));
    ax.set_xticklabels(grid_j, fontsize=9)
    ax.set_yticks(range(len(grid_i)));
    ax.set_yticklabels(grid_i, fontsize=9)
    ax.set_xlabel(xlabel);
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric_label} — GRLS  (min over {other_names})")
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
    path = os.path.join(save_dir, f"heatmap_{metric_key}_{ylabel}_vs_{xlabel}.png")
    plt.savefig(path, dpi=110)
    plt.close(fig)
    logging.info(f"  Saved heatmap → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os as _os

    for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        _os.environ[_var] = "1"

    cfg = load_cfg()
    BETA_GRID = cfg["BETA_GRID"]
    ALPHA_GRID = cfg["ALPHA_GRID"]
    LAMBDA_R_GRID = cfg["LAMBDA_R_GRID"]
    T_MAX_GRID = cfg["T_MAX_GRID"]
    INIT_WEIGHT_GRID = cfg["INIT_WEIGHT_GRID"]
    COV_INIT_GRID = cfg["COV_INIT_GRID"]
    WINDOW_A = cfg["WINDOW_A"]
    N_PLOT_BEST = cfg["N_PLOT_BEST"]
    DATASET_PATTERN = cfg["DATASET_PATTERN"]
    N_DATASETS_MAX = cfg["N_DATASETS_MAX"]
    SAVE_DIR = cfg["SAVE_DIR"]

    # dataset_dirs = sorted(glob.glob(DATASET_PATTERN))
    # dataset_dirs = [d.replace("\\", "/") for d in dataset_dirs]
    # if N_DATASETS_MAX is not None:
    #     dataset_dirs = dataset_dirs[:N_DATASETS_MAX]
    dataset_dirs = ["Power_data\\ieee57_dataset39","Power_data\\ieee57_dataset65","Power_data\\ieee57_dataset69","Power_data\\ieee57_dataset95"]
    param_grid = list(itertools.product(BETA_GRID, ALPHA_GRID, LAMBDA_R_GRID, T_MAX_GRID,
                                        INIT_WEIGHT_GRID, COV_INIT_GRID))
    tasks = [(ds, beta, alpha, lambda_r, t_max, init_weight, cov_init, cfg)
             for beta, alpha, lambda_r, t_max, init_weight, cov_init in param_grid
             for ds in dataset_dirs]

    n_combos = len(param_grid)
    n_tasks = n_combos * len(dataset_dirs)
    n_workers = pick_worker_count()
    logging.info(f"Datasets: {len(dataset_dirs)}  |  Combos: {n_combos}  |  "
                 f"Tasks: {n_tasks}  |  Workers: {n_workers}")

    raw: dict = {}
    t_wall = time.time()
    done = 0

    # for t in tasks:
    #     done += 1
    #     ds, beta, alpha, lambda_r, t_max, init_weight, cov_init, _ = t
    #     try:
    #         beta_, alpha_, lambda_r_, t_max_, init_weight_, cov_init_, traj_mse, traj_eier = _run_dataset_task(t)
    #         entry = raw.setdefault((beta_, alpha_, lambda_r_, t_max_, init_weight_, cov_init_), {"mse": [], "eier": []})
    #         entry["mse"].append(traj_mse)
    #         entry["eier"].append(traj_eier)
    #     except Exception as exc:
    #         logging.warning(f"  [{done}/{n_tasks}] Skipped {ds} "
    #                         f"β={beta} α={alpha} λ_r={lambda_r} t_max={t_max} "
    #                         f"w_init={init_weight} cov_init={cov_init} : {exc}")
    #         continue
    #     logging.info(f"  [{done}/{n_tasks}]  β={beta}  α={alpha}  λ_r={lambda_r}  t_max={t_max}"
    #                  f"  w_init={init_weight}  cov_init={cov_init}"
    #                  f"  ({time.time() - t_wall:.1f}s)")
    #
    # logging.info(f"All tasks done in {time.time() - t_wall:.1f}s")



    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_dataset_task, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            ds, beta, alpha, lambda_r, t_max, init_weight, cov_init, _ = futures[fut]
            try:
                beta_, alpha_, lambda_r_, t_max_, init_weight_, cov_init_, traj_mse, traj_eier = fut.result()
                entry = raw.setdefault((beta_, alpha_, lambda_r_, t_max_, init_weight_, cov_init_), {"mse": [], "eier": []})
                entry["mse"].append(traj_mse)
                entry["eier"].append(traj_eier)
            except Exception as exc:
                logging.warning(f"  [{done}/{n_tasks}] Skipped {ds} "
                                f"β={beta} α={alpha} λ_r={lambda_r} t_max={t_max} "
                                f"w_init={init_weight} cov_init={cov_init} : {exc}")
                continue
            logging.info(f"  [{done}/{n_tasks}]  β={beta}  α={alpha}  λ_r={lambda_r}  t_max={t_max}"
                         f"  w_init={init_weight}  cov_init={cov_init}"
                         f"  ({time.time() - t_wall:.1f}s)")

    logging.info(f"All tasks done in {time.time() - t_wall:.1f}s")



    # ── Metrics ────────────────────────────────────────────────────────────────
    results_grid = {}
    for params, entry in raw.items():
        traj_mse = entry["mse"]
        traj_eier = entry["eier"]
        avg_mse = float(np.mean([t.mean() for t in traj_mse]))
        avg_eier = float(np.mean([t.mean() for t in traj_eier]))
        combined = avg_mse * (1 + avg_eier)
        gap = _gap_metric(traj_mse, WINDOW_A)
        results_grid[params] = dict(avg_mse=avg_mse, avg_eier=avg_eier,
                                    combined=combined, gap_a=gap, trajectories=traj_mse)

    # ── Save raw results ───────────────────────────────────────────────────────
    pkl_path = os.path.join(SAVE_DIR, "grid_results.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(results_grid, fh)
    logging.info(f"Saved raw results → {pkl_path}")

    # ── Trajectory plots — top N_PLOT_BEST ────────────────────────────────────
    ranked = sorted(results_grid.items(), key=lambda kv: kv[1]["combined"])
    for rank, (params, res) in enumerate(ranked[:N_PLOT_BEST]):
        beta, alpha, lambda_r, t_max, init_weight, cov_init = params
        fname = f"traj_rank{rank + 1:02d}_b{beta}_a{alpha}_lr{lambda_r}_t{t_max}_w{init_weight}_c{cov_init}.png"
        _plot_trajectory(res["trajectories"], params,
                         res["avg_mse"], res["avg_eier"], res["combined"], res["gap_a"],
                         os.path.join(SAVE_DIR, fname), WINDOW_A)

    # ── 2D heatmaps for every parameter pair ──────────────────────────────────
    param_names = ["beta", "alpha", "lambda_r", "t_max", "init_weight", "cov_init"]
    all_grids = [BETA_GRID, ALPHA_GRID, LAMBDA_R_GRID, T_MAX_GRID, INIT_WEIGHT_GRID, COV_INIT_GRID]
    for (pi, pj) in itertools.combinations(range(6), 2):
        for metric_key in ("avg_mse", "avg_eier", "combined", "gap_a"):
            _save_heatmap_2d(results_grid, metric_key, pi, pj,
                             all_grids[pi], all_grids[pj], param_names,
                             WINDOW_A, SAVE_DIR)

    # ── Summary table ──────────────────────────────────────────────────────────
    summary_path = os.path.join(SAVE_DIR, "summary.txt")
    with open(summary_path, "w") as fh:
        header = (f"{'rank':>4}  {'beta':>6}  {'alpha':>8}  {'lambda_r':>10}  {'t_max':>6}  "
                  f"{'init_weight':>12}  {'cov_init':>10}  {'avg_mse':>12}  {'avg_eier':>10}  "
                  f"{'combined':>12}  {'gap':>10}\n")
        fh.write(header)
        fh.write("-" * len(header) + "\n")
        for rank, (params, res) in enumerate(ranked):
            beta, alpha, lambda_r, t_max, init_weight, cov_init = params
            fh.write(f"{rank + 1:>4}  {beta:>6}  {alpha:>8}  {lambda_r:>10}  {t_max:>6}  "
                     f"{init_weight:>12}  {cov_init:>10}  "
                     f"{res['avg_mse']:>12.4f}  {res['avg_eier']:>10.4f}  "
                     f"{res['combined']:>12.4f}  {res['gap_a']:>10.4f}\n")
    logging.info(f"Saved summary → {summary_path}")

    # ── Print top 5 ────────────────────────────────────────────────────────────
    print("\n══════════════ GRLS Grid Search — Top 5 by combined score ══════════════")
    for rank, (params, res) in enumerate(ranked[:5]):
        beta, alpha, lambda_r, t_max, init_weight, cov_init = params
        print(f"  #{rank + 1}  β={beta:<6}  α={alpha:<8}  λ_r={lambda_r:<8}  t_max={t_max:<4}  "
              f"w_init={init_weight:<6}  cov_init={cov_init:<6}"
              f"  →  avg_mse={res['avg_mse']:.4f}   avg_eier={res['avg_eier']:.4f}   "
              f"combined={res['combined']:.4f}   gap={res['gap_a']:.4f}")
    print(f"\nAll results saved to {SAVE_DIR}/")