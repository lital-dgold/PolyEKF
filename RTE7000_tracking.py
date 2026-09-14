import os

# Must run before numpy (or anything importing it, e.g. util_func) is loaded --
# OpenBLAS/MKL latch their thread-pool size when the library is first loaded,
# not on every call, so setting these afterward (as the __main__ block used to)
# has no effect and lets each worker process spawn multi-threaded BLAS ops that
# oversubscribe the SLURM allocation's CPUs.
for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_var] = "1"

import json
from util_func import build_L, vector2diag, num_possible_edges
import numpy as np
import networkx as nx
from util_func import extract_x_to_match_B_from_L

n_buses = 100 # ToDo: turn to automate
# Matches S_BASE_MVA in Power_data/Generate_data_rte7000.py -- Y_mat (hence L,
# built from -Im(Y_mat)) is per-unit on this base, so L @ theta must be scaled
# by it to compare against MW quantities (P_load, P_gen, ...). Used only by the
# DC-estimation residual diagnostic below -- NOT applied to the L/full_weights
# that feed the actual EKF tracking state, which stay in their existing units.
S_BASE_MVA = 100.0
##### constants #####
T = 'T'
N_BUSES = 'N_buses'
N_LINES = 'N_lines'
NOISE_STD = 'noise_std'
N_VARYING_EDGES = 'n_varying_edges'
VARYING_LINE_IDX = 'varying_line_idx'
CHANGE_PERIOD = 'change_period'
LOAD_VARIABILITY = 'load_variability'
SEED = 'seed'
FAILED_STEPS = 'failed_steps'
ARRAYS = 'arrays'
# Best parameters
cfg_oracle = {
    "sigma_v": 1,
    "sigma_w": 5,
    "sigma_x": 10,
}

cfg_gsp_ekf = {
    "sigma_v": 1,
    "sigma_w": 0.001,
    "sigma_x": 10,
    "thr1": 1,
}

cfg_fast_ekf = {
    "sigma_v": 1,
    "sigma_w": 0.01,
    "sigma_x": 10,
}

cfg_grls = {
    "beta": 0.8,
    "alpha": 0.01,
    "lambda_r": 1.0,
    "t_max": 10,
    "stateInit_weight_grls": 10.0,
    "cov_init": 20.0
}
cfg_change_det = {
    "window_len": n_buses,
    "stateInit_missmatch": np.zeros(num_possible_edges(n_buses)).reshape([num_possible_edges(n_buses), 1]),
    "lambda_1": 1e-3,
    "lambda_2": 0.0,
}


def prepare_cfg():
    folder_name = "Power_data/rte7000_ac_dataset0"
    # Load all arrays
    bus_voltages  = np.load(f"{folder_name}/bus_voltages.npy")   # (T, 57, 2)
    bus_loads     = np.load(f"{folder_name}/bus_loads.npy")       # (T, 57, 2)
    line_topology = np.load(f"{folder_name}/line_topology.npy")   # (T, 186)
    bus_ids       = np.load(f"{folder_name}/bus_ids.npy")
    line_ids      = np.load(f"{folder_name}/line_ids.npy")
    Y_mat         = np.load(f"{folder_name}/Y_mat.npy")
    # bus_generation always exists; bus_boundary_injection only exists for
    # datasets regenerated after the boundary-tie fix -- fall back to zeros
    # (no net injection correction) for older datasets so this doesn't crash.
    bus_generation = np.load(f"{folder_name}/bus_generation.npy")
    try:
        bus_boundary_injection = np.load(f"{folder_name}/bus_boundary_injection.npy")
    except FileNotFoundError:
        print(f"WARNING: {folder_name} has no bus_boundary_injection.npy (not regenerated with "
              f"the boundary-tie fix yet) -- DC-estimation check will be missing that term.")
        bus_boundary_injection = np.zeros((bus_loads.shape[1], 2))
    with open(f"{folder_name}/metadata.json") as json_data:
        metadata = json.load(json_data)
        json_data.close()
    # ── Voltages ──────────────────────────────────────────────────────
    vm = bus_voltages[:, :, 0]  # (T, 57) voltage magnitude in pu
    va = bus_voltages[:, :, 1]

    for t in range(metadata["T"]):
        if np.isnan(np.sum(vm[t, :])):
            print(vm[t, :])
            print(bus_loads[t, :, 0])
    print("Voltage magnitude (pu):")
    print(f"  shape : {vm.shape}")
    print(f"  min   : {vm.min():.4f}")
    print(f"  max   : {vm.max():.4f}")
    print(f"  mean  : {vm.mean():.4f}")

    print("\nVoltage angle (degrees):")
    print(f"  min   : {va.min():.4f}")
    print(f"  max   : {va.max():.4f}")

    # ── Loads ─────────────────────────────────────────────────────────
    P_load = bus_loads[:, :, 0]   # (T, 57) active power MW
    Q_load = bus_loads[:, :, 1]   # (T, 57) reactive power MVAr
    P_gen = bus_generation[:, :, 0]   # (T, 57) real generation + slack ext_grid dispatch, MW
    P_boundary = bus_boundary_injection[:, 0]   # (57,) constant net boundary-tie injection, MW

    print("\nActive load P (MW):")
    print(f"  total system load at t=0 : {P_load[0].sum():.2f} MW")
    print(f"  total system load at t=1 : {P_load[1].sum():.2f} MW")

    # ── Topology ──────────────────────────────────────────────────────
    print("\nTopology:")
    print(f"  lines in service at t=0  : {line_topology[0].sum()}")
    print(f"  lines in service at t=50 : {line_topology[50].sum()}")

    # ── Single bus / single line inspection ──────────────────────────
    bus_idx  = 0   # change to any bus 0..117
    line_idx = 0   # change to any line 0..185

    print(f"\nBus {bus_ids[bus_idx]} voltage over time (first 5 steps):")
    print(f"  vm (pu)  : {vm[:5, bus_idx]}")
    print(f"  va (deg) : {va[:5, bus_idx]}")

    print(f"\nLine {line_ids[line_idx]} (from {line_ids[line_idx, 0]} to {line_ids[line_idx, 1]}):")
    print(f"  topology (first 10): {line_topology[:10, line_idx]}")
    # print(f"  P_flow   (first 10): {P_flow[:10, line_idx].round(2)} MW")

    # metadata["slack_bus"] is the bus NAME (e.g. "WARANP7#49"), not a row index --
    # look it up against bus_ids to get the actual position. (Newer datasets also
    # carry metadata["slack_bus_index"] directly; this lookup works for older
    # already-generated ones too, so it doesn't require regenerating anything.)
    slack_idx = int(np.where(bus_ids == metadata["slack_bus"])[0][0])
    # G = empty_graph(metadata[N_BUSES])
    # G.add_edges_from(line_ids)
    # B = nx.incidence_matrix(G, oriented=True).todense()# nx.incidence_matrix(nx.complete_graph(cfg_N_20_graph["n"], create_using=None), oriented=True).todense()
    B = nx.incidence_matrix(nx.complete_graph(metadata[N_BUSES], create_using=None), oriented=True).todense()
    L = -np.imag(Y_mat) - vector2diag(-np.imag(Y_mat) @ np.ones([Y_mat.shape[0], 1]))

    ####
    # DC-estimation residual check (diagnostic only -- doesn't feed into the
    # EKF tracking state below, L/full_weights there stay unscaled). Verified
    # empirically: S_BASE_MVA * (L @ theta) matches net injection = P_gen +
    # P_boundary - P_load (generation-positive), NOT P_load alone and NOT
    # unscaled -- Y_mat/L are per-unit on S_BASE_MVA, and L's sign convention
    # here (off-diag = +b_ij, diag = -row-sum) is the negative of the standard
    # "P_injected = B' @ theta" DC power-flow Laplacian.
    net_injection = P_gen + P_boundary[None, :] - P_load
    y = net_injection[0, :].T
    q = va[0, :].T.copy()
    # 1. Create the mask to exclude the 69th column (index slack_idx)
    # This is cleaner than using range(57) != slack_idx
    mask = np.delete(np.arange(L.shape[1]), slack_idx)
    L_sub = S_BASE_MVA * L[:, mask]

    # 2. Least squares (lstsq, not a normal-equations inverse: some buses are
    # structurally islanded within this subgraph -- see conversation -- which
    # makes L_sub rank-deficient and inv(L_sub.T @ L_sub) singular).
    DCestimation, *_ = np.linalg.lstsq(L_sub, y, rcond=None)

    print(np.sum(q[mask] - q[slack_idx] - DCestimation))
    residuals = net_injection - S_BASE_MVA * (L @ q)
    z = np.cov(residuals.T)
    ####

    full_weights = extract_x_to_match_B_from_L(L, B)
    z = (build_L(B, full_weights) - L)
    print(np.sum(z))
    # Pad line_topology to complete-graph column ordering, aligned with B.
    # conn[:, i] == 1 iff the edge corresponding to B's i-th column is active.
    _n = metadata[N_BUSES]
    dict_edge_to_col = {e: i for i, e in enumerate(nx.complete_graph(_n).edges())}
    _M_complete = len(dict_edge_to_col)

    cfg_power_graph = {
        "num_iterations": 1,
        "n": metadata[N_BUSES],
        "m": _M_complete,
        "B": B,
        "poly_coefficients": np.array([0, 1.0]),
        "new_edge_weight": np.mean(full_weights[full_weights > 0]),
        "trajectory_time": np.arange(0, metadata[T]),
        "full_weights": full_weights,
        "_edge_to_col": dict_edge_to_col,
        "line_ids": line_ids,
        "slack_idx": slack_idx,
    }

    cfg_power_graph.update({
        "F": np.dot(1, np.eye(cfg_power_graph["m"])),
    })
    return cfg_power_graph


def prepare_cfg_dc(folder_name="Power_data/rte7000_dc_syn_dataset0"):
    """DC counterpart of prepare_cfg(). Datasets from generate_data_new.py have no
    Q and no voltage magnitude (DC power flow doesn't model either) -- 'y' is the
    net real-power injection per bus, saved directly as bus_net_injection_mw.npy
    (== Bbus @ theta * baseMVA, verified interactively to residual ~1e-11 MW), and
    'q' is the bus voltage angle bus_va_rad.npy (radians). full_weights.npy is the
    static per-edge DC susceptance weight, already extracted from the real Bbus the
    same way prepare_cfg() extracts it from Y_mat, so it plugs into build_L /
    extract_x_to_match_B_from_L unchanged -- no Y_mat, no on-the-fly L rebuild needed.
    """
    bus_va_rad = np.load(f"{folder_name}/bus_va_rad.npy")                      # (T, N)
    bus_net_injection_mw = np.load(f"{folder_name}/bus_net_injection_mw.npy")  # (T, N)
    line_topology = np.load(f"{folder_name}/line_topology.npy")                # (T, N_lines)
    bus_ids = np.load(f"{folder_name}/bus_ids.npy")
    line_ids = np.load(f"{folder_name}/line_ids.npy")
    full_weights = np.load(f"{folder_name}/full_weights.npy")                  # (M, 1)
    with open(f"{folder_name}/metadata.json") as json_data:
        metadata = json.load(json_data)

    print("Bus voltage angle (radians):")
    print(f"  shape : {bus_va_rad.shape}")
    print(f"  min   : {bus_va_rad.min():.4f}")
    print(f"  max   : {bus_va_rad.max():.4f}")

    print("\nNet injection P (MW):")
    print(f"  total system net injection at t=0 : {bus_net_injection_mw[0].sum():.2f} MW "
          f"(near 0 expected -- DC has no losses to unbalance generation vs load)")

    print("\nTopology:")
    print(f"  lines in service at t=0  : {line_topology[0].sum()}")

    slack_idx = int(np.where(bus_ids == metadata["slack_bus"])[0][0])
    B = nx.incidence_matrix(nx.complete_graph(metadata[N_BUSES], create_using=None), oriented=True).todense()

    # Same DC-estimation self-consistency check as prepare_cfg(), just built from
    # full_weights directly instead of round-tripping through Y_mat -> L.
    # L = S_BASE_MVA * build_L(B, full_weights)
    # y0 = bus_net_injection_mw[0, :].T
    # q0 = bus_va_rad[0, :].T.copy()
    # consistency, residual = _dc_estimation_residual(L, q0, y0, slack_idx)
    # print(f"\nDC self-consistency at t=0: consistency={consistency:.4g}, "
    #       f"max|residual|={np.max(np.abs(residual)):.4g} MW "
    #       f"(near 0 expected -- see max_dc_self_consistency_residual_mw in metadata.json: "
    #       f"{metadata.get('max_dc_self_consistency_residual_mw')})")

    dict_edge_to_col = {e: i for i, e in enumerate(nx.complete_graph(metadata[N_BUSES]).edges())}
    _M_complete = len(dict_edge_to_col)

    cfg_power_graph = {
        "num_iterations": 1,
        "n": metadata[N_BUSES],
        "m": _M_complete,
        "B": B,
        "poly_coefficients": np.array([0, 1.0]),
        "new_edge_weight": np.mean(full_weights[full_weights > 0]),
        "trajectory_time": np.arange(0, metadata[T]),
        "full_weights": full_weights,
        "_edge_to_col": dict_edge_to_col,
        "line_ids": line_ids,
        "slack_idx": slack_idx,
    }
    cfg_power_graph.update({
        "F": np.dot(1, np.eye(cfg_power_graph["m"])),
    })
    return cfg_power_graph


def _dc_estimation_residual(L, q, y, slack_idx):
    """DC (linearized) load-flow check for one time step.

    Estimates bus angles from the load injections y via least squares on the
    reduced (slack-removed) Laplacian, and returns:
      - consistency: sum(q[mask] - q[slack_idx] - DCestimation), which should
        be ~0 if q and y are consistent with the DC power-flow model given L.
      - residual: y - L @ q, the raw measurement residual.
    """
    # lstsq (SVD-based) instead of inv(L_sub.T @ L_sub) @ L_sub.T @ y: some buses
    # are isolated at some time steps (no active line), which makes the reduced
    # Laplacian rank-deficient and the normal-equations inverse singular.
    zero_rows = np.where(np.all(L == 0, axis=1))[0]
    mask_rows = np.delete(np.arange(L.shape[1]), zero_rows)
    c = np.concat([zero_rows, np.array([slack_idx])], axis=0)
    mask_cols = np.delete(np.arange(L.shape[1]), c)
    L_sub = L[np.ix_(mask_rows, mask_cols)]
    DCestimation, *_ = np.linalg.lstsq(L_sub, y[mask_rows], rcond=None)
    # DCestimation = np.linalg.inv(L_sub.T @ L_sub + L[mask_cols,mask_cols]) @ L_sub.T @ y[mask_rows]
    consistency = np.sum(q[mask_cols] - q[slack_idx] - DCestimation)
    # Full-length (not masked) so callers can rely on a fixed N_buses shape --
    # islanded (zero_rows) entries come out ~0 here anyway (y and L's row are
    # both forced to 0 for those buses at generation time).
    # residual = y[mask_rows] - L_sub @ q[mask_cols]
    residual = y - L @ q
    return consistency, residual


def _run_one_dataset(args):
    """Worker — must be module-level to be picklable on Windows (spawn)."""
    ds_folder, mc_idx, n_total, active_methods, cfg_power_graph1 = args
    import copy, os, time, logging
    import numpy as np
    from util_func import build_L, one_method_evaluation
    from constants import METHOD_REGISTRY

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"[MC {mc_idx + 1}/{n_total}] {ds_folder}")

    va_mc = np.load(f"{ds_folder}/bus_va_rad.npy")  # (T, N) radians
    net_injection_mc = np.load(f"{ds_folder}/bus_net_injection_mw.npy")     # (T, N) MW
    pos_mc = np.load(f"{ds_folder}/pos_mc.npy")

    updated_connections_mc = [np.sort(np.where(col > 0.001)[0]) for col in pos_mc]

    # DC-estimation residual check over the trajectory
    DC_RESIDUAL_CHECK = False
    if DC_RESIDUAL_CHECK:
        B_mc = cfg_power_graph1["B"]
        slack_idx_mc = cfg_power_graph1["slack_idx"]
        T_mc = va_mc.shape[0]
        dc_consistency_mc = np.zeros(T_mc)
        dc_residuals_mc = np.zeros_like(net_injection_mc)
        for _t in range(T_mc):
            L_t = build_L(B_mc, pos_mc[_t, :])
            dc_consistency_mc[_t], dc_residuals_mc[_t, :] = _dc_estimation_residual(
                L_t, va_mc[_t, :], net_injection_mc[_t, :], slack_idx_mc
            )
        dc_residual_cov_mc = np.cov(dc_residuals_mc.T)
        q_cov_mc = np.cov(va_mc.T)
        y_cov_mc = np.cov(net_injection_mc.T)
        logging.info(
            f"  [{mc_idx+1}] DC residual check: mean|consistency|="
            f"{np.mean(np.abs(dc_consistency_mc)):.4g}, mean|residual|="
            f"{np.mean(np.abs(dc_residuals_mc)):.4g}"
            f"q_cov_mc rank = {np.linalg.matrix_rank(q_cov_mc):.4g}"
            f"y_cov_mc rank = {np.linalg.matrix_rank(y_cov_mc):.4g}"
        )

    cfg_run = copy.deepcopy(cfg_power_graph1)
    if active_methods[0] == "oracle-block":
        # Binarize conn_mc here -- this is "connected or not" for a generic
        # new_edge_weight guess, not the true live weight (that's what pos_mc is
        # for). conn_mc[0,:] can now be a fraction (e.g. 1/3) for a bus pair with
        # several parallel branches where only some are up; using the raw
        # fraction would scale the initial guess down for no good reason (it's
        # not a more accurate guess, just a different, arbitrary one) and leak
        # partial-connectivity info the "know topology, guess magnitude" oracle
        # premise isn't meant to have.
        conn_binary_0 = (pos_mc[0, :] > 0.001).astype(float)
        stateInit_mc = cfg_power_graph1["new_edge_weight"] * conn_binary_0.reshape(-1, 1)
        cfg_run["stateInit"] = stateInit_mc
        # # Need also to contain uncertinity in new added weight
        # # and + nonexisiting weight are set to zero inside the update stage
        cfg_run["C_x"] = cfg_power_graph1["new_edge_weight"] * cfg_run["C_x_missmatch"].copy()

    results = {}
    for name in active_methods:
        filt = METHOD_REGISTRY[name](cfg_run)
        t0 = time.time()
        mse, normalized_mse, f1, eier, normalized_eier, times = one_method_evaluation(
            filt, va_mc, net_injection_mc, pos_mc, updated_connections_mc
        )
        logging.info(f"  [{mc_idx + 1}] {name}: {time.time() - t0:.2f}s")
        results[name] = dict(mse=mse, normalized_mse=normalized_mse, f1=f1,
                             eier=eier, normalized_eier=normalized_eier, times=times)
    if DC_RESIDUAL_CHECK:
        results["dc_diag"] = dict(consistency=dc_consistency_mc, residuals=dc_residuals_mc,
                                   residual_cov=dc_residual_cov_mc)
    results[PROVENANCE_KEY] = os.path.basename(ds_folder)
    return mc_idx, results


if __name__ == "__main__":
    import glob as _glob
    import os
    import time
    import logging
    import pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from util_func import pick_worker_count, plot_metric
    from constants import LABELS, METHODS_ORDER, PROVENANCE_KEY, FOLDER_POWER_DATA_RTE7000

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


    dataset_dirs = sorted(_glob.glob("Power_data/rte7000_ac_syn_dataset*"), key=lambda p: int(p.split("dataset")[-1]))
    prepare_cfg_fn = prepare_cfg_dc

    #    dataset_dirs = dataset_dirs[:2]
    n_total = len(dataset_dirs)
    logging.info(f"Found {n_total} dataset(s) for Monte-Carlo evaluation.")
    active_methods = ("oracle-block",)#"fast-ekf", )#"oracle-block", "gsp-ekf", )"prob_ssm", "change-det", "grls",)
    reserve_cores = 1 if "SLURM_JOB_ID" in os.environ else 5
    n_workers = min(25, pick_worker_count(reserve_cores=reserve_cores))
    logging.info(f"Launching {n_workers} parallel workers.")

    run_power = [None] * n_total
    cfg_power_graph = prepare_cfg_dc(dataset_dirs[0])
    cfg_power_graph["sigma_x1"] = 1
    for method_name in active_methods:
        if method_name == "oracle-block":
            cfg_power_graph.update(cfg_oracle)
        elif method_name == "gsp-ekf":
            cfg_power_graph.update(cfg_gsp_ekf)
        elif method_name == "fast-ekf":
            cfg_power_graph.update(cfg_fast_ekf)
        elif method_name == "grls":
            cfg_power_graph.update(cfg_grls)
        elif method_name == "change-det":
            cfg_power_graph.update(cfg_change_det)
        if method_name == "oracle-block" or method_name == "gsp-ekf" or method_name == "fast-ekf":
            cfg_power_graph.update({
                "C_w_sqrt": np.dot(cfg_power_graph["sigma_w"], np.eye(cfg_power_graph["n"])),
                "C_u_sqrt": np.dot(cfg_power_graph["sigma_v"], np.eye(cfg_power_graph["m"])),
                "C_x_missmatch": np.dot(cfg_power_graph["sigma_x"] ** 2, np.eye(cfg_power_graph["m"])),
                "stateInit_missmatch": cfg_power_graph["new_edge_weight"] *
                                       np.ones(cfg_power_graph["m"]).reshape([cfg_power_graph["m"], 1]),
            })
            cfg_power_graph.update({
                "C_w": cfg_power_graph["C_w_sqrt"] @ cfg_power_graph["C_w_sqrt"],
                "C_u": cfg_power_graph["C_u_sqrt"] @ cfg_power_graph["C_u_sqrt"],
            })
        tasks = [(ds, i, n_total, (method_name,), cfg_power_graph) for i, ds in enumerate(dataset_dirs)]
        t_wall = time.time()

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one_dataset, t): t[1] for t in tasks}
            for fut in as_completed(futures):
                mc_idx, results = fut.result()
                run_power[mc_idx] = results
                done = sum(r is not None for r in run_power)
                logging.info(f"Completed {done}/{n_total} ({time.time() - t_wall:.1f}s elapsed)")
        logging.info(f"All {n_total} MC runs done in {time.time() - t_wall:.1f}s")

        save_folder = FOLDER_POWER_DATA_RTE7000
        FILE_POWER_GRAPH_VS_TIME = os.path.join(save_folder, method_name + ".pkl")
        with open(FILE_POWER_GRAPH_VS_TIME, "wb") as f:
            pickle.dump(run_power, f)

        plot_metric(cfg_power_graph["trajectory_time"], run_power, "mse", labels=LABELS,
                    methods_to_plot=METHODS_ORDER, log_format=True, to_save=True,
                    folder_name=save_folder, suffix="power_mc")
        plot_metric(cfg_power_graph["trajectory_time"], run_power, "f1", labels=LABELS,
                    methods_to_plot=METHODS_ORDER, to_save=True,
                    folder_name=save_folder, suffix="power_mc")

    a = 5
