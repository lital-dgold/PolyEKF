import json
from util_func import build_L, vector2diag, num_possible_edges
import numpy as np
import networkx as nx
from util_func import extract_x_to_match_B_from_L

n_buses = 57 # ToDo: turn to automate
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
#Best parameters
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
    folder_name = "Power_data/ieee57_dataset0"
    # Load all arrays
    bus_voltages  = np.load(f"{folder_name}/bus_voltages.npy")   # (T, 57, 2)
    bus_loads     = np.load(f"{folder_name}/bus_loads.npy")       # (T, 57, 2)
    line_topology = np.load(f"{folder_name}/line_topology.npy")   # (T, 186)
    bus_ids       = np.load(f"{folder_name}/bus_ids.npy")
    line_ids      = np.load(f"{folder_name}/line_ids.npy")
    Y_mat         = np.load(f"{folder_name}/Y_mat.npy")
    with open(f"{folder_name}/metadata.json") as json_data:
        metadata = json.load(json_data)
        json_data.close()
    # ── Voltages ──────────────────────────────────────────────────────
    vm  = bus_voltages[:, :, 0]   # (T, 57) voltage magnitude in pu
    va = bus_voltages[:, :, 1]


    for t in range(metadata["T"]):
        if np.isnan(np.sum(vm[t,:])):
            print(vm[t,:])
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

    print(f"\nLine {line_ids[line_idx]} (from {line_ids[line_idx,0]} to {line_ids[line_idx,1]}):")
    print(f"  topology (first 10): {line_topology[:10, line_idx]}")
    # print(f"  P_flow   (first 10): {P_flow[:10, line_idx].round(2)} MW")

    slack_idx = metadata["slack_bus"]
    # G = empty_graph(metadata[N_BUSES])
    # G.add_edges_from(line_ids)
    # B = nx.incidence_matrix(G, oriented=True).todense()# nx.incidence_matrix(nx.complete_graph(cfg_N_20_graph["n"], create_using=None), oriented=True).todense()
    B = nx.incidence_matrix(nx.complete_graph(metadata[N_BUSES], create_using=None), oriented=True).todense()
    L = -np.imag(Y_mat) - vector2diag(-np.imag(Y_mat) @ np.ones([Y_mat.shape[0],1]))

    ####
    y = P_load[0,:].T
    q = va[0,:].T.copy()
    # 1. Create the mask to exclude the 69th column (index slack_idx)
    # This is cleaner than using range(57) != slack_idx
    mask = np.delete(np.arange(L.shape[1]), slack_idx)
    L_sub = L[:, mask]

    # 2. Perform the Least Squares calculation
    # Formula: (L.T @ L)^-1 @ L.T @ y
    DCestimation = np.linalg.inv(L_sub.T @ L_sub) @ L_sub.T @ y

    print(np.sum(q[mask]-q[slack_idx]-DCestimation))
    residuals = P_load - (L @ q)
    z = np.cov(residuals.T)
    ####

    full_weights = extract_x_to_match_B_from_L(L, B)
    z = (build_L(B, full_weights)-L)
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
        "new_edge_weight": np.mean(full_weights[full_weights>0]),
        "trajectory_time": np.arange(0, metadata[T]),
        "full_weights": full_weights,
        "_edge_to_col": dict_edge_to_col,
        "line_ids": line_ids,
    }

    cfg_power_graph.update({
        "F": np.dot(1, np.eye(cfg_power_graph["m"])),
    })
    return cfg_power_graph

def _run_one_dataset(args):
    """Worker — must be module-level to be picklable on Windows (spawn)."""
    ds_folder, mc_idx, n_total, active_methods, cfg_power_graph1 = args
    import copy, time, logging
    import numpy as np
    from util_func import vector2diag, one_method_evaluation
    from constants import METHOD_REGISTRY

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"[MC {mc_idx+1}/{n_total}] {ds_folder}")

    bv = np.load(f"{ds_folder}/bus_voltages.npy")
    bl = np.load(f"{ds_folder}/bus_loads.npy")
    lt = np.load(f"{ds_folder}/line_topology.npy")

    va_mc = bv[:, :, 1]
    P_load_mc = bl[:, :, 0]

    conn_mc = np.zeros((lt.shape[0], cfg_power_graph1["m"]))
    for _k in range(cfg_power_graph1["line_ids"].shape[0]):
        _u, _v = int(cfg_power_graph1["line_ids"][_k, 0]), int(cfg_power_graph1["line_ids"][_k, 1])
        conn_mc[:, cfg_power_graph1["_edge_to_col"][(min(_u, _v), max(_u, _v))]] = lt[:, _k]

    pos_mc = (conn_mc.T * cfg_power_graph1["full_weights"]).T
    updated_connections_mc = [np.sort(np.where(col > 0)[0]) for col in conn_mc]

    cfg_run = copy.deepcopy(cfg_power_graph1)
    if active_methods[0] == "oracle-block":# or active_methods[0] == "gsp-ekf" or active_methods[0] == "fast-ekf":
        stateInit_mc = cfg_power_graph1["new_edge_weight"] * conn_mc[0, :].reshape(-1, 1)
        cfg_run["stateInit"] = stateInit_mc
        cfg_run["C_x"] = np.dot(cfg_run["sigma_x"] ** 2, vector2diag(stateInit_mc))


    results = {}
    for name in active_methods:
        filt = METHOD_REGISTRY[name](cfg_run)
        t0 = time.time()
        mse, normalized_mse, f1, eier, normalized_eier, times = one_method_evaluation(
            filt, va_mc, P_load_mc, pos_mc, updated_connections_mc
        )
        logging.info(f"  [{mc_idx+1}] {name}: {time.time()-t0:.2f}s")
        results[name] = dict(mse=mse, normalized_mse=normalized_mse, f1=f1,
                             eier=eier, normalized_eier=normalized_eier, times=times)
    return mc_idx, results


if __name__ == "__main__":
    import glob as _glob
    import os
    import time
    import logging
    import pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from util_func import pick_worker_count, plot_metric
    from constants import LABELS, METHODS_ORDER


    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[var] = "1"



    dataset_dirs = sorted(_glob.glob("Power_data/ieee57_dataset*"), key=lambda p: int(p.split("dataset")[-1]))#
    # dataset_dirs = ["Power_data\\ieee57_dataset33","Power_data\\ieee57_dataset66"]

    n_total = len(dataset_dirs)
    logging.info(f"Found {n_total} dataset(s) for Monte-Carlo evaluation.")
    active_methods = ("change-det",)#"fast-ekf", )#"oracle-block", "gsp-ekf", )"prob_ssm", "change-det", "grls",)
    n_workers = pick_worker_count()
    logging.info(f"Launching {n_workers} parallel workers.")

    run_power = [None] * n_total

    cfg_power_graph = prepare_cfg()
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
        # for t in tasks:
        #     mc_idx, results = _run_one_dataset(t)
        #     run_power[mc_idx] = results
        #     done = sum(r is not None for r in run_power)
        #     logging.info(f"Completed {done}/{n_total} ({time.time()-t_wall:.1f}s elapsed)")

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one_dataset, t): t[1] for t in tasks}
            for fut in as_completed(futures):
                mc_idx, results = fut.result()
                run_power[mc_idx] = results
                done = sum(r is not None for r in run_power)
                logging.info(f"Completed {done}/{n_total} ({time.time()-t_wall:.1f}s elapsed)")
        logging.info(f"All {n_total} MC runs done in {time.time()-t_wall:.1f}s")

        save_folder = "Power_data"
        FILE_POWER_GRAPH_VS_TIME = os.path.join(save_folder, method_name + ".pkl")
        with open(FILE_POWER_GRAPH_VS_TIME, "wb") as f:
            pickle.dump(run_power, f)

        plot_metric(cfg_power_graph["trajectory_time"], run_power, "mse", labels=LABELS,
                    methods_to_plot=METHODS_ORDER, log_format=True, to_save=True,
                    folder_name=save_folder, suffix="power_mc")
        plot_metric(cfg_power_graph["trajectory_time"], run_power, "f1", labels=LABELS,
                    methods_to_plot=METHODS_ORDER, to_save=True,
                    folder_name=save_folder, suffix="power_mc")

    a=5