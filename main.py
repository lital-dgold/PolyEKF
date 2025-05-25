# -*- coding: utf-8 -*-
import networkx as nx

import numpy as np
import scipy

import scipy.sparse as sp
import pickle

import time
import logging
import multiprocessing
from functools import partial

from EKF_modules import get_trajectory, one_method_evaluation, ExtendedKalmanFilter, \
    sparseKalmanFilter, sparseKalmanFilterISTA, oraclKalmanFilt_paper, \
    oraclKalmanFilt_paper_delayed, oraclKalmanFilt_nocovariance_update, \
    oraclKalmanFilt_diagonalovariance_update, \
    num_possible_edges, chose_indices_without_repeating, ChangeDetectionMethod, FastExtendedKalmanFilter
from util_func import build_L, vector2diag
# from util_func import compute_poly, compute_Jacobian_poly
from change_detection_module import FastChangeDetectionMethod

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class simulation:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        logging.info(f"[{self.name}] started...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        elapsed = self.end - self.start
        logging.info(f"[{self.name}] completed in {elapsed:.3f} seconds.")


def single_monte_carlo(
        cfg: dict,                       # all fixed simulation parameters
        active_methods: tuple = ("ekf", "gsp-ekf", "gsp-istap")
    ) -> dict[str,dict[str,np.ndarray]]:

    # -----------------------------------------------------------------------
    # 1.  Synthesise data ----------------------------------------------------
    traj_t = cfg["trajectory_time"]
    pos, q_meas, y_meas, conn, stateInit = get_trajectory(
        traj_t, cfg["F"], cfg["B"],
        cfg["C_w_sqrt"], cfg["C_u"], cfg["n"], cfg["k"], cfg["poly_coefficients"], cfg["new_edge_weight"], cfg["num_edges_stateinit"]
    )

    # -----------------------------------------------------------------------
    # 2.  Build every requested estimator -----------------------------------
    results = {}
    cfg.update({"stateInit": stateInit})
    cfg.update({"C_x": np.dot(cfg["sigma_x"] ** 2, vector2diag(stateInit))})
    for name in active_methods:
        filt = METHOD_REGISTRY[name](cfg)
        start = time.time()
        mse, f1, eier, times = one_method_evaluation(
            filt, q_meas, y_meas, pos, conn
        )
        end = time.time()
        elapsed = end - start
        logging.info(f"Run {name}: Execution time = {elapsed:.6f} seconds")
        results[name] = dict(mse=mse, f1=f1, eier=eier, times=times)

    return results            # {"ekf": {...}, "gsp-ekf": {...}, ...}


def run_monte_carlo_simulation(cfg, num_iter, active_methods):
    return [single_monte_carlo(cfg, active_methods) for _ in range(num_iter)]


def single_monte_carlo_iteration(cfg, active_methods, iterable):
    return single_monte_carlo(cfg, active_methods)



METHOD_REGISTRY = {
    # ----- Baseline EKF -----------------------------------------------------
    "ekf": lambda cfg: ExtendedKalmanFilter(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"]
    ),
    "fast-ekf":lambda cfg: FastExtendedKalmanFilter(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"]
    ),

    # ----- Sparse EKF (hard threshold) -------------------------------------
    "gsp-ekf": lambda cfg: sparseKalmanFilter(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], thr=cfg["thr1"]
    ),

    "gsp-ekf25": lambda cfg: sparseKalmanFilter(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], thr=0.25
    ),
    "gsp-ekf2": lambda cfg: sparseKalmanFilter(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], thr=0.2
    ),
    "gsp-ekf15": lambda cfg: sparseKalmanFilter(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], thr=0.15
    ),

    "gsp-ekf1": lambda cfg: sparseKalmanFilter(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], thr=0.1
    ),
    "gsp-ekf05": lambda cfg: sparseKalmanFilter(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], thr=0.05
    ),
    # ----- Sparse EKF with ISTA refinement ---------------------------------
    "gsp-istap-0.4": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=0.4  # tweak as needed
    ),
    "gsp-istap-0.5": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=0.5  # tweak as needed
    ),
    # ----- Sparse EKF with ISTA refinement ---------------------------------
    "gsp-istap-0.6": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=0.6  # tweak as needed
    ),
    # ----- Sparse EKF with ISTA refinement ---------------------------------
    "gsp-istap-0.7": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=0.7  # tweak as needed
    ),
    # ----- Sparse EKF with ISTA refinement ---------------------------------
    "gsp-istap-0.8": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=0.8  # tweak as needed
    ),
    # ----- Sparse EKF with ISTA refinement ---------------------------------
    "gsp-istap-0.9": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=0.9          # tweak as needed
    ),
    "gsp-istap-1": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=1  # tweak as needed
    ),
    "gsp-istap-1.1": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=1.1  # tweak as needed
    ),
    "gsp-istap-1.2": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=1.2  # tweak as needed
    ),
    "gsp-istap-1.3": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=1.3  # tweak as needed
    ),
    "gsp-istap-1.4": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=1.4  # tweak as needed
    ),
    "gsp-istap": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=0.05  # tweak as needed
    ),

    # ----- SBL EKF ---------------------------------------
    # "sbl-ekf": lambda cfg: SBL_EKF(
    #     cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
    #     cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
    #     cfg["poly_coefficients"], cfg["mu"],
    # ),

    # ----- Oracle variants --------------------------------------------------
    "oracle-diag": lambda cfg: oraclKalmanFilt_diagonalovariance_update(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x"], cfg["stateInit"], cfg["poly_coefficients"]
    ),

    "oracle-delayedCov": lambda cfg: oraclKalmanFilt_paper_delayed(
    cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
    cfg["C_x"], cfg["stateInit"], cfg["poly_coefficients"]
    ),

    "oracle-nocov": lambda cfg: oraclKalmanFilt_nocovariance_update(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x"], cfg["stateInit"], cfg["poly_coefficients"]
    ),

    "oracle-block": lambda cfg: oraclKalmanFilt_paper(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x"], cfg["stateInit"], cfg["poly_coefficients"]
    ),
    # ----- Change-detection baseline ---------------------------------------
    "change-det": lambda cfg: ChangeDetectionMethod(
        cfg["B"], cfg["n"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=cfg["lambda_1"], lambda_2=cfg["lambda_2"]
    ),

    # ----- Change-detection baseline ---------------------------------------
    "change-det-fast": lambda cfg: FastChangeDetectionMethod(
        cfg["B"], cfg["n"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=cfg["lambda_1"], lambda_2=cfg["lambda_2"]
    ),
}


def add_method(list_of_methods, cfg, existing_method_dict):
    new_runs = run_monte_carlo_simulation(cfg, num_iterations, list_of_methods)
    new_runs[0].update(existing_method_dict[0])
    return list(new_runs)


import copy
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

def _performance_vs_graph_size(n, base_cfg, sigma_x, sigma_v, sigma_w, num_iterations, active_methods):
    cfg = copy.deepcopy(base_cfg)  # avoid concurrent mutation
    k = 2 * n  # num of time samples to concat
    m = num_possible_edges(n)
    num_edges = 3 * n
    stateInit_missmatch = np.ones(m).reshape([m, 1])
    G = nx.complete_graph(n, create_using=None)
    C_w_sqrt = np.dot(sigma_w, np.eye(n))
    B = nx.incidence_matrix(G, oriented=True).todense()
    F = np.dot(1, np.eye(m))
    C_u = np.dot(sigma_v ** 2, np.eye(m))
    C_x_missmatch = np.dot(sigma_x ** 2, np.eye(m))

    cfg.update({
        "F": F,
        "B": B,
        "C_u": C_u,
        "C_w_sqrt": C_w_sqrt,
        "C_w": C_w_sqrt @ C_w_sqrt,
        "C_x_missmatch": C_x_missmatch,
        "stateInit_missmatch": stateInit_missmatch,
        "num_edges_stateinit": num_edges,
        "n": n,
        "k": k,
        "thr1": thr1,
        "thr2": thr2,
    })
    return n, run_monte_carlo_simulation(cfg, num_iterations, active_methods)

def _performance_vs_snr(sigma_w1,base_cfg, n, num_iterations, active_methods):
    cfg = copy.deepcopy(base_cfg)  # avoid concurrent mutation
    logging.info(f"sigma_w={10 * np.log10(sigma_w1)} db\n")
    C_w_sqrt = np.dot(sigma_w1, np.eye(n))
    cfg.update({"C_w_sqrt": C_w_sqrt})
    cfg.update({"C_w": C_w_sqrt @ C_w_sqrt})
    return sigma_w1, run_monte_carlo_simulation(cfg, num_iterations, active_methods)

def _performance_vs_change_rate(k, base_cfg, n, num_iterations, active_methods):
    cfg = copy.deepcopy(base_cfg)  # avoid concurrent mutation
    logging.info(f"change rate={k} * n %\n")
    cfg.update({"k": k * n})
    return k, run_monte_carlo_simulation(cfg, num_iterations, active_methods)

def _performance_vs_sparsity(num_edges, base_cfg, m, num_iterations, active_methods):
    """
    Runs one Monte-Carlo simulation for a given sparsity level.
    Executed in a separate process → all arguments must be picklable.
    """
    cfg = copy.deepcopy(base_cfg)  # avoid concurrent mutation
    cfg["num_edges_stateinit"] = int(round(m * num_edges / 100))
    logging.info(f"sparsity={num_edges}%")  # this log is process-local
    return num_edges, run_monte_carlo_simulation(cfg, num_iterations, active_methods)


if __name__ == "__main__":
    num_time_samples = 159
    sigma_v = 0.1
    sigma_w = 0.1
    sigma_x = 0.5
    n = 20
    thr1 = 0.2
    thr2 = 0.2
    num_iterations = 5#1000
    k = 2 * n  # num of time samples to concat
    m = num_possible_edges(n)
    num_edges = 3 * n
    new_edge_weight = 1
    # idx_list = chose_indices_without_repeating(m, num_edges)
    # stateInit = np.zeros(m).reshape([m, 1])
    # stateInit[idx_list] = new_edge_weight
    stateInit_missmatch = np.ones(m).reshape([m, 1])
    trajectory_time = np.arange(0, num_time_samples)
    G = nx.complete_graph(n, create_using=None)

    sparse_flag = False
    if sparse_flag:
        C_w_sqrt = sigma_w * sp.identity(n, format="csr", dtype=np.float64)
        B = nx.incidence_matrix(G, oriented=True).tocsr()
        F = sp.identity(m, format="csr", dtype=np.float64)
        C_u = (sigma_v ** 2) * sp.identity(m, format="csr", dtype=np.float64)
        # C_x = (sigma_x ** 2) * sp.identity(m, format="csr", dtype=np.float64)
        C_x_missmatch =(sigma_x ** 2) * sp.identity(m, format="csr", dtype=np.float64)
    else:
        C_w_sqrt = np.dot(sigma_w, np.eye(n))
        B = nx.incidence_matrix(G, oriented=True).todense()
        F = np.dot(1, np.eye(m))
        C_u = np.dot(sigma_v ** 2, np.eye(m))
        # C_x = np.dot(sigma_x ** 2, vector2diag(stateInit))
        C_x_missmatch = np.dot(sigma_x ** 2, np.eye(m))
    mu = 1
    #("gsp-ekf05","gsp-ekf1","gsp-ekf15","gsp-ekf2","gsp-ekf25")#
    active_methods = ("fast-ekf", "gsp-ekf")#, "oracle-block")#, "change-det")
    # active_methods = ("change-det","change-det-fast",) # "oracle-diag", "oracle-delayedCov", "oracle-nocov",
                       #"oracle-delayedCov",
                       # "change-det")
    cfg = {
        "F": F,
        "B": B,
        "C_u": C_u,
        "C_w_sqrt": C_w_sqrt,
        "C_w": C_w_sqrt @ C_w_sqrt,
        "C_x_missmatch": C_x_missmatch,
        "sigma_x": sigma_x,
        "stateInit_missmatch": stateInit_missmatch,
        # "stateInit": stateInit,
        "new_edge_weight": new_edge_weight,
        "num_edges_stateinit": num_edges,
        "trajectory_time": trajectory_time,
        "n": n,
        "k": k,
        "thr1": thr1,
        "thr2": thr2,
        "mu": mu,
        "lambda_1": 0.2,
        "lambda_2": 0,
    }
    # ------ Linear case -------
    # with simulation("Linear case"):
    #     poly_coefficients = np.array([0, 1.0])
    #     cfg.update({"poly_coefficients": poly_coefficients})
    #     runs_linear = run_monte_carlo_simulation(cfg, num_iterations, active_methods)
    #
    #     # with multiprocessing.Pool() as pool:
    #     #     func = partial(single_monte_carlo_iteration,cfg, active_methods)
    #     #     runs_linear = pool.map(func, np.arange(num_iterations))
    #     # runs_linear2 = run_monte_carlo_simulation(cfg, num_iterations, active_methods)
    #
    #     # # Save
    #     # with open("runs_linear_data.pkl", "wb") as f:
    #     #     pickle.dump(runs_linear, f)
    #     #
    #     # # # Load
    #     # with open("runs_linear_data.pkl", "rb") as f:
    #     #     runs_linear_data1000MC = pickle.load(f)
    #     # # runs_linear = add_method(("change-det-fast",), cfg, runs_linear)
    #
    #     plot_metric(cfg["trajectory_time"], runs_linear, "mse", log_format=True)
    #     plot_metric(cfg["trajectory_time"], runs_linear, "f1")
    #     plot_metric(cfg["trajectory_time"], runs_linear, "eier")
    #     plot_metric(cfg["trajectory_time"], runs_linear, "times", log_format=True)
    #     # plot_metric(cfg["trajectory_time"], runs_linear2, "times", log_format=True)
    #     linear_table = create_table(runs_linear, "times")
    #     # linear_table2 = create_table(runs_linear2, "times")
    #
    # # ------ Non-Linear case -------
    # with simulation("Non-Linear case"):
    #     poly_coefficients = np.array([1.0, 1.0, 0.1, 1.0])
    #     cfg.update({"poly_coefficients": poly_coefficients})
    #     cfg.update({"lambda_1": 0.1})
    #     cfg.update({"lambda_2": 0})
    #
    #     with multiprocessing.Pool() as pool:
    #         func = partial(single_monte_carlo_iteration,cfg, active_methods)
    #         runs_nonlinear = pool.map(func, np.arange(num_iterations))
    #     runs_nonlinear2 = run_monte_carlo_simulation(cfg, num_iterations, active_methods)
    #     # Save
    #     # with open("runs_nonlinear_data.pkl", "wb") as f:
    #     #     pickle.dump(runs_nonlinear, f)
    #
    #     # # Load
    #     # with open("runs_nonlinear_data1000MC.pkl", "rb") as f:
    #     #     run111 = pickle.load(f)
    #     # # runs_nonlinear = add_method(("gsp-ekf", "gsp-istap-0.4", "gsp-istap-0.5", "gsp-istap-0.6", "gsp-istap-0.7", "gsp-istap-0.8", "gsp-istap-0.9", "gsp-istap-1", "gsp-istap-1.1", "gsp-istap-1.2", "gsp-istap-1.3", "gsp-istap-1.4"),# "oracle-delayedCov", "oracle-block"),
    #     # #  cfg, runs_nonlinear)
    #
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear2, "mse", log_format=True)
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear, "f1")
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear, "eier")
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear, "times", log_format=True)
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear2, "times", log_format=True)
    #     nonlinear_table = create_table(runs_nonlinear, "times")
    #     nonlinear_table2 = create_table(runs_nonlinear2, "times")

    # #########################################################################
    # ######################### - Performance vs. time  #######################
    # #########################################################################
    # with simulation("Non-Linear case"):
    #     num_iterations = 1000
    #     sigma_w = 0.2 ** 0.5
    #     poly_coefficients =  np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2])#np.array([1.0, 1.0, 0.1, 1.0])
    #     cfg.update({"poly_coefficients": poly_coefficients})
    #     cfg.update({"lambda_1": 4})
    #     cfg.update({"lambda_2": 0})
    #
    #     with multiprocessing.Pool() as pool:
    #         func = partial(single_monte_carlo_iteration,cfg, active_methods)
    #         runs_nonlinear_5order = pool.map(func, np.arange(num_iterations))
    #     # runs_nonlinear2 = run_monte_carlo_simulation(cfg, num_iterations, active_methods)
    #     # Save
    #     with open("runs_nonlinear_data_5order.pkl", "wb") as f:
    #         pickle.dump(runs_nonlinear_5order, f)
    #
    #     # # Load
    #     # with open("runs_nonlinear_data1000MC.pkl", "rb") as f:
    #     #     run111 = pickle.load(f)
    #     # # runs_nonlinear = add_method(("gsp-ekf", "gsp-istap-0.4", "gsp-istap-0.5", "gsp-istap-0.6", "gsp-istap-0.7", "gsp-istap-0.8", "gsp-istap-0.9", "gsp-istap-1", "gsp-istap-1.1", "gsp-istap-1.2", "gsp-istap-1.3", "gsp-istap-1.4"),# "oracle-delayedCov", "oracle-block"),
    #     # #  cfg, runs_nonlinear)
    #
    #     # plot_metric(cfg["trajectory_time"], runs_nonlinear2, "mse", log_format=True)
    #     # plot_metric(cfg["trajectory_time"], runs_nonlinear, "f1")
    #     # plot_metric(cfg["trajectory_time"], runs_nonlinear, "eier")
    #     # plot_metric(cfg["trajectory_time"], runs_nonlinear, "times", log_format=True)
    #     # plot_metric(cfg["trajectory_time"], runs_nonlinear2, "times", log_format=True)
    #     # nonlinear_table = create_table(runs_nonlinear, "times")
    #     # nonlinear_table2 = create_table(runs_nonlinear2, "times")
    #
    # #########################################################################
    # ################# - Performance vs. noise level  ########################
    # #########################################################################
    # with simulation("Non-Linear case - Performance vs. noise level"):
    #     num_iterations = 1000
    #     poly_coefficients = np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2]) #np.array([1.0, 1.0, 1.0, 1.0]) #
    #     cfg["lambda_1"] = 4
    #     cfg.update({"poly_coefficients": poly_coefficients})
    #     num_time_samples = 79
    #     trajectory_time = np.arange(0, num_time_samples)
    #     cfg.update({"trajectory_time": trajectory_time})
    #     sigma_w_list = np.logspace(-1.5, 1.5, 10)
    #
    #     # ---------- parallel sweep ----------
    #     snr_dict_list = [None] * len(sigma_w_list)  # preserve order
    #     max_workers = max(1, os.cpu_count() - 1)  # leave one core free
    #
    #     with ProcessPoolExecutor(max_workers=max_workers) as exe:
    #         futures = {exe.submit(_performance_vs_snr, e, cfg, n, num_iterations, active_methods): i
    #                    for i, e in enumerate(sigma_w_list)}
    #         for fut in as_completed(futures):
    #             idx = futures[fut]  # original position
    #             _, result = fut.result()  # (num_edges, runs_linear)
    #             snr_dict_list[idx] = result
    #
    #
    # # Save
    # with open("performance_vs_snr_5order.pkl", "wb") as f:
    #     pickle.dump(snr_dict_list, f)
    #
    # def mean_func_without_first_n(table):
    #     return table[:,n:].mean(axis=1)
    #
    # def mean_func(table):
    #     return table.mean(axis=1)

    # plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "mse", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="sigma_W [dB]")
    # plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "eier", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="sigma_W [dB]")
    #
    # plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "mse", aggregation_func=mean_func, log_format=False, x_label1="sigma_W [dB]")
    # plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "eier", aggregation_func=mean_func, log_format=False, x_label1="sigma_W [dB]")

    #########################################################################
    ############## - Performance vs. sparsity level  ########################
    #########################################################################
    with simulation("Non-Linear case - Performance vs. sparsity level"):
        num_iterations = 1000
        sigma_w = 0.2 ** 0.5
        # if num_edges <= 30:
        #     cfg["thr1"] = 0.2
        #     cfg["lambda_1"] = 0.2
        # else:
        cfg["lambda_1"] = 4
        poly_coefficients = np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2])#np.array([0.0, 1.0, 0.75, 0.5, 0.25])##np.array([1.0, 1.0, 1.0, 1.0])#
        cfg.update({"poly_coefficients": poly_coefficients})
        num_time_samples = 79
        trajectory_time = np.arange(0, num_time_samples)
        cfg.update({"trajectory_time": trajectory_time})
        c = np.linspace(1, 1 / n, n)  # start=1, stop=1/n, n points
        C_w_sqrt = np.dot(sigma_w, np.eye(n))#scipy.linalg.toeplitz(c)#
        cfg.update({"C_w_sqrt": C_w_sqrt})
        cfg.update({"C_w": C_w_sqrt @ C_w_sqrt})
        sparsity_list = np.linspace(5, 30, 6)
        # ---------- parallel sweep ----------
        sparsity_dict_list = [None] * len(sparsity_list)  # preserve order
        max_workers = max(1, os.cpu_count() - 1)  # leave one core free

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_performance_vs_sparsity, e, cfg, m, num_iterations, active_methods): i
                       for i, e in enumerate(sparsity_list)}
            for fut in as_completed(futures):
                idx = futures[fut]  # original position
                _, result = fut.result()  # (num_edges, runs_linear)
                sparsity_dict_list[idx] = result

    # Save
    with open("performance_vs_sparsity_5order.pkl", "wb") as f:
        pickle.dump(sparsity_dict_list, f)

    def mean_func_without_first_n(table):
        return table[:,n:].mean(axis=1)

    def mean_func(table):
        return table.mean(axis=1)

    # plot_vs_parameter(sparsity_list, sparsity_dict_list, "mse", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")
    # plot_vs_parameter(sparsity_list, sparsity_dict_list, "eier", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")
    #
    #
    # plot_vs_parameter(sparsity_list, sparsity_dict_list, "mse", aggregation_func=mean_func, log_format=False, x_label1="Sparsity [%]")
    # plot_vs_parameter(sparsity_list, sparsity_dict_list, "eier", aggregation_func=mean_func, log_format=False, x_label1="Sparsity [%]")

    #########################################################################
    ############## - Performance vs. rate of graph variations  ##############
    #########################################################################
    with simulation("Non-Linear case - Performance vs. graph variation"):
        num_iterations = 1000
        # poly_coefficients = np.array([0.0, 1.0, 0.75, 0.5, 0.25])
        # cfg.update({"poly_coefficients": poly_coefficients})
        num_time_samples = 79
        sigma_w = 0.2 ** 0.5
        trajectory_time = np.arange(0, num_time_samples)
        cfg.update({"trajectory_time": trajectory_time})
        C_w_sqrt = np.dot(sigma_w, np.eye(n))
        cfg.update({"C_w_sqrt": C_w_sqrt})
        cfg.update({"C_w": C_w_sqrt @ C_w_sqrt})
        cfg.update({"num_edges_stateinit": 2 * n})
        k_list = np.linspace(0.25, 2, 8)
        # ---------- parallel sweep ----------
        k_dict_list = [None] * len(k_list)  # preserve order
        max_workers = max(1, os.cpu_count() - 1)  # leave one core free

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_performance_vs_change_rate, e, cfg, n, num_iterations, active_methods): i
                       for i, e in enumerate(k_list)}
            for fut in as_completed(futures):
                idx = futures[fut]  # original position
                _, result = fut.result()  # (num_edges, runs_linear)
                k_dict_list[idx] = result
    # Save
    with open("performance_vs_graph_variation_5order.pkl", "wb") as f:
        pickle.dump(k_dict_list, f)

    def mean_func_without_first_n(table):
        return table[:,n:].mean(axis=1)

    def mean_func(table):
        return table.mean(axis=1)

    # plot_vs_parameter(sparsity_list, k_dict_list, "mse", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")
    # plot_vs_parameter(sparsity_list, k_dict_list, "eier", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")

    ## - Scalability to different graph sizes
    # with simulation("Non-Linear case - Performance vs. graph sizes"):
    #     num_iterations = 1
    #     # poly_coefficients = np.array([0.0, 1.0, 0.75, 0.5, 0.25])
    #     # cfg.update({"poly_coefficients": poly_coefficients})
    #     num_time_samples = 79
    #     trajectory_time = np.arange(0, num_time_samples)
    #     cfg.update({"trajectory_time": trajectory_time})
    #     C_w_sqrt = np.dot(sigma_w, np.eye(n))
    #     cfg.update({"C_w_sqrt": C_w_sqrt})
    #     cfg.update({"C_w": C_w_sqrt @ C_w_sqrt})
    #     n_list = np.round(np.logspace(10, 1000, 8)).astype(int)
    #     dict_list = []
    #
    #
    #     # ---------- parallel sweep ----------
    #     dict_list = [None] * len(n_list)  # preserve order
    #     max_workers = max(1, os.cpu_count() - 1)  # leave one core free
    #
    #     with ProcessPoolExecutor(max_workers=max_workers) as exe:
    #         futures = {exe.submit(_performance_vs_graph_size, e, cfg, n, cfg, sigma_x, sigma_v, sigma_w, num_iterations, active_methods): i
    #                    for i, e in enumerate(n_list)}
    #         for fut in as_completed(futures):
    #             idx = futures[fut]  # original position
    #             _, result = fut.result()  # (num_edges, runs_linear)
    #             dict_list[idx] = result
    #
    # # Save
    # with open("performance_vs_graph_sizes.pkl", "wb") as f:
    #     pickle.dump(dict_list, f)
    #
    # def mean_func_without_first_n(table):
    #     return table[:,n:].mean(axis=1)
    #
    # def mean_func(table):
    #     return table.mean(axis=1)
    #
    # plot_vs_parameter(sparsity_list, dict_list, "mse", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")
    # plot_vs_parameter(sparsity_list, dict_list, "eier", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")

    a = 5





