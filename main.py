# -*- coding: utf-8 -*-
import itertools
from functools import partial
from multiprocessing import Pool
import networkx as nx

import numpy as np
import scipy

import scipy.sparse as sp
import pickle

import time
import logging
import multiprocessing
from functools import partial
import copy
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from EKF_modules import get_trajectory, one_method_evaluation, ExtendedKalmanFilter, \
    sparseKalmanFilter, sparseKalmanFilterISTA, oraclKalmanFilt_paper, \
    oraclKalmanFilt_paper_delayed, oraclKalmanFilt_nocovariance_update, \
    oraclKalmanFilt_diagonalovariance_update, \
    num_possible_edges, chose_indices_without_repeating, ChangeDetectionMethod, FastExtendedKalmanFilter
from util_func import build_L, vector2diag, compute_metric_summary, pick_worker_count
# from util_func import compute_poly, compute_Jacobian_poly
from change_detection_module import FastChangeDetectionMethod
from constants import METHOD_REGISTRY

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)
import os
for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[var] = "1"           # one thread per Python process

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
        cfg["C_w_sqrt"], cfg["C_u_sqrt"], cfg["n"], cfg["k"], cfg["poly_coefficients"], cfg["new_edge_weight"], cfg["num_edges_stateinit"], cfg["delta_n"]
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


def add_method(list_of_methods, cfg, existing_method_dict):
    new_runs = run_monte_carlo_simulation(cfg, num_iterations, list_of_methods)
    new_runs[0].update(existing_method_dict[0])
    return list(new_runs)


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
    sigma_v = sigma_w1
    C_u_sqrt = np.dot(sigma_v, np.eye(cfg["C_u_sqrt"].shape[0]))
    cfg.update({"C_u_sqrt": C_u_sqrt})
    cfg.update({"C_u": C_u_sqrt @ C_u_sqrt})
    return sigma_w1, run_monte_carlo_simulation(cfg, num_iterations, active_methods)


def _performance_vs_change_size(delta_n, base_cfg, n, num_iterations, active_methods):
    cfg = copy.deepcopy(base_cfg)  # avoid concurrent mutation
    logging.info(f"change rate={delta_n} * n %\n")
    cfg.update({"delta_n": int(delta_n * n)})
    return delta_n, run_monte_carlo_simulation(cfg, num_iterations, active_methods)


def _performance_vs_change_rate(k, base_cfg, n, num_iterations, active_methods):
    cfg = copy.deepcopy(base_cfg)  # avoid concurrent mutation
    logging.info(f"change rate={k} * n %\n")
    cfg.update({"k": k * n})
    return k, run_monte_carlo_simulation(cfg, num_iterations, active_methods)

def _performance_vs_sparsity(num_edges, base_cfg, m, num_iterations, active_methods):
    """
    Runs one Monte-Carlo simulation for poly_c given sparsity level.
    Executed in poly_c separate process → all arguments must be picklable.
    """
    cfg = copy.deepcopy(base_cfg)  # avoid concurrent mutation
    cfg["num_edges_stateinit"] = int(round(m * num_edges / 100))
    logging.info(f"sparsity={num_edges}%")  # this log is process-local
    return num_edges, run_monte_carlo_simulation(cfg, num_iterations, active_methods)

# Define the evaluation function
def evaluate_lambdas(cfg, params):
    num_iterations = 2
    lambda_1, lambda_2 = params
    local_cfg = cfg.copy()
    local_cfg.update({"lambda_1": lambda_1, "lambda_2": lambda_2})
    try:
        result = run_monte_carlo_simulation(local_cfg, num_iterations, ("change-det",))
        methods_dict = compute_metric_summary(result, "mse", methods_to_plot= ("change-det",))
        avg_error = methods_dict[ "change-det"].mean()  # update key as needed
        return (lambda_1, lambda_2, avg_error)
    except Exception as e:
        print(f"Failed for lambda_1={lambda_1}, lambda_2={lambda_2}: {e}")
        return (lambda_1, lambda_2, float("inf"))


if __name__ == "__main__":
    num_time_samples = 159
    sigma_v = 0.01 ** 0.5
    sigma_w = 0.2 ** 0.5
    sigma_x = 0.5
    n = 10
    thr1 = 0.25
    thr2 = 0.2
    num_iterations = 1000
    k = int(2 * n)  # num of time samples to concat
    m = num_possible_edges(n)
    num_edges = int(1.5 * n)
    new_edge_weight = 1
    # idx_list = chose_indices_without_repeating(m, num_edges)
    # stateInit = np.zeros(m).reshape([m, 1])
    # stateInit[idx_list] = new_edge_weight
    stateInit_missmatch = new_edge_weight * np.ones(m).reshape([m, 1])
    trajectory_time = np.arange(0, num_time_samples)
    G = nx.complete_graph(n, create_using=None)

    sparse_flag = False
    if sparse_flag:
        C_w_sqrt = sigma_w * sp.identity(n, format="csr", dtype=np.float64)
        B = nx.incidence_matrix(G, oriented=True).tocsr()
        F = sp.identity(m, format="csr", dtype=np.float64)
        C_u_sqrt = (sigma_v) * sp.identity(m, format="csr", dtype=np.float64)
        # C_x = (sigma_x ** 2) * sp.identity(m, format="csr", dtype=np.float64)
        C_x_missmatch =(sigma_x ** 2) * sp.identity(m, format="csr", dtype=np.float64)
    else:
        C_w_sqrt = np.dot(sigma_w, np.eye(n))
        B = nx.incidence_matrix(G, oriented=True).todense()
        F = np.dot(1, np.eye(m))
        C_u_sqrt  = np.dot(sigma_v, np.eye(m))
        # C_x = np.dot(sigma_x ** 2, vector2diag(stateInit))
        C_x_missmatch = np.dot(sigma_x ** 2, np.eye(m))
    mu = 1
    #("gsp-ekf05","gsp-ekf1","gsp-ekf15","gsp-ekf2","gsp-ekf25")#
    active_methods = ("fast-ekf", "gsp-ekf", "oracle-block", "change-det")
    # active_methods = ("change-det","change-det-fast",) # "oracle-diag", "oracle-delayedCov", "oracle-nocov",
                       #"oracle-delayedCov",
                       # "change-det")
    cfg = {
        "F": F,
        "B": B,
        "C_u": C_u_sqrt @ C_u_sqrt,
        "C_u_sqrt": C_u_sqrt,
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
        "lambda_1": 3.16,
        "lambda_2": 0.316,
        "delta_n": max(1, int(0.01 * m)),
    }
    # # ------ Linear case -------
    # with simulation("Linear case"):
    #     poly_coefficients = np.array([0, 1.0])
    #     cfg.update({"poly_coefficients": poly_coefficients})
    #     num_iterations = 1
    #     runs_linear = run_monte_carlo_simulation(cfg, num_iterations, active_methods)
    # #
    #     # with multiprocessing.Pool() as pool:
    #     #     func = partial(single_monte_carlo_iteration,cfg, active_methods)
    #     #     runs_linear = pool.map(func, np.arange(num_iterations))
    #     # runs_linear = run_monte_carlo_simulation(cfg, num_iterations, active_methods)
    #
    #     # # Save
    #     # with open("runs_linear_data.pkl", "wb") as f:
    #     #     pickle.dump(runs_linear, f)
    #     #
    #     # # # Load
    #     # with open("runs_linear_data.pkl", "rb") as f:
    #     #     runs_linear_data1000MC = pickle.load(f)
    #     # # runs_linear = add_method(("change-det-fast",), cfg, runs_linear)
    #     from util_func import plot_metric, create_table
    #     plot_metric(cfg["trajectory_time"], runs_linear, "mse", log_format=True, to_save=True,suffix="linear_try")
    # #     plot_metric(cfg["trajectory_time"], runs_linear, "f1")
    # #     plot_metric(cfg["trajectory_time"], runs_linear, "eier")
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
    #     # runs_nonlinear2 = run_monte_carlo_simulation(cfg, num_iterations, active_methods)
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
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear, "mse", log_format=True)
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear, "f1")
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear, "eier")
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear, "times", log_format=True)
    #     # plot_metric(cfg["trajectory_time"], runs_nonlinear2, "times", log_format=True)
    #     nonlinear_table = create_table(runs_nonlinear, "times")
    #     # nonlinear_table2 = create_table(runs_nonlinear2, "times")

    #########################################################################
    ######################### - Tuning parameters  #######################
    #########################################################################
    # with simulation("Tuning parameters"):
    #     sigma_w = 0.2 ** 0.5
    #     poly_coefficients =  np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2])#np.array([1.0, 1.0, 0.1, 1.0])
    #     cfg.update({"poly_coefficients": poly_coefficients})
        # lambda_1_values = np.logspace(-1, 2, 5)  # e.g., [0.1, 1, 10, 100]
        # lambda_2_values = np.logspace(-2, 1, 5)  # e.g., [0.01, 0.1, 1, 10]
        # grid = list(itertools.product(lambda_1_values, lambda_2_values))
        #
        # results = [evaluate_lambdas(cfg, params) for params in grid]

        # # 'results' now contains one entry per grid point, in the same order as 'grid'
        #
        # with open("tuning_change_det.pkl", "wb") as f:
        #     pickle.dump(results, f)
        # # Find best
        # best_lambda = min(results, key=lambda x: x[2])
        # print(f"Best: lambda_1={best_lambda[0]}, lambda_2={best_lambda[1]}, avg_error={best_lambda[2]}")
        #
        # cfg.update({"lambda_1": best_lambda[0]})
        # cfg.update({"lambda_2": best_lambda[1]})
        # results = run_monte_carlo_simulation(cfg, num_iterations, active_methods)

    # #########################################################################
    # ######################### - Performance vs. time  #######################
    # #########################################################################
    # with simulation("Non-Linear case"):
    #     num_iterations = 1000
    #     num_time_samples = 79
    #     trajectory_time = np.arange(0, num_time_samples)
    #     poly_coefficients =  np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2])#np.array([1.0, 1.0, 0.1, 1.0])
    #     cfg.update({"poly_coefficients": poly_coefficients, "trajectory_time":trajectory_time})
    #     with multiprocessing.Pool() as pool:
    #         func = partial(single_monte_carlo_iteration,cfg, active_methods)
    #         runs_nonlinear_5order = pool.map(func, np.arange(num_iterations))
    #
    #     # Save
    #     with open("runs_nonlinear_data_5order_10nodes_1000MC_results_k2n_all.pkl", "wb") as f:
    #         pickle.dump(runs_nonlinear_5order, f)
    #     from util_func import plot_metric, create_table
    #
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear_5order, "mse", log_format=True, to_save=True, suffix="10nodes_1000mc_nonlinear")
    #     plot_metric(cfg["trajectory_time"], runs_nonlinear_5order, "eier",  to_save=True, suffix="10nodes_1000mc_nonlinear")
    #########################################################################
    ################# - Performance vs. noise level  ########################
    #########################################################################
    with simulation("Non-Linear case - Performance vs. noise level"):
        active_methods = ("fast-ekf", "gsp-ekf", "change-det")#"oracle-block",
        num_iterations = 100
        poly_coefficients = np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2]) #np.array([1.0, 1.0, 1.0, 1.0]) #
        cfg.update({"poly_coefficients": poly_coefficients})
        num_time_samples = 79
        trajectory_time = np.arange(0, num_time_samples)
        cfg.update({"trajectory_time": trajectory_time})
        sigma_w_list = np.logspace(-2, -0.5, 5)
        snr_dict_list = [None] * len(sigma_w_list)  # preserve order
        max_workers = pick_worker_count()#max(1, os.cpu_count() - 1)  # leave one core free

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_performance_vs_snr, e, cfg, n, num_iterations, active_methods): i
                       for i, e in enumerate(sigma_w_list)}
            for fut in as_completed(futures):
                idx = futures[fut]  # original position
                _, result = fut.result()  # (num_edges, runs_linear)
                snr_dict_list[idx] = result


    # Save
    with open("performance_vs_snr_5order_10nodes100MC_all.pkl", "wb") as f:
        pickle.dump(snr_dict_list, f)
    #
    # def mean_func_without_first_n(table):
    #     return table[:,n:].mean(axis=1)
    #
    def mean_func(table):
        return table.mean(axis=1)
    from util_func import plot_vs_parameter
    plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "mse", aggregation_func=mean_func, log_format=False, x_label1="sigma_W [dB]", to_save=True, suffix="mse_vs_snr_nonlinear")
    plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "eier", aggregation_func=mean_func, log_format=False, x_label1="sigma_W [dB]",to_save=True, suffix="eier_vs_snr_nonlinear")
    #
    # plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "mse", aggregation_func=mean_func, log_format=False, x_label1="sigma_W [dB]")
    # plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "eier", aggregation_func=mean_func, log_format=False, x_label1="sigma_W [dB]")
    #

    #########################################################################
    ############## - Performance vs. rate of graph variations  ##############
    #########################################################################
    with simulation("Non-Linear case - Performance vs. graph variation"):
        num_iterations = 1000
        num_time_samples = 79
        sigma_w = 0.2 ** 0.5
        trajectory_time = np.arange(0, num_time_samples)
        cfg.update({"trajectory_time": trajectory_time})
        C_w_sqrt = np.dot(sigma_w, np.eye(n))

        cfg.update({"C_w_sqrt": C_w_sqrt})
        cfg.update({"C_w": C_w_sqrt @ C_w_sqrt})
        C_u_sqrt = np.dot(sigma_v, np.eye(cfg["C_u"].shape[0]))
        cfg.update({"C_u_sqrt": C_u_sqrt, "C_u": C_u_sqrt @ C_u_sqrt})
        cfg.update({"num_edges_stateinit": 2 * n})
        k_list = np.linspace(0.25, 2, 8)
        # ---------- parallel sweep ----------
        k_dict_list = [None] * len(k_list)  # preserve order
        max_workers = 1#max(1, os.cpu_count() - 1)  # leave one core free

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_performance_vs_change_rate, e, cfg, n, num_iterations, active_methods): i
                       for i, e in enumerate(k_list)}
            for fut in as_completed(futures):
                idx = futures[fut]  # original position
                _, result = fut.result()  # (num_edges, runs_linear)
                k_dict_list[idx] = result
    # Save
    with open("performance_vs_graph_variation_5order_10nodes1000MC.pkl", "wb") as f:
        pickle.dump(k_dict_list, f)

    #
    # plot_vs_parameter(sparsity_list, k_dict_list, "mse", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")
    # plot_vs_parameter(sparsity_list, k_dict_list, "eier", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")
    #########################################################################
    ############## - Performance vs. rate of graph variations  ##############
    #########################################################################
    with simulation("Non-Linear case - Performance vs. graph variation"):
        # num_iterations = 2
        # cfg["lambda_1"] = 10
        # poly_coefficients = np.array(
        #     [0.0, 1.0, 0.8, 0.6, 0.4, 0.2])  # np.array([0.0, 1.0, 0.75, 0.5, 0.25])##np.array([1.0, 1.0, 1.0, 1.0])#
        # cfg.update({"poly_coefficients": poly_coefficients})
        #  # poly_coefficients = np.array([0.0, 1.0, 0.75, 0.5, 0.25])
        # cfg.update({"poly_coefficients": poly_coefficients})
        num_time_samples = 79
        sigma_w = 0.2 ** 0.5
        trajectory_time = np.arange(0, num_time_samples)
        cfg.update({"trajectory_time": trajectory_time})
        C_w_sqrt = np.dot(sigma_w, np.eye(n))
        cfg.update({"C_w_sqrt": C_w_sqrt})
        cfg.update({"C_w": C_w_sqrt @ C_w_sqrt})
        cfg.update({"k": 2 * n})
        # cfg.update({"num_edges_stateinit": 2 * n})

        delta_n_list = np.linspace(1, 9, 5)#np.logspace(-1, 0.5, 8)
        # ---------- parallel sweep ----------
        delta_n_dict_list = [None] * len(delta_n_list)  # preserve order
        max_workers = 1#max(1, os.cpu_count() - 1)  # leave one core free

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_performance_vs_change_size, e, cfg, n, num_iterations, active_methods): i
                       for i, e in enumerate(delta_n_list)}
            for fut in as_completed(futures):
                idx = futures[fut]  # original position
                _, result = fut.result()  # (num_edges, runs_linear)
                delta_n_dict_list[idx] = result
    # Save
    with open("performance_vs_change_size_5order_10nodes.pkl", "wb") as f:
        pickle.dump(delta_n_dict_list, f)

    #
    # plot_vs_parameter(sparsity_list, k_dict_list, "mse", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")
    # plot_vs_parameter(sparsity_list, k_dict_list, "eier", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")

    ########################################################################
    ############# - Performance vs. sparsity level  ########################
    ########################################################################
    with simulation("Non-Linear case - Performance vs. sparsity level"):
        num_iterations = 1000
        sigma_w = 0.2 ** 0.5
        # if num_edges <= 30:
        #     cfg["thr1"] = 0.2
        #     cfg["lambda_1"] = 0.2
        # else:
        # cfg["lambda_1"] = 4
        poly_coefficients = np.array(
            [0.0, 1.0, 0.8, 0.6, 0.4, 0.2])  # np.array([0.0, 1.0, 0.75, 0.5, 0.25])##np.array([1.0, 1.0, 1.0, 1.0])#
        cfg.update({"poly_coefficients": poly_coefficients})
        num_time_samples = 79
        trajectory_time = np.arange(0, num_time_samples)
        cfg.update({"trajectory_time": trajectory_time})
        c = np.linspace(1, 1 / n, n)  # start=1, stop=1/n, n points
        C_w_sqrt = np.dot(sigma_w, np.eye(n))  # scipy.linalg.toeplitz(c)#
        cfg.update({"C_w_sqrt": C_w_sqrt})
        cfg.update({"C_w": C_w_sqrt @ C_w_sqrt})
        sparsity_list = np.linspace(10, 50, 5)
        # Run in series
        # sparsity_dict_list = []
        #
        # for num_edges in sparsity_list:
        #     _, result = _performance_vs_sparsity(num_edges, cfg, m, num_iterations, active_methods)
        #     sparsity_dict_list.append(result)
        # ---------- parallel sweep ----------
        sparsity_dict_list = [None] * len(sparsity_list)  # preserve order
        max_workers = 1#max(1, os.cpu_count() - 1)  # leave one core free

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_performance_vs_sparsity, e, cfg, m, num_iterations, active_methods): i
                       for i, e in enumerate(sparsity_list)}
            for fut in as_completed(futures):
                idx = futures[fut]  # original position
                _, result = fut.result()  # (num_edges, runs_linear)
                sparsity_dict_list[idx] = result

        # Save
    with open("performance_vs_sparsity_5order_10nodes.pkl", "wb") as f:
        pickle.dump(sparsity_dict_list, f)

    # plot_vs_parameter(sparsity_list, sparsity_dict_list, "mse", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")
    # plot_vs_parameter(sparsity_list, sparsity_dict_list, "eier", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="Sparsity [%]")
    #
    #
    # plot_vs_parameter(sparsity_list, sparsity_dict_list, "mse", aggregation_func=mean_func, log_format=False, x_label1="Sparsity [%]")
    # plot_vs_parameter(sparsity_list, sparsity_dict_list, "eier", aggregation_func=mean_func, log_format=False, x_label1="Sparsity [%]")

    #########################################################################
    ################# - Performance vs. graph sizes  ########################
    #########################################################################
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





