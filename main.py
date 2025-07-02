# -*- coding: utf-8 -*-
import multiprocessing
from functools import partial

import networkx as nx

import numpy as np
import scipy.sparse as sp
import pickle

import time
import copy
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from EKF_modules import get_trajectory, one_method_evaluation, num_possible_edges
from util_func import vector2diag, compute_metric_summary, pick_worker_count, plot_vs_parameter, mean_func, plot_metric
from constants import METHOD_REGISTRY, LABELS, METHODS_ORDER
from constants import (cfg_linear, cfg_non_linear_case1, cfg_non_linear_case2, cfg_non_linear_vs_snr,
                       cfg_non_linear_vs_delta_n, cfg_non_linear_vs_k, cfg_non_linear_vs_sparsity,
                       cfg_non_linear_vs_filter_order)

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
        # try:
        start = time.time()
        mse, normalized_mse, f1, eier, normalized_eier, times = one_method_evaluation(
            filt, q_meas, y_meas, pos, conn
        )
        end = time.time()
        elapsed = end - start
        logging.info(f"Run {name}: Execution time = {elapsed:.6f} seconds")
        results[name] = dict(mse=mse, normalized_mse=normalized_mse, f1=f1, eier=eier, normalized_eier=normalized_eier, times=times)

    return results            # {"ekf": {...}, "gsp-ekf": {...}, ...}


def run_monte_carlo_simulation(cfg, num_iter, active_methods):
    return [single_monte_carlo(cfg, active_methods) for _ in range(num_iter)]


def single_monte_carlo_iteration(cfg, active_methods, iterable):
    return single_monte_carlo(cfg, active_methods)


def add_method(list_of_methods, cfg, existing_method_dict):
    new_runs = run_monte_carlo_simulation(cfg, cfg["num_iterations"], list_of_methods)
    new_runs[0].update(existing_method_dict[0])
    return list(new_runs)


def _performance_vs_poly_order(p, base_cfg, num_iterations, active_methods):
    cfg = copy.deepcopy(base_cfg)  # avoid concurrent mutation
    poly_coefficients = np.geomspace(1, 2**p, num=p+1)
    poly_coefficients = 1 / poly_coefficients
    cfg.update({"poly_coefficients": poly_coefficients})
    if p > 7:
        cfg.update({"thr1": 0.1})
    return p, run_monte_carlo_simulation(cfg, num_iterations, active_methods)


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


def _performance_vs_change_size(delta_n, base_cfg, num_iterations, active_methods):
    cfg = copy.deepcopy(base_cfg)  # avoid concurrent mutation
    logging.info(f"change rate={delta_n} * n %\n")
    cfg.update({"delta_n": delta_n})
    if delta_n > 6:
        cfg.update({"thr1": 0.1})
    elif delta_n > 3:
        cfg.update({"thr1": 0.15})
    return delta_n, run_monte_carlo_simulation(cfg, num_iterations, active_methods)


def _performance_vs_change_rate(k, base_cfg, num_iterations, active_methods):
    cfg = copy.deepcopy(base_cfg)  # avoid concurrent mutation
    logging.info(f"change rate={k} %\n")
    cfg.update({"k": int(k)})
    if k < 4:
        cfg.update({"thr1": 0.1})
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
    data_folder_name = "Results"
    active_methods = ("fast-ekf", "gsp-ekf", "oracle-block")#, "change-det")
    # Informative flags to control which plots are generated
    to_plot_linear_case_vs_time = True  # True
    to_plot_non_linear_case_vs_time = True  # True#True#False
    to_plot_non_linear_case_2_vs_time = True  # True#True
    to_plot_non_linear_case_2_vs_snr = True
    to_plot_non_linear_case_2_vs_sparsity = True  # True#True
    to_plot_non_linear_case_2_vs_delta_n = True  # True#True#False
    to_plot_non_linear_case_2_vs_k = False  # True
    to_plot_non_linear_case_2_vs_change_sizes = False  # True
    to_plot_n10_vs_poly_order = False  # True#True#False
    #########################################################################
    #################### - Performance vs. time Linear case #################
    #########################################################################
    if to_plot_linear_case_vs_time:
        try:
            with simulation("Linear case"):
                with multiprocessing.Pool() as pool:
                    func = partial(single_monte_carlo_iteration,cfg_linear, active_methods)
                    runs_linear = pool.map(func, np.arange(cfg_linear["num_iterations"]))

                # Save
                linear_file_name = "runs_linear_data.pkl"
                full_path = os.path.join(data_folder_name, linear_file_name)
                with open(full_path, "wb") as f:
                    pickle.dump(runs_linear, f)

            plot_metric(cfg_linear["trajectory_time"], runs_linear, "mse", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="linear")
            plot_metric(cfg_linear["trajectory_time"], runs_linear, "f1", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="linear")
            plot_metric(cfg_linear["trajectory_time"], runs_linear, "eier", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="linear")
            plot_metric(cfg_linear["trajectory_time"], runs_linear, "times", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="linear")
        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")
    #########################################################################
    ############## - Performance vs. time Non-Linear case 1 #################
    #########################################################################
    if to_plot_non_linear_case_vs_time:
        try:
            with simulation("Non-Linear case"):
                with multiprocessing.Pool() as pool:
                    func = partial(single_monte_carlo_iteration, cfg_non_linear_case1, active_methods)
                    runs_nonlinear = pool.map(func, np.arange(cfg_non_linear_case1["num_iterations"]))
                # Save
                runs_nonlinear_data_fast_ekf_file_name = "runs_nonlinear_data_fast_ekf.pkl"
                full_path = os.path.join(data_folder_name, runs_nonlinear_data_fast_ekf_file_name)
                with open(full_path, "wb") as f:
                    pickle.dump(runs_nonlinear, f)

            plot_metric(cfg_non_linear_case1["trajectory_time"], runs_nonlinear,  "mse", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, log_format=True,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1")
            plot_metric(cfg_non_linear_case1["trajectory_time"], runs_nonlinear, "f1", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1")
            plot_metric(cfg_non_linear_case1["trajectory_time"], runs_nonlinear, "eier", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1")
            plot_metric(cfg_non_linear_case1["trajectory_time"], runs_nonlinear, "times", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, log_format=True,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1")
        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")
    #########################################################################
    ############## - Performance vs. time - Non-Linear case 2 ###############
    #########################################################################
    if to_plot_non_linear_case_2_vs_time:
        try:
            with simulation("Non-Linear case"):
                with multiprocessing.Pool() as pool:
                    func = partial(single_monte_carlo_iteration, cfg_non_linear_case2, active_methods)
                    non_linear_case_ver2 = pool.map(func, np.arange(cfg_non_linear_case2["num_iterations"]))
                # Save
                runs_nonlinear_data_5order_10nodes_1000MC_results_k2n_all_file_name = "runs_nonlinear_data_5order_10nodes_1000MC_results_k2n_all.pkl"
                full_path = os.path.join(data_folder_name, runs_nonlinear_data_5order_10nodes_1000MC_results_k2n_all_file_name)
                with open(full_path, "wb") as f:
                    pickle.dump(non_linear_case_ver2, f)

            plot_metric(cfg_non_linear_case2["trajectory_time"], non_linear_case_ver2, "mse", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, log_format=True, to_save=True, folder_name=data_folder_name,
                        suffix="nonlinear_ver2")
            plot_metric(cfg_non_linear_case2["trajectory_time"], non_linear_case_ver2, "f1", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, to_save=True, folder_name=data_folder_name,
                        suffix="nonlinear_ver2")
            plot_metric(cfg_non_linear_case2["trajectory_time"], non_linear_case_ver2, "eier", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, to_save=True, folder_name=data_folder_name,
                        suffix="nonlinear_ver2")
        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")
    #########################################################################
    ################# - Performance vs. noise level  ########################
    #########################################################################
    if to_plot_non_linear_case_2_vs_snr:
        try:
            with simulation("Non-Linear case - Performance vs. noise level"):
                snr_dict_list = [None] * len(cfg_non_linear_vs_snr["sigma_w_list"])  # preserve order
                max_workers = pick_worker_count()#max(1, os.cpu_count() - 1)  # leave one core free

                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    futures = {exe.submit(_performance_vs_snr, e, cfg_non_linear_vs_snr, cfg_non_linear_vs_snr["n"],
                                          cfg_non_linear_vs_snr["num_iterations"], active_methods): i
                               for i, e in enumerate(cfg_non_linear_vs_snr["sigma_w_list"])}
                    for fut in as_completed(futures):
                        idx = futures[fut]  # original position
                        _, result = fut.result()  # (num_edges, runs_linear)
                        snr_dict_list[idx] = result
            # Save
            performance_vs_snr_5order_10nodes100MC_file_name = "performance_vs_snr_5order_10nodes100MC.pkl"
            full_path = os.path.join(data_folder_name, runs_nonlinear_data_fast_ekf_file_name)
            with open(full_path, "wb") as f:
                pickle.dump(snr_dict_list, f)
            plot_vs_parameter(10 * np.log10(cfg_non_linear_vs_snr["sigma_w_list"]), snr_dict_list, "mse",
                              aggregation_func=mean_func, labels=LABELS, methods_to_plot=METHODS_ORDER,
                              log_format=True, x_label1="sigma_e [dB]", to_save=True, folder_name=data_folder_name,
                              suffix="snr_5order_all")
            plot_vs_parameter(10 * np.log10(cfg_non_linear_vs_snr["sigma_w_list"]), snr_dict_list, "eier",
                              aggregation_func=mean_func, labels=LABELS, methods_to_plot=METHODS_ORDER,
                              log_format=False, x_label1="sigma_e [dB]", to_save=True, folder_name=data_folder_name,
                              suffix="snr_5order_all")
        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")
    #########################################################################
    ############## - Performance vs. rate of graph variations  ##############
    #########################################################################
    if to_plot_non_linear_case_2_vs_delta_n:
        try:
            with simulation("Non-Linear case - Performance vs. graph variation"):
                delta_n_dict_list = [None] * len(cfg_non_linear_vs_delta_n["delta_n_list"])  # preserve order
                max_workers = pick_worker_count()  # max(1, os.cpu_count() - 1)  # leave one core free

                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    futures = {exe.submit(_performance_vs_change_size, e, cfg_non_linear_vs_delta_n,
                                          cfg_non_linear_vs_delta_n["num_iterations"], active_methods): i
                               for i, e in enumerate(cfg_non_linear_vs_delta_n["delta_n_list"])}
                    for fut in as_completed(futures):
                        idx = futures[fut]  # original position
                        _, result = fut.result()  # (num_edges, runs_linear)
                        delta_n_dict_list[idx] = result
            # Save
            performance_vs_change_size_5order_10nodes_k3n_100MC_file_name = "performance_vs_change_size_5order_10nodes_k3n_100MC.pkl"
            full_path = os.path.join(data_folder_name, performance_vs_change_size_5order_10nodes_k3n_100MC_file_name)
            with open(full_path, "wb") as f:
                pickle.dump(delta_n_dict_list, f)

            delta_n_percentage = (100 / cfg_non_linear_vs_delta_n["m"]) * cfg_non_linear_vs_delta_n["delta_n_list"]
            plot_vs_parameter(delta_n_percentage, delta_n_dict_list, "mse", aggregation_func=mean_func,
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=False, log_x_axis=False,
                              x_label1="Connection Changes [%]", to_save=True, folder_name=data_folder_name,
                              suffix="connection_change_nonlinear")
            plot_vs_parameter(delta_n_percentage, delta_n_dict_list, "eier", aggregation_func=mean_func,
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=False, log_x_axis=False,
                              x_label1="Connection Changes [%]", to_save=True, folder_name=data_folder_name,
                              suffix="connection_change_nonlinear")
        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")

    #########################################################################
    ############## - Performance vs. rate of graph variations  ##############
    #########################################################################
    if to_plot_non_linear_case_2_vs_k:
        try:
            with simulation("Non-Linear case - Performance vs. graph variation"):
                k_dict_list = [None] * len(cfg_non_linear_vs_k["k_list"])  # preserve order
                max_workers = pick_worker_count()  # max(1, os.cpu_count() - 1)  # leave one core free

                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    futures = {exe.submit(_performance_vs_change_rate, e, cfg_non_linear_vs_k, cfg_non_linear_vs_k["num_iterations"], active_methods): i
                               for i, e in enumerate(cfg_non_linear_vs_k["k_list"])}
                    for fut in as_completed(futures):
                        idx = futures[fut]  # original position
                        _, result = fut.result()  # (num_edges, runs_linear)
                        k_dict_list[idx] = result
            # Save
            performance_vs_k_5order_10nodes100MC_order2_scale_file_name = "performance_vs_k_5order_10nodes100MC_order2_scale.pkl"
            full_path = os.path.join(data_folder_name,
                                     performance_vs_k_5order_10nodes100MC_order2_scale_file_name)
            with open(full_path, "wb") as f:
                pickle.dump(k_dict_list, f)

            plot_vs_parameter(cfg_non_linear_vs_k["k_list"], k_dict_list, "mse", aggregation_func=mean_func, labels=LABELS,
                              methods_to_plot=METHODS_ORDER, log_format=False,
                              x_label1="Change Rate [time units]", log_x_axis=True, to_save=True,
                              folder_name=data_folder_name, suffix="change_rate_5order_all")
            plot_vs_parameter(cfg_non_linear_vs_k["k_list"], k_dict_list, "eier", aggregation_func=mean_func, labels=LABELS,
                              methods_to_plot=METHODS_ORDER, log_format=False,
                              x_label1="Change Rate [time units]", log_x_axis=True, to_save=True,
                              folder_name=data_folder_name, suffix="change_rate_5order_all")
        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")
    ########################################################################
    ############# - Performance vs. sparsity level  ########################
    ########################################################################
    if to_plot_non_linear_case_2_vs_sparsity:
        try:
            with simulation("Non-Linear case - Performance vs. sparsity level"):
                sparsity_dict_list = [None] * len(cfg_non_linear_vs_sparsity["sparsity_list"])  # preserve order
                max_workers = pick_worker_count()  ##max(1, os.cpu_count() - 1)  # leave one core free

                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    futures = {exe.submit(_performance_vs_sparsity, e, cfg_non_linear_vs_sparsity,
                                          cfg_non_linear_vs_sparsity["m"], cfg_non_linear_vs_sparsity["num_iterations"],
                                          active_methods): i
                               for i, e in enumerate(cfg_non_linear_vs_sparsity["sparsity_list"])}
                    for fut in as_completed(futures):
                        idx = futures[fut]  # original position
                        _, result = fut.result()  # (num_edges, runs_linear)
                        sparsity_dict_list[idx] = result

            # Save
            performance_vs_sparsity_5order_10nodes100MC_new_file_name = "performance_vs_sparsity_5order_10nodes100MC_new.pkl"
            full_path = os.path.join(data_folder_name,
                                     performance_vs_sparsity_5order_10nodes100MC_new_file_name)
            with open(full_path, "wb") as f:
                pickle.dump(sparsity_dict_list, f)

            plot_vs_parameter(cfg_non_linear_vs_sparsity["sparsity_list"], sparsity_dict_list, "mse",
                              aggregation_func=mean_func, labels=LABELS, methods_to_plot=METHODS_ORDER,
                              log_format=False, x_label1="Connected Edges [%]", to_save=True,
                              folder_name=data_folder_name, suffix="sparsity_5order_all")
            plot_vs_parameter(cfg_non_linear_vs_sparsity["sparsity_list"], sparsity_dict_list, "eier",
                              aggregation_func=mean_func, labels=LABELS, methods_to_plot=METHODS_ORDER,
                              log_format=False, x_label1="Connected Edges [%]", to_save=True,
                              folder_name=data_folder_name, suffix="sparsity_5order_all")

        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")
    #########################################################################
    ########################### - Run time vs. poly order  ##################
    #########################################################################
    if to_plot_n10_vs_poly_order:
        try:
            with simulation("Non-Linear case - Performance vs. poly order"):
                poly_order_dict_list = [None] * len(cfg_non_linear_vs_filter_order["p_list"])  # preserve order
                max_workers = pick_worker_count()  # max(1, os.cpu_count() - 1)  # leave one core free

                with ProcessPoolExecutor(max_workers=max_workers) as exe:
                    futures = {exe.submit(_performance_vs_poly_order, e, cfg_non_linear_vs_filter_order,
                                          cfg_non_linear_vs_filter_order["num_iterations"], active_methods): i
                               for i, e in enumerate(cfg_non_linear_vs_filter_order["p_list"])}
                    for fut in as_completed(futures):
                        idx = futures[fut]  # original position
                        _, result = fut.result()  # (num_edges, runs_linear)
                        poly_order_dict_list[idx] = result

            # Save
            performance_vs_poly_order_file_name = "performance_vs_poly_order.pkl"
            full_path = os.path.join(data_folder_name,
                                     performance_vs_poly_order_file_name)
            with open(full_path, "wb") as f:
                pickle.dump(poly_order_dict_list, f)

            plot_vs_parameter(cfg_non_linear_vs_filter_order["p_list"], poly_order_dict_list, "mse", aggregation_func=mean_func,
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=False, log_x_axis=False,
                              x_label1="Polynomial Order [int]", to_save=True, folder_name=data_folder_name,
                              suffix="n10")
            plot_vs_parameter(cfg_non_linear_vs_filter_order["p_list"], poly_order_dict_list, "eier", aggregation_func=mean_func,
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=False, log_x_axis=False,
                              x_label1="Polynomial Order [int]", to_save=True, folder_name=data_folder_name,
                              suffix="n10")
            plot_vs_parameter(cfg_non_linear_vs_filter_order["p_list"], poly_order_dict_list, "times", aggregation_func=mean_func,
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=True, log_x_axis=False,
                              x_label1="Polynomial Order [int]", to_save=True, folder_name=data_folder_name,
                              suffix="n10")

        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")

    a = 5