# -*- coding: utf-8 -*-
import os

import networkx as nx
import matplotlib
import numpy as np
import pandas as pd
import pickle
import logging
from EKF_modules import num_possible_edges
from power_system_tracking import prepare_cfg
from util_func import (plot_metric, plot_vs_parameter, create_table, mean_func, update_performance_vs_time_data,
                       update_performance_vs_parameter_data, load_method_files, local_dataset_dirs,
                       resolve_index_mappings, build_aligned_runs, topology_change_counts, select_trajectories,
                       topology_change_times)
from constants import (cfg_linear, cfg_non_linear_case1, cfg_non_linear_case2, cfg_non_linear_vs_snr,
                       cfg_non_linear_vs_delta_n, cfg_non_linear_vs_k, cfg_non_linear_vs_sparsity,
                       cfg_non_linear_vs_filter_order, cfg_non_linear_case1_vs_degree_std,
                       FILE_NONLINEAR_CASE1_VS_DEGREE_STD, FOLDER_NONLINEAR_CASE1_VS_DEGREE_STD,
                       FILE_NONLINEAR_CASE1_VS_DEGREE_STD_GRLS, FILE_NONLINEAR_CASE1_VS_DEGREE_STD_CHANGE_DET,
                       METHODS_ORDER, LABELS, FILE_LINEAR_VS_TIME_V1,
                       FILE_LINEAR_VS_TIME_GRLS, FILE_LINEAR_VS_TIME_PROB_SSM, FOLDER_LINEAR_VS_TIME,
                       FOLDER_NONLINEAR_CASE1_VS_TIME,
                       FILE_NONLINEAR_CASE1_VS_TIME_V1, FILE_NONLINEAR_CASE1_VS_TIME_GRLS,
                       FILE_NONLINEAR_CASE1_VS_TIME_PROB_SSM,
                       FILE_NONLINEAR_CASE2_VS_TIME_V1,
                       FILE_NONLINEAR_CASE2_VS_TIME_GRLS, FILE_NONLINEAR_CASE2_VS_TIME_PROB_SSM,
                       FOLDER_NONLINEAR_CASE2_VS_TIME, FOLDER_NONLINEAR_CASE2_VS_SNR,
                       FILE_NONLINEAR_CASE2_VS_SNR_V1, FILE_NONLINEAR_CASE2_VS_SNR_GRLS,
                       FILE_NONLINEAR_CASE2_VS_SNR_PROB_SSM, FOLDER_NONLINEAR_CASE2_VS_DELTA_N,
                       FILE_NONLINEAR_CASE2_VS_DELTA_N_V1, FILE_NONLINEAR_CASE2_VS_DELTA_N_GRLS,
                       FILE_NONLINEAR_CASE2_VS_DELTA_N_PROB_SSM, FOLDER_NONLINEAR_CASE2_VS_K,
                       FILE_NONLINEAR_CASE2_VS_K_V1, FILE_NONLINEAR_CASE2_VS_K_GRLS, FILE_NONLINEAR_CASE2_VS_K_PROB_SSM,
                       FOLDER_NONLINEAR_CASE2_VS_SPARSITY, FILE_NONLINEAR_CASE2_VS_SPARSITY_V1,
                       FILE_NONLINEAR_CASE2_VS_SPARSITY_GRLS,
                       FILE_NONLINEAR_CASE2_VS_SPARSITY_PROB_SSM, FILE_N10_VS_POLY_ORDER_GRLS,
                       FILE_N10_VS_POLY_ORDER_PROB_SSM, FILE_N10_VS_POLY_ORDER_V1,
                       FOLDER_N10_VS_POLY_ORDER, FOLDER_POWER_DATA, FILE_POWER_DATA_ORACLE, FILE_POWER_DATA_GRLS,
                       FILE_POWER_DATA_CHANGE_DET, FILE_POWER_DATA_GSP_EKF, FILE_POWER_DATA_EKF, UR, LR, C, UL, CL, LL,
                       CR, FOLDER_PERFORMANCE_VS_THR, cfg_linear_vs_thr, cfg_non_linear_case1_vs_thr,
                       cfg_non_linear_case2_vs_thr, FILE_LINEAR_VS_THR, FILE_NONLINEAR_CASE1_VS_THR,
                       FILE_NONLINEAR_CASE2_VS_THR, FOLDER_POWER_DATA_RTE7000, PROVENANCE_KEY, DATASET_GLOB)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc. depending on your setup

if __name__ == "__main__":
    # Informative flags to control which plots are generated
    to_plot_linear_case_vs_time = False
    to_plot_non_linear_case_vs_time = False
    to_plot_non_linear_case_2_vs_time = False
    to_plot_non_linear_case_2_vs_snr = False
    to_plot_non_linear_case_2_vs_sparsity = False
    to_plot_non_linear_case_2_vs_delta_n = False
    to_plot_non_linear_case_2_vs_k = False
    to_plot_n10_vs_poly_order = False
    to_plot_non_linear_case1_vs_degree_std = False
    to_plot_power_data = False
    to_plot_rte7000_power_data = True
    to_plot_gsp_ekf_vs_mu = False
    #########################################################################
    #################### - Performance vs. time Linear case #################
    #########################################################################
    if to_plot_linear_case_vs_time:
        try:
            data_folder_name = FOLDER_LINEAR_VS_TIME
            runs_linear = update_performance_vs_time_data(FILE_LINEAR_VS_TIME_V1, FILE_LINEAR_VS_TIME_GRLS)
            runs_linear = update_performance_vs_time_data(runs_linear, FILE_LINEAR_VS_TIME_PROB_SSM)
            plot_metric(cfg_linear["trajectory_time"], runs_linear, "mse", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="linear",  legend_loc=UR)
            plot_metric(cfg_linear["trajectory_time"], runs_linear, "f1", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="linear", legend_loc=LR)
            plot_metric(cfg_linear["trajectory_time"], runs_linear, "eier", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="linear", legend_loc=UR)
            plot_metric(cfg_linear["trajectory_time"], runs_linear, "times", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="linear", legend_loc=C)
            linear_table = create_table(runs_linear, "times", methods_to_plot=METHODS_ORDER)
        except FileNotFoundError:
            print(f"The file {FILE_LINEAR_VS_TIME_V1} or {FILE_LINEAR_VS_TIME_GRLS} {FILE_LINEAR_VS_TIME_PROB_SSM}does not exist.")
    #########################################################################
    ############## - Performance vs. time Non-Linear case 1 #################
    #########################################################################
    if to_plot_non_linear_case_vs_time:
        try:
            data_folder_name = FOLDER_NONLINEAR_CASE1_VS_TIME
            run111 = update_performance_vs_time_data(FILE_NONLINEAR_CASE1_VS_TIME_V1, FILE_NONLINEAR_CASE1_VS_TIME_GRLS)
            # run111 = update_performance_vs_time_data(run111, FILE_NONLINEAR_CASE1_VS_TIME_PROB_SSM)

            plot_metric(cfg_non_linear_case1["trajectory_time"], run111,  "mse", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, log_format=True,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1", legend_loc=UR)
            plot_metric(cfg_non_linear_case1["trajectory_time"], run111, "f1", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1", legend_loc=LR)
            plot_metric(cfg_non_linear_case1["trajectory_time"], run111, "eier", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1", legend_loc=UR)
            plot_metric(cfg_non_linear_case1["trajectory_time"], run111, "times", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, log_format=True,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1", legend_loc=C)
            nonlinear_table = create_table(run111, "times", methods_to_plot=METHODS_ORDER)
        except FileNotFoundError:
            print(f"The file {FILE_NONLINEAR_CASE1_VS_TIME_V1} or {FILE_NONLINEAR_CASE1_VS_TIME_GRLS} does not exist.")
    #########################################################################
    ############## - Performance vs. time - Non-Linear case 2 ###############
    #########################################################################
    if to_plot_non_linear_case_2_vs_time:
        try:
            data_folder_name = FOLDER_NONLINEAR_CASE2_VS_TIME
            non_linear_case_ver2 = update_performance_vs_time_data(FILE_NONLINEAR_CASE2_VS_TIME_V1, FILE_NONLINEAR_CASE2_VS_TIME_GRLS)
            # non_linear_case_ver2 = update_performance_vs_time_data(non_linear_case_ver2, FILE_NONLINEAR_CASE2_VS_TIME_PROB_SSM)

            plot_metric(cfg_non_linear_case2["trajectory_time"], non_linear_case_ver2, "mse", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, log_format=True, to_save=True, folder_name=data_folder_name,
                        suffix="nonlinear_ver2", legend_loc=UR, ylim_lower = 0, ylim_upper = 0.25)
            plot_metric(cfg_non_linear_case2["trajectory_time"], non_linear_case_ver2, "f1", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, to_save=True, folder_name=data_folder_name,
                        suffix="nonlinear_ver2", legend_loc=LR, ylim_lower = 0.1, ylim_upper = 0)
            plot_metric(cfg_non_linear_case2["trajectory_time"], non_linear_case_ver2, "eier", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, to_save=True, folder_name=data_folder_name,
                        suffix="nonlinear_ver2", legend_loc=UR)
            nonlinear_table = create_table(non_linear_case_ver2, "times", methods_to_plot=METHODS_ORDER)
            print(nonlinear_table)
        except FileNotFoundError:
            print(f"The file {FILE_NONLINEAR_CASE2_VS_TIME_V1} or {FILE_NONLINEAR_CASE2_VS_TIME_GRLS} does not exist.")
    #########################################################################
    ################# - Performance vs. noise level  ########################
    #########################################################################
    if to_plot_non_linear_case_2_vs_snr:
        try:
            data_folder_name = FOLDER_NONLINEAR_CASE2_VS_SNR
            snr_dict_list = update_performance_vs_parameter_data(FILE_NONLINEAR_CASE2_VS_SNR_V1, FILE_NONLINEAR_CASE2_VS_SNR_GRLS)
            # snr_dict_list = update_performance_vs_parameter_data(snr_dict_list, FILE_NONLINEAR_CASE2_VS_SNR_PROB_SSM)


            plot_vs_parameter(10 * np.log10(cfg_non_linear_vs_snr["sigma_w_list"]), snr_dict_list, "mse",
                              aggregation_func=mean_func, labels=LABELS, methods_to_plot=METHODS_ORDER,
                              log_format=True, x_label1="sigma_e [dB]", to_save=True, folder_name=data_folder_name,
                              suffix="snr_5order_all", legend_loc=UL, ylim_lower = 0, ylim_upper = 0.3)
            plot_vs_parameter(10 * np.log10(cfg_non_linear_vs_snr["sigma_w_list"]), snr_dict_list, "eier",
                              aggregation_func=mean_func, labels=LABELS, methods_to_plot=METHODS_ORDER,
                              log_format=False, x_label1="sigma_e [dB]", to_save=True, folder_name=data_folder_name,
                              suffix="snr_5order_all", legend_loc=UL, ylim_lower = 0.01, ylim_upper = 0.35)
        except FileNotFoundError:
            print(f"The file {FILE_NONLINEAR_CASE2_VS_SNR_V1} or {FILE_NONLINEAR_CASE2_VS_SNR_GRLS} does not exist.")
    #########################################################################
    ############## - Performance vs. rate of graph variations  ##############
    #########################################################################
    if to_plot_non_linear_case_2_vs_delta_n:
        try:
            data_folder_name = FOLDER_NONLINEAR_CASE2_VS_DELTA_N
            delta_n_dict_list = update_performance_vs_parameter_data(FILE_NONLINEAR_CASE2_VS_DELTA_N_V1,
                                                            FILE_NONLINEAR_CASE2_VS_DELTA_N_GRLS)
            delta_n_dict_list = update_performance_vs_parameter_data(delta_n_dict_list,
                                                            "Results\\GSP-EKF\\nonlinear_case2_vs_delta_n.pkl")
            delta_n_percentage = (100 / cfg_non_linear_vs_delta_n["m"]) * cfg_non_linear_vs_delta_n["delta_n_list"]
            plot_vs_parameter(delta_n_percentage, delta_n_dict_list, "mse", aggregation_func=mean_func,
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=True, log_x_axis=False,
                              x_label1="Connection Changes [%]", to_save=True, folder_name=data_folder_name,
                              suffix="connection_change_nonlinear", legend_loc=UL, ylim_lower = 0.01, ylim_upper = 0.5)
            plot_vs_parameter(delta_n_percentage, delta_n_dict_list, "eier", aggregation_func=mean_func,
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=False, log_x_axis=False,
                              x_label1="Connection Changes [%]", to_save=True, folder_name=data_folder_name,
                              suffix="connection_change_nonlinear", legend_loc=(0.9, 0.17), ylim_lower = 0, ylim_upper = 0)
        except FileNotFoundError:
            print(f"The file {FILE_NONLINEAR_CASE2_VS_DELTA_N_V1} or {FILE_NONLINEAR_CASE2_VS_DELTA_N_GRLS} does not exist.")

    #########################################################################
    ############## - Performance vs. rate of graph variations  ##############
    #########################################################################
    if to_plot_non_linear_case_2_vs_k:
        try:
            data_folder_name = FOLDER_NONLINEAR_CASE2_VS_K
            k_dict_list = update_performance_vs_parameter_data(FILE_NONLINEAR_CASE2_VS_K_V1,
                                                            FILE_NONLINEAR_CASE2_VS_K_GRLS)
            k_dict_list = update_performance_vs_parameter_data(k_dict_list,
                                                                     "Results\\GSP-EKF\\nonlinear_case2_vs_k.pkl")


            plot_vs_parameter(cfg_non_linear_vs_k["k_list"], k_dict_list, "mse", aggregation_func=mean_func, labels=LABELS,
                              methods_to_plot=METHODS_ORDER, log_format=True,
                              x_label1="Interval between structural changes [time units]", log_x_axis=True, to_save=True,
                              folder_name=data_folder_name, suffix="change_rate_5order_all", legend_loc=UR,
                              ylim_lower = 0.01, ylim_upper = 0.45)
            plot_vs_parameter(cfg_non_linear_vs_k["k_list"], k_dict_list, "eier", aggregation_func=mean_func, labels=LABELS,
                              methods_to_plot=METHODS_ORDER, log_format=False,
                              x_label1="Interval between structural changes [time units]", log_x_axis=True, to_save=True,
                              folder_name=data_folder_name, suffix="change_rate_5order_all", legend_loc=(0.9, 0.2), ylim_lower = 0, ylim_upper = 0)
        except FileNotFoundError:
            print(f"The file {FILE_NONLINEAR_CASE2_VS_K_V1} or {FILE_NONLINEAR_CASE2_VS_K_GRLS} does not exist.")
    ########################################################################
    ############# - Performance vs. sparsity level  ########################
    ########################################################################
    if to_plot_non_linear_case_2_vs_sparsity:
        try:
            data_folder_name = FOLDER_NONLINEAR_CASE2_VS_SPARSITY
            sparsity_dict_list = update_performance_vs_parameter_data(FILE_NONLINEAR_CASE2_VS_SPARSITY_V1,
                                                            FILE_NONLINEAR_CASE2_VS_SPARSITY_GRLS)
            # sparsity_dict_list = update_performance_vs_parameter_data(sparsity_dict_list,
            #                                                 FILE_NONLINEAR_CASE2_VS_SPARSITY_PROB_SSM)
            plot_vs_parameter(cfg_non_linear_vs_sparsity["sparsity_list"], sparsity_dict_list, "mse",
                              aggregation_func=mean_func, labels=LABELS, methods_to_plot=METHODS_ORDER,
                              log_format=True, x_label1="Connected Edges [%]", to_save=True,
                              folder_name=data_folder_name, suffix="sparsity_5order_all", legend_loc=UL,
                              ylim_lower = 0, ylim_upper = 0.5)
            plot_vs_parameter(cfg_non_linear_vs_sparsity["sparsity_list"], sparsity_dict_list, "eier",
                              aggregation_func=mean_func, labels=LABELS, methods_to_plot=METHODS_ORDER,
                              log_format=False, x_label1="Connected Edges [%]", to_save=True,
                              folder_name=data_folder_name, suffix="sparsity_5order_all", legend_loc=(0.9, 0.21), ylim_lower = 0.01, ylim_upper = 0)

        except FileNotFoundError:
            print(f"The file {FILE_NONLINEAR_CASE2_VS_SPARSITY_V1} or {FILE_NONLINEAR_CASE2_VS_SPARSITY_GRLS} does not exist.")
    #########################################################################
    ###### - Performance vs. Sparsity and Degree STD Non-Linear case 1 ######
    #########################################################################
    if to_plot_non_linear_case1_vs_degree_std:
        try:
            data_folder_name = FOLDER_NONLINEAR_CASE1_VS_DEGREE_STD
            os.makedirs(data_folder_name, exist_ok=True)
            with open(FILE_NONLINEAR_CASE1_VS_DEGREE_STD, "rb") as f:
                results_grid = pickle.load(f)  # {(num_edges, degree_std): runs}

            for extra_file in (FILE_NONLINEAR_CASE1_VS_DEGREE_STD_GRLS,
                              FILE_NONLINEAR_CASE1_VS_DEGREE_STD_CHANGE_DET):
                with open(extra_file, "rb") as f:
                    extra_results_grid = pickle.load(f)
                for key, runs in results_grid.items():
                    for run, extra_run in zip(runs, extra_results_grid[key]):
                        run.update(extra_run)

            cfg = cfg_non_linear_case1_vs_degree_std
            num_edges_values = cfg["num_edges_values"]
            degree_std_values = cfg["degree_std_values"]

            # Table: the grid looks fairly flat, so summarize each grid point
            # as one row (mean over time and Monte-Carlo runs per method)
            # instead of relying on the curves alone.
            mse_in_db = True  # set True to report MSE in dB (10*log10), matching e.g. the sparsity plots above
            for metric in ("mse", "eier"):
                log_format = mse_in_db and metric == "mse"
                rows = []
                for num_edges in num_edges_values:
                    for degree_std in degree_std_values:
                        row = {"num_edges": num_edges, "degree_std_low": degree_std[0], "degree_std_high": degree_std[1]}
                        row.update(create_table(results_grid[(num_edges, degree_std)], metric,
                                                methods_to_plot=METHODS_ORDER, log_format=log_format))
                        rows.append(row)
                table_df = pd.DataFrame(rows)
                metric_label = f"{metric}_db" if log_format else metric
                csv_path = os.path.join(data_folder_name, f"{metric_label}_vs_sparsity_degree_std_table.csv")
                table_df.to_csv(csv_path, index=False)
                print(f"Saved {csv_path}")
                print(table_df)

        except FileNotFoundError:
            print(f"The file {FILE_NONLINEAR_CASE1_VS_DEGREE_STD} does not exist.")
    #########################################################################
    ########################### - Run time vs. poly order  ##################
    #########################################################################
    if to_plot_n10_vs_poly_order:
        try:
            data_folder_name = FOLDER_N10_VS_POLY_ORDER
            poly_order_dict_list = update_performance_vs_parameter_data(FILE_N10_VS_POLY_ORDER_V1,
                                                            FILE_N10_VS_POLY_ORDER_GRLS)
            poly_order_dict_list = update_performance_vs_parameter_data(poly_order_dict_list,
                                                            FILE_N10_VS_POLY_ORDER_PROB_SSM)

            plot_vs_parameter(cfg_non_linear_vs_filter_order["p_list"], poly_order_dict_list, "mse", aggregation_func=mean_func,
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=True, log_x_axis=False,
                              x_label1="Polynomial Order [int]", to_save=True, folder_name=data_folder_name,
                              suffix="n10", legend_loc=UL, ylim_lower = 0, ylim_upper = 0.45)
            plot_vs_parameter(cfg_non_linear_vs_filter_order["p_list"], poly_order_dict_list, "eier", aggregation_func=mean_func,
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=False, log_x_axis=False,
                              x_label1="Polynomial Order [int]", to_save=True, folder_name=data_folder_name,
                              suffix="n10", legend_loc=UR, ylim_lower = 0.01, ylim_upper = 0.5)
            plot_vs_parameter(cfg_non_linear_vs_filter_order["p_list"], poly_order_dict_list, "times", aggregation_func=mean_func,
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=True, log_x_axis=False,
                              x_label1="Polynomial Order [int]", to_save=True, folder_name=data_folder_name,
                              suffix="n10", legend_loc=(0.1, 0.5), ylim_lower = 0, ylim_upper = 0)

        except FileNotFoundError:
            print(f"The file {FILE_N10_VS_POLY_ORDER_V1} or {FILE_N10_VS_POLY_ORDER_GRLS} or {FILE_N10_VS_POLY_ORDER_PROB_SSM} does not exist.")

    #########################################################################
    ########################### - Power data  ##################
    #########################################################################
    if to_plot_power_data:
        try:
            data_folder_name = FOLDER_POWER_DATA
            # power_data_results = pickle.load(open(FILE_POWER_DATA_ORACLE, "rb"))
            power_data_results = update_performance_vs_time_data(FILE_POWER_DATA_ORACLE,
                                                            FILE_POWER_DATA_GSP_EKF)
            power_data_results = update_performance_vs_time_data(power_data_results,
                                                            FILE_POWER_DATA_EKF)
            power_data_results = update_performance_vs_time_data(power_data_results,
                                                            FILE_POWER_DATA_GRLS)
            power_data_results = update_performance_vs_time_data(power_data_results,
                                                            FILE_POWER_DATA_CHANGE_DET)
            power_data_results = power_data_results[2:]
            trajectory_time = np.arange(0, len(next(iter(power_data_results[0].values()))["mse"]))
            plot_metric(trajectory_time, power_data_results, "mse", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="power", legend_loc=UR)
            plot_metric(trajectory_time, power_data_results, "f1", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="power", legend_loc=UL)
            plot_metric(trajectory_time, power_data_results, "eier", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="power", legend_loc=UR)
            plot_metric(trajectory_time, power_data_results, "times", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="power", legend_loc=C)
            # linear_table = create_table(runs_linear, "times", methods_to_plot=METHODS_ORDER)

        except FileNotFoundError:
            print(f"The file {FILE_N10_VS_POLY_ORDER_V1} or {FILE_N10_VS_POLY_ORDER_GRLS} or {FILE_N10_VS_POLY_ORDER_PROB_SSM} does not exist.")


    #########################################################################
    ########################### - Power data  ##################
    #########################################################################
    if to_plot_rte7000_power_data:
        try:
            data_folder_name = FOLDER_POWER_DATA_RTE7000
            trajectory_mapping_key = PROVENANCE_KEY
            N_TRAJECTORIES = 1
            print(f"Loading method result files from {FOLDER_POWER_DATA_RTE7000}:")
            methods = load_method_files(FOLDER_POWER_DATA_RTE7000)
            idx_to_dir = local_dataset_dirs(DATASET_GLOB)

            print("\nResolving per-method index alignment (embedded provenance):")
            mappings = resolve_index_mappings(methods, trajectory_mapping_key)

            runs = build_aligned_runs(methods, mappings)
            methods_found = sorted(mappings.keys())
            excluded = sorted(set(methods.keys()) - set(mappings.keys()))
            print(f"\nAligned methods present: {methods_found}")
            if excluded:
                print(f"Excluded (missing/partial provenance): {excluded}")

            counts = topology_change_counts(idx_to_dir)
            if not counts:
                raise RuntimeError(f"No local dataset folders matching {DATASET_GLOB} with line_topology.npy found.")

            eligible = {i: c for i, c in counts.items() if i < len(runs) and runs[i]}
            if not eligible:
                raise RuntimeError("No trajectory index has both local topology data and result data.")

            selected = select_trajectories(eligible, N_TRAJECTORIES)
            print(f"\nSelected trajectory indices (most topology changes): {selected}")
            for i in selected:
                print(f"  idx {i}: {eligible[i]} topology changes over the trajectory")

            methods_order_present = [m for m in METHODS_ORDER if m in methods_found]

            summary_rows = []
            for idx in selected:
                run_dict = runs[idx]
                methods_here = [m for m in methods_order_present if m in run_dict]
                T = len(next(iter(run_dict.values()))["mse"])
                time = np.arange(T)
                single_run = [run_dict]
                traj_suffix = f"rte7000_power_traj{idx}"
                # Vertical markers at this trajectory's own topology-change steps
                change_times = topology_change_times(idx_to_dir[idx])

                plot_metric(time, single_run, "mse", labels=LABELS,
                            methods_to_plot=METHODS_ORDER,
                            log_format=True, to_save=True, folder_name=data_folder_name, suffix=traj_suffix,
                            legend_loc=UR, ylabel_metric="NSE", change_times=change_times, ylim_lower = 0, ylim_upper = 0.2)
                plot_metric(time, single_run, "f1", labels=LABELS, methods_to_plot=METHODS_ORDER,
                            to_save=True, folder_name=data_folder_name, suffix=traj_suffix, legend_loc=UL,
                            change_times=change_times)
                plot_metric(time, single_run, "eier", labels=LABELS,
                            methods_to_plot=METHODS_ORDER,
                            to_save=True, folder_name=data_folder_name, suffix=traj_suffix, legend_loc=LL,
                            change_times=change_times, ylim_lower = 0.01, ylim_upper = 0.15)

            all_runs = [r for r in runs if any(m in r for m in methods_order_present)]
            # sample_entry = all_runs[0]
            # sample_method = next(m for m in methods_order_present if m in sample_entry)
            plot_metric(time, all_runs, "mse", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="rte7000_power_all",
                        legend_loc=UR, ylim_lower = 0, ylim_upper = 0.1)
            plot_metric(time, all_runs, "f1", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="rte7000_power_all", legend_loc=UL)
            plot_metric(time, all_runs, "eier", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="rte7000_power_all", legend_loc=(0.56, 0.4))


        except FileNotFoundError:
            print(f"The file {FILE_N10_VS_POLY_ORDER_V1} or {FILE_N10_VS_POLY_ORDER_GRLS} or {FILE_N10_VS_POLY_ORDER_PROB_SSM} does not exist.")


    #########################################################################
    ########################### - Performance vs mu  ##################
    #########################################################################
    if to_plot_gsp_ekf_vs_mu:
        data_folder_name = FOLDER_PERFORMANCE_VS_THR
        suffix_list = ("linear_vs_thr", "nonlinear_case1_vs_thr", "nonlinear_case2_vs_thr")
        cfg_vs_thr_list = (cfg_linear_vs_thr, cfg_non_linear_case1_vs_thr, cfg_non_linear_case2_vs_thr)
        file_names = (FILE_LINEAR_VS_THR, FILE_NONLINEAR_CASE1_VS_THR, FILE_NONLINEAR_CASE2_VS_THR)
        for cfg_vs_thr, suffix, file_name in zip(cfg_vs_thr_list, suffix_list, file_names):
            try:
                thr_dict_list = update_performance_vs_parameter_data(file_name)

                plot_vs_parameter(cfg_vs_thr["thr_list"], thr_dict_list, "mse",
                                  aggregation_func=mean_func,
                                  labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=True, log_x_axis=False,
                                  x_label1="Threshold", to_save=True, folder_name=data_folder_name, suffix=suffix)
                plot_vs_parameter(cfg_vs_thr["thr_list"], thr_dict_list, "eier",
                                  aggregation_func=mean_func,
                                  labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=False, log_x_axis=False,
                                  x_label1="Threshold", to_save=True, folder_name=data_folder_name, suffix=suffix)

            except FileNotFoundError:
                print(f"The file {file_name} does not exist.")
    a = 5
