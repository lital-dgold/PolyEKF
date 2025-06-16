# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib
import numpy as np
import pickle
import logging
from EKF_modules import num_possible_edges
from util_func import (plot_metric, plot_vs_parameter, create_table, mean_func,
                       update_performance_vs_parameter_data)


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
    to_plot_non_linear_case_2_vs_snr = True
    to_plot_non_linear_case_2_vs_sparsity = False
    to_plot_non_linear_case_2_vs_graph_variation = False
    to_plot_non_linear_case_2_vs_change_sizes = True
    num_time_samples = 159
    sigma_v = 0.1
    sigma_w = 0.1
    sigma_x = 0.5
    n = 20


    def mean_func_without_first_n(table):
        return table[:, n:].mean(axis=1)
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


    C_w_sqrt = np.dot(sigma_w, np.eye(n))
    B = nx.incidence_matrix(G, oriented=True).todense()
    F = np.dot(1, np.eye(m))
    C_u = np.dot(sigma_v ** 2, np.eye(m))
    # C_x = np.dot(sigma_x ** 2, vector2diag(stateInit))
    C_x_missmatch = np.dot(sigma_x ** 2, np.eye(m))
    mu = 1

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
    if to_plot_linear_case_vs_time:
        try:
            file_name = "runs_linear_data.pkl"
                # Load
            with open(file_name, "rb") as f:
                runs_linear = pickle.load(f)

            plot_metric(cfg["trajectory_time"], runs_linear, "mse", log_format=True)
            plot_metric(cfg["trajectory_time"], runs_linear, "f1")
            plot_metric(cfg["trajectory_time"], runs_linear, "eier")
            plot_metric(cfg["trajectory_time"], runs_linear, "times", log_format=True)
            linear_table = create_table(runs_linear, "times")
        except FileNotFoundError:
            print(f"The file {file_name} does not exist.")

    # ------ Non-Linear case -------
    if to_plot_non_linear_case_vs_time:
        try:
            file_name = "runs_nonlinear_data1000MC.pkl"
            # Load
            with open(file_name, "rb") as f:
                run111 = pickle.load(f)
            # runs_nonlinear = add_method(("gsp-ekf", "gsp-istap-0.4", "gsp-istap-0.5", "gsp-istap-0.6", "gsp-istap-0.7", "gsp-istap-0.8", "gsp-istap-0.9", "gsp-istap-1", "gsp-istap-1.1", "gsp-istap-1.2", "gsp-istap-1.3", "gsp-istap-1.4"),# "oracle-delayedCov", "oracle-block"),
            #  cfg, runs_nonlinear)

            plot_metric(cfg["trajectory_time"], run111,  "mse", log_format=True)
            plot_metric(cfg["trajectory_time"], run111, "f1")
            plot_metric(cfg["trajectory_time"], run111, "eier")
            plot_metric(cfg["trajectory_time"], run111, "times", log_format=True)
            nonlinear_table = create_table(run111, "times")
        except FileNotFoundError:
            print(f"The file {file_name} does not exist.")
    #########################################################################
    ######################### - Performance vs. time  #######################
    #########################################################################
    if to_plot_non_linear_case_2_vs_time:
        try:
            file_name = "runs_nonlinear_data_5order.pkl"
            # Load
            with open(file_name, "rb") as f:
                non_linear_case_ver2 = pickle.load(f)

            plot_metric(cfg["trajectory_time"], non_linear_case_ver2, "mse", log_format=True, to_save=True, suffix="nonlinear_ver2")
            plot_metric(cfg["trajectory_time"], non_linear_case_ver2, "f1", to_save=True, suffix="nonlinear_ver2")
            plot_metric(cfg["trajectory_time"], non_linear_case_ver2, "eier", to_save=True, suffix="nonlinear_ver2")
            nonlinear_table = create_table(non_linear_case_ver2, "times")
            print(nonlinear_table)
        except FileNotFoundError:
            print(f"The file {file_name} does not exist.")
    #########################################################################
    ################# - Performance vs. noise level  ########################
    #########################################################################
    if to_plot_non_linear_case_2_vs_snr:
        try:
            orig_file_name = "performance_vs_snr_5order_10nodes100MC_oracle.pkl"
            sigma_w_list = np.logspace(-2, -0.5, 5)
            #
            # orig_file_name = "performance_vs_snr_5order.pkl"
            # new_data_file_name = "performance_vs_snr_5order_oracle.pkl"
            # merged_data_file_name = "performance_vs_snr_5order_merged.pkl"
            # snr_dict_list = update_performance_vs_parameter_data(orig_file_name, new_data_file_name, merged_data_file_name)
            # # orig_file_name = "performance_vs_snr_5order.pkl"
            with open(orig_file_name, "rb") as f:
                snr_dict_list = pickle.load(f)
            # sigma_w_list = np.logspace(-1.5, 1.5, 10)

            # plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "mse", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="sigma_W [dB]", to_save=True, suffix="snr_5order_without_first_n")
            # plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "eier", aggregation_func=mean_func_without_first_n, log_format=False, x_label1="sigma_W [dB]", to_save=True, suffix="snr_5order_without_first_n")

            plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "mse", aggregation_func=mean_func, log_format=False, x_label1="sigma_W [dB]", to_save=True, suffix="snr_5order_all")
            plot_vs_parameter(10 * np.log10(sigma_w_list), snr_dict_list, "eier", aggregation_func=mean_func, log_format=False, x_label1="sigma_W [dB]", to_save=True, suffix="snr_5order_all")
        except FileNotFoundError:
            print(f"The file {orig_file_name} does not exist.")
    #########################################################################
    ############## - Performance vs. sparsity level  ########################
    #########################################################################
    if to_plot_non_linear_case_2_vs_sparsity:
        try:
            orig_file_name = "performance_vs_sparsity_5order.pkl"
            new_data_file_name = "performance_vs_sparsity_5order_oracle.pkl"
            merged_data_file_name = "performance_vs_sparsity_5order_merged.pkl"
            sparsity_dict_list = update_performance_vs_parameter_data(orig_file_name, new_data_file_name,
                                                                 merged_data_file_name)
            # with open(merged_data_file_name, "rb") as f:
            #     sparsity_dict_list = pickle.load(f)

            sparsity_list = np.linspace(10, 30, 5)

            plot_vs_parameter(sparsity_list, sparsity_dict_list, "mse", aggregation_func=mean_func_without_first_n,
                              log_format=False, x_label1="Sparsity [%]", to_save=True, suffix="sparsity_5order_without_first_n")
            plot_vs_parameter(sparsity_list, sparsity_dict_list, "eier", aggregation_func=mean_func_without_first_n,
                              log_format=False, x_label1="Sparsity [%]", to_save=True, suffix="sparsity_5order_without_first_n")

            plot_vs_parameter(sparsity_list, sparsity_dict_list, "mse", aggregation_func=mean_func, log_format=False,
                              x_label1="Sparsity [%]", to_save=True, suffix="sparsity_5order_all")
            plot_vs_parameter(sparsity_list, sparsity_dict_list, "eier", aggregation_func=mean_func, log_format=False,
                              x_label1="Sparsity [%]", to_save=True, suffix="sparsity_5order_all")

        except FileNotFoundError:
            print(f"The file {file_name} does not exist.")
    #########################################################################
    ############## - Performance vs. size of graph changes  ##############
    #########################################################################
    if to_plot_non_linear_case_2_vs_change_sizes:
        try:
            orig_file_name = "performance_vs_change_size_5order.pkl"
            new_data_file_name = "performance_vs_change_size_5order_oracle.pkl"
            merged_data_file_name = "performance_vs_change_size_5order_merged.pkl"
            # k_dict_list = update_performance_vs_parameter_data(orig_file_name, new_data_file_name,
            #                                                      merged_data_file_name)
            with open(new_data_file_name, "rb") as f:
                delta_n_dict_list = pickle.load(f)
            delta_n_list = np.logspace(-1, 0.5, 8)

            plot_vs_parameter(10 * np.log10(delta_n_list), delta_n_dict_list, "mse", aggregation_func=mean_func_without_first_n, log_format=False,
                              x_label1="Change Size (xN) [dB]", to_save=True, suffix="change_rate_5order_without_first_n")
            plot_vs_parameter(10 * np.log10(delta_n_list), delta_n_dict_list, "eier", aggregation_func=mean_func_without_first_n, log_format=False,
                              x_label1="Change Size (xN) [dB]", to_save=True, suffix="change_rate_5order_without_first_n")

            plot_vs_parameter(10 * np.log10(delta_n_list), delta_n_dict_list, "mse", aggregation_func=mean_func, log_format=False,
                              x_label1="Change Size (xN) [dB]", to_save=True, suffix="change_rate_5order_all")
            plot_vs_parameter(10 * np.log10(delta_n_list), delta_n_dict_list, "eier", aggregation_func=mean_func, log_format=False,
                              x_label1="Change Size (xN) [dB]", to_save=True, suffix="change_rate_5order_all")
        except FileNotFoundError:
            print(f"The file {merged_data_file_name} does not exist.")


    #########################################################################
    ############## - Performance vs. rate of graph variations  ##############
    #########################################################################
    if to_plot_non_linear_case_2_vs_graph_variation:
        try:
            orig_file_name = "performance_vs_graph_variation_5order.pkl"
            new_data_file_name = "performance_vs_graph_variation_5order_oracle.pkl"
            merged_data_file_name = "performance_vs_graph_variation_5order_merged.pkl"
            # k_dict_list = update_performance_vs_parameter_data(orig_file_name, new_data_file_name,
            #                                                      merged_data_file_name)
            with open(new_data_file_name, "rb") as f:
                k_dict_list = pickle.load(f)
            k_list = np.linspace(0.25, 2, 8)


            plot_vs_parameter(k_list, k_dict_list, "mse", aggregation_func=mean_func_without_first_n, log_format=False,
                              x_label1="Change Rate [N]", to_save=True, suffix="change_rate_5order_without_first_n")
            plot_vs_parameter(k_list, k_dict_list, "eier", aggregation_func=mean_func_without_first_n, log_format=False,
                              x_label1="Change Rate [N]", to_save=True, suffix="change_rate_5order_without_first_n")

            plot_vs_parameter(k_list, k_dict_list, "mse", aggregation_func=mean_func, log_format=False,
                              x_label1="Change Rate [N]", to_save=True, suffix="change_rate_5order_all")
            plot_vs_parameter(k_list, k_dict_list, "eier", aggregation_func=mean_func, log_format=False,
                              x_label1="Change Rate [N]", to_save=True, suffix="change_rate_5order_all")
        except FileNotFoundError:
            print(f"The file {merged_data_file_name} does not exist.")


    a = 5
