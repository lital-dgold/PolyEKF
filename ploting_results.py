# -*- coding: utf-8 -*-
import os

import networkx as nx
import matplotlib
import numpy as np
import pickle
import logging
from EKF_modules import num_possible_edges
from util_func import (plot_metric, plot_vs_parameter, create_table, mean_func, update_performance_vs_time_data,
                       update_performance_vs_parameter_data)
from constants import (cfg_linear, cfg_non_linear_case1, cfg_non_linear_case2, cfg_non_linear_vs_snr,
                       cfg_non_linear_vs_delta_n, cfg_non_linear_vs_k, cfg_non_linear_vs_sparsity,
                       cfg_non_linear_vs_filter_order, METHODS_ORDER, LABELS)


logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more detail
    format="%(asctime)s - %(levelname)s - %(message)s"
)
matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc. depending on your setup

if __name__ == "__main__":
    # Informative flags to control which plots are generated
    to_plot_linear_case_vs_time = True  # True
    to_plot_non_linear_case_vs_time = True  # True#True#False
    to_plot_non_linear_case_2_vs_time = True  # True#True
    to_plot_non_linear_case_2_vs_snr = False
    to_plot_non_linear_case_2_vs_sparsity = False  # True#True
    to_plot_non_linear_case_2_vs_delta_n = False  # True#True#False
    to_plot_non_linear_case_2_vs_k = False  # True
    to_plot_non_linear_case_2_vs_change_sizes = False  # True
    to_plot_n10_vs_poly_order = False  # True#True#False
    #########################################################################
    #################### - Performance vs. time Linear case #################
    #########################################################################
    if to_plot_linear_case_vs_time:
        try:
            data_folder_name = "performance_vs_time_linear"
            data_file_name = "runs_linear_data1000MC.pkl"
            full_path = os.path.join(data_folder_name, data_file_name)
            with open(full_path, "rb") as f:
                runs_linear = pickle.load(f)

            plot_metric(cfg_linear["trajectory_time"], runs_linear, "mse", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="linear")
            plot_metric(cfg_linear["trajectory_time"], runs_linear, "f1", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="linear")
            plot_metric(cfg_linear["trajectory_time"], runs_linear, "eier", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="linear")
            plot_metric(cfg_linear["trajectory_time"], runs_linear, "times", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="linear")
            linear_table = create_table(runs_linear, "times", methods_to_plot=METHODS_ORDER)
        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")
    #########################################################################
    ############## - Performance vs. time Non-Linear case 1 #################
    #########################################################################
    if to_plot_non_linear_case_vs_time:
        try:
            data_folder_name = "performance_vs_time_nonlinear_case1"
            data_file_name = "runs_nonlinear_data1000MC_merged.pkl"
            full_path = os.path.join(data_folder_name, data_file_name)
            with open(full_path, "rb") as f:
                run111 = pickle.load(f)

            plot_metric(cfg_non_linear_case1["trajectory_time"], run111,  "mse", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, log_format=True,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1")
            plot_metric(cfg_non_linear_case1["trajectory_time"], run111, "f1", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1")
            plot_metric(cfg_non_linear_case1["trajectory_time"], run111, "eier", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1")
            plot_metric(cfg_non_linear_case1["trajectory_time"], run111, "times", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, log_format=True,
                        to_save=True, folder_name=data_folder_name, suffix="nonlinear_ver1")
            nonlinear_table = create_table(run111, "times", methods_to_plot=METHODS_ORDER)
        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")
    #########################################################################
    ############## - Performance vs. time - Non-Linear case 2 ###############
    #########################################################################
    if to_plot_non_linear_case_2_vs_time:
        try:
            data_folder_name = "performace_vs_time_nonlinear_ver2"
            data_file_name = "runs_nonlinear_data_5order_10nodes_1000MC_results_k2n_all.pkl"
            full_path = os.path.join(data_folder_name, data_file_name)
            with open(full_path, "rb") as f:
                non_linear_case_ver2 = pickle.load(f)
            plot_metric(cfg_non_linear_case2["trajectory_time"], non_linear_case_ver2, "mse", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, log_format=True, to_save=True, folder_name=data_folder_name,
                        suffix="nonlinear_ver2")
            plot_metric(cfg_non_linear_case2["trajectory_time"], non_linear_case_ver2, "f1", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, to_save=True, folder_name=data_folder_name,
                        suffix="nonlinear_ver2")
            plot_metric(cfg_non_linear_case2["trajectory_time"], non_linear_case_ver2, "eier", labels=LABELS,
                        methods_to_plot=METHODS_ORDER, to_save=True, folder_name=data_folder_name,
                        suffix="nonlinear_ver2")
            nonlinear_table = create_table(non_linear_case_ver2, "times", methods_to_plot=METHODS_ORDER)
            print(nonlinear_table)
        except FileNotFoundError:
            print(f"The file {full_path} does not exist.")
    #########################################################################
    ################# - Performance vs. noise level  ########################
    #########################################################################
    if to_plot_non_linear_case_2_vs_snr:
        try:
            data_folder_name = "performance_vs_snr"
            data_file_name = "performance_vs_snr_5order_merged111.pkl"
            full_path = os.path.join(data_folder_name, data_file_name)
            with open(full_path, "rb") as f:
                snr_dict_list = pickle.load(f)

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
            data_folder_name = "performance_vs_delta_n"
            data_file_name = "performance_vs_change_size_5order_10nodes_k3n_100MC.pkl"
            full_path = os.path.join(data_folder_name, data_file_name)
            with open(full_path, "rb") as f:
                delta_n_dict_list = pickle.load(f)
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
            data_folder_name = "performance_vs_k"
            data_file_name = "performance_vs_k_5order_10nodes100MC_order2_scale_merged.pkl"
            full_path = os.path.join(data_folder_name, data_file_name)
            with open(full_path, "rb") as f:
                k_dict_list = pickle.load(f)
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
            data_folder_name = "perforamce_vs_sparsity"
            data_file_name = "performance_vs_sparsity_5order_10nodes100MC_new.pkl"
            full_path = os.path.join(data_folder_name, data_file_name)
            with open(full_path, "rb") as f:
                sparsity_dict_list = pickle.load(f)

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
            data_folder_name = "performance_vs_poly_order"
            data_file_name = "performance_vs_poly_order.pkl"
            full_path = os.path.join(data_folder_name, data_file_name)
            with open(full_path, "rb") as f:
                poly_order_dict_list = pickle.load(f)
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
