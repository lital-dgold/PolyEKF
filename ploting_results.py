# -*- coding: utf-8 -*-
import os

import networkx as nx
import matplotlib
import numpy as np
import pickle
import logging
from EKF_modules import num_possible_edges
from power_system_tracking import prepare_cfg
from util_func import (plot_metric, plot_vs_parameter, create_table, mean_func, update_performance_vs_time_data,
                       update_performance_vs_parameter_data)
from constants import (cfg_linear, cfg_non_linear_case1, cfg_non_linear_case2, cfg_non_linear_vs_snr,
                       cfg_non_linear_vs_delta_n, cfg_non_linear_vs_k, cfg_non_linear_vs_sparsity,
                       cfg_non_linear_vs_filter_order, METHODS_ORDER, LABELS, FILE_LINEAR_VS_TIME_V1,
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
                       FILE_NONLINEAR_CASE2_VS_THR)


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
    to_plot_non_linear_case_2_vs_change_sizes = False
    to_plot_n10_vs_poly_order = False
    to_plot_power_data = False
    to_plot_gsp_ekf_vs_mu = True
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
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=False, log_x_axis=False,
                              x_label1="Connection Changes [%]", to_save=True, folder_name=data_folder_name,
                              suffix="connection_change_nonlinear", legend_loc=UL, ylim_lower = 0.01, ylim_upper = 0.35)
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
                              methods_to_plot=METHODS_ORDER, log_format=False,
                              x_label1="Interval between structural changes [time units]", log_x_axis=True, to_save=True,
                              folder_name=data_folder_name, suffix="change_rate_5order_all", legend_loc=UR, ylim_lower = 0.01, ylim_upper = 0.3)
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
                              log_format=False, x_label1="Connected Edges [%]", to_save=True,
                              folder_name=data_folder_name, suffix="sparsity_5order_all", legend_loc=UL, ylim_lower = 0, ylim_upper = 0.35)
            plot_vs_parameter(cfg_non_linear_vs_sparsity["sparsity_list"], sparsity_dict_list, "eier",
                              aggregation_func=mean_func, labels=LABELS, methods_to_plot=METHODS_ORDER,
                              log_format=False, x_label1="Connected Edges [%]", to_save=True,
                              folder_name=data_folder_name, suffix="sparsity_5order_all", legend_loc=(0.9, 0.21), ylim_lower = 0.01, ylim_upper = 0)

        except FileNotFoundError:
            print(f"The file {FILE_NONLINEAR_CASE2_VS_SPARSITY_V1} or {FILE_NONLINEAR_CASE2_VS_SPARSITY_GRLS} does not exist.")
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
                              labels=LABELS, methods_to_plot=METHODS_ORDER, log_format=False, log_x_axis=False,
                              x_label1="Polynomial Order [int]", to_save=True, folder_name=data_folder_name,
                              suffix="n10", legend_loc=UL, ylim_lower = 0, ylim_upper = 0.25)
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
            cfg_power_graph = prepare_cfg()
            plot_metric(cfg_power_graph["trajectory_time"], power_data_results, "mse", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="power", legend_loc=UR)
            plot_metric(cfg_power_graph["trajectory_time"], power_data_results, "f1", labels=LABELS, methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="power", legend_loc=UL)
            plot_metric(cfg_power_graph["trajectory_time"], power_data_results, "eier", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        to_save=True, folder_name=data_folder_name, suffix="power", legend_loc=UR)
            plot_metric(cfg_power_graph["trajectory_time"], power_data_results, "times", labels=LABELS,
                        methods_to_plot=METHODS_ORDER,
                        log_format=True, to_save=True, folder_name=data_folder_name, suffix="power", legend_loc=C)
            # linear_table = create_table(runs_linear, "times", methods_to_plot=METHODS_ORDER)

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
