from EKF_modules import (ExtendedKalmanFilter, FastExtendedKalmanFilter, sparseKalmanFilter, sparseKalmanFilterISTA,
                         oraclKalmanFilt_paper)
from baseline_grls import GRLSSimple
from baseline_online_pred_corr import OnlinePredCorrEstimator
from baseline_prob_ssm import ProbSSMBaseline
from change_detection_module import ChangeDetectionMethod
from baseline_laplacian_lms import LaplacianLMSConstrained
import networkx as nx
import numpy as np

LABELS = {"change-det": "Change-det", "oracle-block": "Oracle", "ekf": "EKF", "fast-ekf": "EKF",
          "gsp-ekf": "GSP-EKF", "lms": "Lap-LMS", "grls": "GRLS", "pred_corr": "TV-GT", "prob_ssm": "ProbSSM"}
METHODS_ORDER = ["prob_ssm", "pred_corr", "lms", "grls", "change-det", "oracle-block", "ekf", "fast-ekf", "gsp-ekf", ]

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
    # ----- Sparse EKF with ISTA refinement ---------------------------------
    "gsp-istap": lambda cfg: sparseKalmanFilterISTA(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=0.01  # tweak as needed
    ),

    # ----- Oracle variants --------------------------------------------------
    "oracle-block": lambda cfg: oraclKalmanFilt_paper(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x"], cfg["stateInit"], cfg["poly_coefficients"], cfg["new_edge_weight"]
    ),
    # ----- Change-detection baseline ---------------------------------------
    "change-det": lambda cfg: ChangeDetectionMethod(
        cfg["B"], cfg["window_len"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=cfg["lambda_1"], lambda_2=cfg["lambda_2"]
    ),

    # ----- Laplacian LMS baseline (Algorithm 1) ----------------------------
    "lms": lambda cfg: LaplacianLMSConstrained(
        cfg["B"], mu=cfg.get("lms_mu", 1.0), T=cfg.get("lms_T", 1.0)
    ),
    # -----
    "grls": lambda cfg: GRLSSimple(
        cfg["B"], cfg["poly_coefficients"], cfg["cov_init"], cfg["stateInit_weight_grls"], beta=cfg["beta"],
            alpha=cfg["alpha"],
            lambda_r=cfg["lambda_r"],
            t_max=cfg["t_max"]
),

    "pred_corr": lambda cfg: OnlinePredCorrEstimator(
            cfg["B"], cfg["poly_coefficients"],
            gamma=0.9,
            alpha=0.1,
            beta=0.1,
            h=1.0,
            P=1,
            C=1,
        ),
    "prob_ssm": lambda cfg: ProbSSMBaseline(
    cfg["B"], cfg["poly_coefficients"], cfg["p_e"], cfg["sigma_prob_ssm"],  cfg["w_ref"],  cfg["state_est"],
    ),
}

#########################################################################
################## - Results folder and file names  #####################
#########################################################################
import os as _os
RESULTS_DIR = "Results"
FILE_LINEAR_VS_TIME = "linear_vs_time.pkl"
FILE_NONLINEAR_CASE1_VS_TIME = "nonlinear_case1_vs_time.pkl"
FILE_NONLINEAR_CASE2_VS_TIME = "nonlinear_case2_vs_time.pkl"
FILE_NONLINEAR_CASE2_VS_SNR = "nonlinear_case2_vs_snr.pkl"
FILE_NONLINEAR_CASE2_VS_DELTA_N = "nonlinear_case2_vs_delta_n.pkl"
FILE_NONLINEAR_CASE2_VS_K = "nonlinear_case2_vs_k.pkl"
FILE_NONLINEAR_CASE2_VS_SPARSITY = "nonlinear_case2_vs_sparsity.pkl"
FILE_NONLINEAR_CASE1_VS_DEGREE_STD = "nonlinear_case1_vs_degree_std.pkl"
FILE_N10_VS_POLY_ORDER = "n10_vs_poly_order.pkl"
RESULTS_DIR_GRLS = "Results\\GRLS"

# ── Performance vs. time ─────────────────────────────────────────────
FILE_LINEAR_VS_TIME_GRLS             = _os.path.join(RESULTS_DIR_GRLS, FILE_LINEAR_VS_TIME)
FILE_NONLINEAR_CASE1_VS_TIME_GRLS    = _os.path.join(RESULTS_DIR_GRLS, FILE_NONLINEAR_CASE1_VS_TIME)
FILE_NONLINEAR_CASE2_VS_TIME_GRLS    = _os.path.join(RESULTS_DIR_GRLS, FILE_NONLINEAR_CASE2_VS_TIME)

# ── Performance vs. parameter ─────────────────────────────────────────
FILE_NONLINEAR_CASE2_VS_SNR_GRLS      = _os.path.join(RESULTS_DIR_GRLS, FILE_NONLINEAR_CASE2_VS_SNR)
FILE_NONLINEAR_CASE2_VS_DELTA_N_GRLS  = _os.path.join(RESULTS_DIR_GRLS, FILE_NONLINEAR_CASE2_VS_DELTA_N)
FILE_NONLINEAR_CASE2_VS_K_GRLS        = _os.path.join(RESULTS_DIR_GRLS, FILE_NONLINEAR_CASE2_VS_K)
FILE_NONLINEAR_CASE2_VS_SPARSITY_GRLS = _os.path.join(RESULTS_DIR_GRLS, FILE_NONLINEAR_CASE2_VS_SPARSITY)
FILE_NONLINEAR_CASE1_VS_DEGREE_STD_GRLS = _os.path.join(RESULTS_DIR_GRLS, FILE_NONLINEAR_CASE1_VS_DEGREE_STD)
FILE_N10_VS_POLY_ORDER_GRLS           = _os.path.join(RESULTS_DIR_GRLS, FILE_N10_VS_POLY_ORDER)


RESULTS_DIR_PROB_SSM = "Results\\PROB_SSM"

# ── Performance vs. time ─────────────────────────────────────────────
FILE_LINEAR_VS_TIME_PROB_SSM             = _os.path.join(RESULTS_DIR_PROB_SSM, FILE_LINEAR_VS_TIME)
FILE_NONLINEAR_CASE1_VS_TIME_PROB_SSM    = _os.path.join(RESULTS_DIR_PROB_SSM, FILE_NONLINEAR_CASE1_VS_TIME)
FILE_NONLINEAR_CASE2_VS_TIME_PROB_SSM    = _os.path.join(RESULTS_DIR_PROB_SSM, FILE_NONLINEAR_CASE2_VS_TIME)

# ── Performance vs. parameter ─────────────────────────────────────────
FILE_NONLINEAR_CASE2_VS_SNR_PROB_SSM      = _os.path.join(RESULTS_DIR_PROB_SSM, FILE_NONLINEAR_CASE2_VS_SNR)
FILE_NONLINEAR_CASE2_VS_DELTA_N_PROB_SSM  = _os.path.join(RESULTS_DIR_PROB_SSM, FILE_NONLINEAR_CASE2_VS_DELTA_N)
FILE_NONLINEAR_CASE2_VS_K_PROB_SSM        = _os.path.join(RESULTS_DIR_PROB_SSM, FILE_NONLINEAR_CASE2_VS_K)
FILE_NONLINEAR_CASE2_VS_SPARSITY_PROB_SSM = _os.path.join(RESULTS_DIR_PROB_SSM, FILE_NONLINEAR_CASE2_VS_SPARSITY)
FILE_N10_VS_POLY_ORDER_PROB_SSM           = _os.path.join(RESULTS_DIR_PROB_SSM, FILE_N10_VS_POLY_ORDER)

# Current file names
# ── Performance vs. time ─────────────────────────────────────────────
FOLDER_LINEAR_VS_TIME = "Results\\performance_vs_time_linear"
FILE_LINEAR_VS_TIME_V1 =  _os.path.join(FOLDER_LINEAR_VS_TIME, "runs_linear_data1000MC.pkl")

FOLDER_NONLINEAR_CASE1_VS_TIME = "Results\\performance_vs_time_nonlinear_case1"
FILE_NONLINEAR_CASE1_VS_TIME_V1 =  _os.path.join(FOLDER_NONLINEAR_CASE1_VS_TIME, "runs_nonlinear_data1000MC_merged.pkl")
##
FOLDER_NONLINEAR_CASE2_VS_TIME = "Results\\performance_vs_time_nonlinear_case2"
FILE_NONLINEAR_CASE2_VS_TIME_V1 =  _os.path.join(FOLDER_NONLINEAR_CASE2_VS_TIME, "runs_nonlinear_data_5order_10nodes_1000MC_results_k2n_all.pkl")

# ── Performance vs. parameter ─────────────────────────────────────────
FOLDER_NONLINEAR_CASE2_VS_SNR = "Results\\performance_vs_snr"
FILE_NONLINEAR_CASE2_VS_SNR_V1 =  _os.path.join(FOLDER_NONLINEAR_CASE2_VS_SNR, "performance_vs_snr_5order_merged111.pkl")

FOLDER_NONLINEAR_CASE2_VS_DELTA_N = "Results\\performance_vs_delta_n"
FILE_NONLINEAR_CASE2_VS_DELTA_N_V1 =  _os.path.join(FOLDER_NONLINEAR_CASE2_VS_DELTA_N, "performance_vs_change_size_5order_10nodes_k3n_100MC.pkl")

FOLDER_NONLINEAR_CASE2_VS_K = "Results\\performance_vs_k"
FILE_NONLINEAR_CASE2_VS_K_V1 =  _os.path.join(FOLDER_NONLINEAR_CASE2_VS_K, "performance_vs_k_5order_10nodes100MC_order2_scale_merged.pkl")

FOLDER_NONLINEAR_CASE2_VS_SPARSITY = "Results\\performance_vs_sparsity"
FILE_NONLINEAR_CASE2_VS_SPARSITY_V1 =  _os.path.join(FOLDER_NONLINEAR_CASE2_VS_SPARSITY, "performance_vs_sparsity_5order_10nodes100MC_new.pkl")

FOLDER_NONLINEAR_CASE1_VS_DEGREE_STD = "Results\\performance_vs_degree_std"
FILE_NONLINEAR_CASE1_VS_DEGREE_STD = _os.path.join(FOLDER_NONLINEAR_CASE1_VS_DEGREE_STD, FILE_NONLINEAR_CASE1_VS_DEGREE_STD)
FILE_NONLINEAR_CASE1_VS_DEGREE_STD_CHANGE_DET = _os.path.join(FOLDER_NONLINEAR_CASE1_VS_DEGREE_STD,
                                                              "nonlinear_case1_vs_degree_std_change_det.pkl")

FOLDER_N10_VS_POLY_ORDER = "Results\\performance_vs_poly_order"
FILE_N10_VS_POLY_ORDER_V1 =  _os.path.join(FOLDER_N10_VS_POLY_ORDER, "performance_vs_poly_order.pkl")

FOLDER_PERFORMANCE_VS_THR          = "Results\\gsp_ekf_vs_thr"
FILE_LINEAR_VS_THR            = _os.path.join(FOLDER_PERFORMANCE_VS_THR, "gsp_ekf_vs_thr_linear.pkl")
FILE_NONLINEAR_CASE1_VS_THR   = _os.path.join(FOLDER_PERFORMANCE_VS_THR, "gsp_ekf_vs_thr_nonlinear_case1.pkl")
FILE_NONLINEAR_CASE2_VS_THR   = _os.path.join(FOLDER_PERFORMANCE_VS_THR, "gsp_ekf_vs_thr_nonlinear_case2.pkl")

# -------------- Power data experiment
FOLDER_POWER_DATA = "Results\\Power_data_exp\\Results57"
FILE_POWER_DATA_EKF = _os.path.join(FOLDER_POWER_DATA,  "fast-ekf.pkl")
FILE_POWER_DATA_ORACLE = _os.path.join(FOLDER_POWER_DATA,  "oracle-block.pkl")
FILE_POWER_DATA_GSP_EKF = _os.path.join(FOLDER_POWER_DATA,  "gsp-ekf.pkl")
FILE_POWER_DATA_GRLS = _os.path.join(FOLDER_POWER_DATA,  "grls.pkl")
FILE_POWER_DATA_PROB_SSM = _os.path.join(FOLDER_POWER_DATA,  "prob-ssm.pkl")
FILE_POWER_DATA_CHANGE_DET = _os.path.join(FOLDER_POWER_DATA,  "change-det.pkl")

FOLDER_POWER_DATA_RTE7000 = "Results\\Power_data_exp\\ResultsRTE7000"
DATASET_GLOB = "Power_data/rte7000_ac_syn_dataset*"
PROVENANCE_KEY = "_dataset_folder"
#########################################################################
B = 'best'
UR = 'upper right'
UL = 'upper left'
LR = 'lower right'
LL = 'lower left'
CR = 'center right'
CL = 'center left'
UC = 'upper center'
LC = 'lower center'
C = 'center'
#########################################################################
######################## - 20 nodes graph  ##############################
#########################################################################
cfg_N_20_graph = {
    "num_iterations": 1000,
    "n": 20,
    "new_edge_weight": 1,
    "trajectory_time": np.arange(0, 159),
    "thr1": 0.2,
    "thr2": 0.2,
    "mu": 1,
    "lms_mu": 0.88,
    "lms_T": 11.4,
    "lambda_1": 0.1,
    "lambda_2": 0,
    "delta_n": 1,
    "sigma_v": 0.01,
    "sigma_w": 0.01,
    "sigma_x": 0.5,
}
cfg_N_20_graph.update({
    "B": nx.incidence_matrix(nx.complete_graph(cfg_N_20_graph["n"], create_using=None), oriented=True).todense(),
    "k": int(2 * cfg_N_20_graph["n"]),
    "num_edges_stateinit": int(3 * cfg_N_20_graph["n"]),
    "window_len": cfg_N_20_graph["n"],
})
cfg_N_20_graph.update({"m": cfg_N_20_graph["B"].shape[1],
                       "C_w_sqrt": np.dot(cfg_N_20_graph["sigma_w"], np.eye(cfg_N_20_graph["n"])),
                       })
cfg_N_20_graph.update({
    "F": np.dot(1, np.eye(cfg_N_20_graph["m"])),
    "C_w": cfg_N_20_graph["C_w_sqrt"] @ cfg_N_20_graph["C_w_sqrt"],
    "C_u_sqrt": np.dot(cfg_N_20_graph["sigma_v"], np.eye(cfg_N_20_graph["m"])),
    "C_x_missmatch": np.dot(cfg_N_20_graph["sigma_x"] ** 2, np.eye(cfg_N_20_graph["m"])),
    "stateInit_missmatch": cfg_N_20_graph["new_edge_weight"] * np.ones(cfg_N_20_graph["m"]).reshape([cfg_N_20_graph["m"], 1]),
})
cfg_N_20_graph.update({
    "C_u": cfg_N_20_graph["C_u_sqrt"] @ cfg_N_20_graph["C_u_sqrt"],
})
cfg_N_20_graph.update({
"cov_init": 1.0,
"stateInit_weight_grls": 1.0,
"beta": 0.8,
"alpha": 0.05,
"lambda_r": 0.001,
"t_max": 10,
})
#########################################################################
######################## - 10 nodes graph  ##############################
#########################################################################
cfg_N_10_graph = {
    "num_iterations": 1000,
    "n": 10,
    "poly_coefficients": np.array([0.0, 1.0, 0.8, 0.6, 0.4, 0.2]),
    "new_edge_weight": 1,
    "trajectory_time": np.arange(0, 79),
    "thr1": 0.25,
    "thr2": 0.2,
    "mu": 1,
    "lambda_1": 3.16,
    "lambda_2": 0.316,
    "delta_n": 1,
    "sigma_v": 0.01 ** 0.5,
    "sigma_w": 0.2 ** 0.5,
    "sigma_x": 0.5,
}
cfg_N_10_graph.update({
    "B": nx.incidence_matrix(nx.complete_graph(cfg_N_10_graph["n"], create_using=None), oriented=True).todense(),
    "k": int(2 * cfg_N_10_graph["n"]),
    "num_edges_stateinit": int(1.5 * cfg_N_10_graph["n"]),
    "window_len": int(0.5 * cfg_N_10_graph["n"]),
})
cfg_N_10_graph.update({"m": cfg_N_10_graph["B"].shape[1],
                       "C_w_sqrt": np.dot(cfg_N_10_graph["sigma_w"], np.eye(cfg_N_10_graph["n"])),
                       })

cfg_N_10_graph.update({
    "F": np.dot(1, np.eye(cfg_N_10_graph["m"])),
    "C_u_sqrt": np.dot(cfg_N_10_graph["sigma_v"], np.eye(cfg_N_10_graph["m"])),
    "C_w": cfg_N_10_graph["C_w_sqrt"] @ cfg_N_10_graph["C_w_sqrt"],
    "C_x_missmatch": np.dot(cfg_N_10_graph["sigma_x"] ** 2, np.eye(cfg_N_10_graph["m"])),
    "stateInit_missmatch": cfg_N_10_graph["new_edge_weight"] * np.ones(cfg_N_10_graph["m"]).reshape([cfg_N_10_graph["m"], 1]),
})
cfg_N_10_graph.update({
    "C_u": cfg_N_10_graph["C_u_sqrt"] @ cfg_N_10_graph["C_u_sqrt"],
})
cfg_N_10_graph.update({
"cov_init": 0.0,
"stateInit_weight_grls": 1.0,
"beta": 0.9,
"alpha": 0.01,
"lambda_r": 0.1,
"t_max": 20,
})
#########################################################################
############## - Performance vs. time Linear case  ######################
#########################################################################
cfg_linear = cfg_N_20_graph.copy()
cfg_linear.update({
    "poly_coefficients": np.array([0, 1.0])
})
cfg_linear.update({
"p_e": 0.01,
"sigma_prob_ssm": 0.5,
"w_ref":1.0,
"state_est": "map,"
})
#########################################################################
############ - Performance vs. thr - Linear case (GSP-EKF) #############
#########################################################################
cfg_linear_vs_thr = cfg_linear.copy()
cfg_linear_vs_thr.update({
    "thr_list": np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]),
    "num_iterations": 100,
})
#########################################################################
############## - Performance vs. time Non-Linear case 1 #################
#########################################################################
cfg_non_linear_case1 = cfg_N_20_graph.copy()
cfg_non_linear_case1.update({
    "poly_coefficients": np.array([1.0, 1.0, 0.1, 1.0])
})
#########################################################################
########## - Performance vs. thr - Non-Linear case 1 (GSP-EKF) #########
#########################################################################
cfg_non_linear_case1_vs_thr = cfg_non_linear_case1.copy()
cfg_non_linear_case1_vs_thr.update({
    "thr_list": np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]),
    "num_iterations": 100,
})
cfg_non_linear_case1.update({
"cov_init": 0.0,
"stateInit_weight_grls": 1.0,
"beta": 0.9,
"alpha": 0.01,
"lambda_r": 0.001,
"t_max": 20,
})
#########################################################################
############## - Performance vs. time - Non-Linear case 2 ###############
#########################################################################
cfg_non_linear_case2 = cfg_N_10_graph.copy()
cfg_non_linear_case2.update({
    "num_iterations": 1000,
})
#########################################################################
######## - Performance vs. thr - Non-Linear case 2 (GSP-EKF) ###########
#########################################################################
cfg_non_linear_case2_vs_thr = cfg_non_linear_case2.copy()
cfg_non_linear_case2_vs_thr.update({
    "thr_list": np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]),
    "num_iterations": 100,
})
#########################################################################
################# - Performance vs. noise level  ########################
#########################################################################
cfg_non_linear_vs_snr = cfg_N_10_graph.copy()
for key1 in ["sigma_v", "sigma_w", "C_u_sqrt", "C_u", "C_w_sqrt", "C_w"]:
    del cfg_non_linear_vs_snr[key1]
cfg_non_linear_vs_snr.update({"sigma_w_list": np.logspace(-2, -0.5, 5),
                              "num_iterations": 100,})
#########################################################################
############## - Performance vs. rate of graph variations  ##############
#########################################################################
cfg_non_linear_vs_delta_n = cfg_N_10_graph.copy()
for key1 in ["delta_n",]:
    del cfg_non_linear_vs_delta_n[key1]
cfg_non_linear_vs_delta_n.update({
    "k": 3 * cfg_non_linear_vs_delta_n["n"],
    "delta_n_list": np.linspace(1,10,10).astype(int),
    "num_iterations": 100,
})
#########################################################################
############## - Performance vs. rate of graph variations  ##############
#########################################################################
cfg_non_linear_vs_k = cfg_N_10_graph.copy()
for key1 in ["k",]:
    del cfg_non_linear_vs_k[key1]
cfg_non_linear_vs_k.update({"k_list": np.geomspace(1, 32, num=6),
                            "num_iterations": 100,
                            })
########################################################################
############# - Performance vs. sparsity level  ########################
########################################################################
cfg_non_linear_vs_sparsity = cfg_N_10_graph.copy()
for key1 in ["num_edges_stateinit",]:
    del cfg_non_linear_vs_sparsity[key1]
cfg_non_linear_vs_sparsity.update({
    "sparsity_list": np.linspace(10, 50, 5),
    "C_u_sqrt": np.dot(0.05 ** 0.5, np.eye(cfg_non_linear_vs_sparsity["m"])),
    "k": cfg_non_linear_vs_sparsity["n"],
    "num_iterations": 100,
})

########################################################################
##### - Performance vs. sparsity and degree std - Non-Linear case 1 ####
########################################################################
cfg_non_linear_case1_vs_degree_std = cfg_non_linear_case1.copy()
for key1 in ["num_edges_stateinit",]:
    del cfg_non_linear_case1_vs_degree_std[key1]
cfg_non_linear_case1_vs_degree_std.update({
    "num_edges_values": [57, 114],
    "degree_std_values": [(0, 0.5), (1.5, 2), (3, 3.5), (4, 4.5)],
    "num_iterations": 100,
    "trajectory_time": np.arange(0, 159),#159
})

#########################################################################
########################### - Run time vs. poly order  ##################
#########################################################################
cfg_non_linear_vs_filter_order = cfg_N_10_graph.copy()
for key1 in ["poly_coefficients",]:
    del cfg_non_linear_vs_filter_order[key1]
cfg_non_linear_vs_filter_order.update({
    "p_list": np.round(np.linspace(1, cfg_non_linear_vs_filter_order["n"]-1, 5)).astype(int),
    "num_iterations": 100,
})
cfg_non_linear_vs_filter_order.update({
"cov_init": 1.0,
"stateInit_weight_grls": 1.0,
"beta": 0.9,
"alpha": 0.01,
"lambda_r": 0.1,
"t_max": 20,
"p_e": 0.05,
"sigma_prob_ssm": 0.25,
"w_ref": 1.0,
"state_est":'avg',
})