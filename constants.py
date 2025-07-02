from EKF_modules import (ExtendedKalmanFilter, FastExtendedKalmanFilter, sparseKalmanFilter, sparseKalmanFilterISTA,
                         oraclKalmanFilt_paper, oraclKalmanFilt_paper_delayed,
                         oraclKalmanFilt_diagonalovariance_update, oraclKalmanFilt_nocovariance_update)
from change_detection_module import ChangeDetectionMethod
import networkx as nx
import numpy as np

LABELS = {"change-det": "Change-det", "oracle-block": "Oracle", "ekf": "EKF", "fast-ekf": "EKF", "gsp-ekf": "GSP-EKF"}
METHODS_ORDER = ["change-det", "oracle-block", "ekf", "fast-ekf", "gsp-ekf"]

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
    "gsp-ekf5": lambda cfg: sparseKalmanFilter(
        cfg["F"], cfg["B"], cfg["C_u"], cfg["C_w"],
        cfg["C_x_missmatch"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], thr=0.5
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
        cfg["poly_coefficients"], lambda_1=0.01  # tweak as needed
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
        cfg["C_x"], cfg["stateInit"], cfg["poly_coefficients"], cfg["new_edge_weight"]
    ),
    # ----- Change-detection baseline ---------------------------------------
    "change-det": lambda cfg: ChangeDetectionMethod(
        cfg["B"], cfg["window_len"], cfg["stateInit_missmatch"],
        cfg["poly_coefficients"], lambda_1=cfg["lambda_1"], lambda_2=cfg["lambda_2"]
    ),
}
#########################################################################
######################## - 20 nodes graph  ##############################
#########################################################################
cfg_N_20_graph = {
    "num_iterations": 1,
    "n": 20,
    "new_edge_weight": 1,
    "trajectory_time": np.arange(0, 159),
    "thr1": 0.2,
    "thr2": 0.2,
    "mu": 1,
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
#########################################################################
######################## - 10 nodes graph  ##############################
#########################################################################
cfg_N_10_graph = {
    "num_iterations": 1,
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
#########################################################################
############## - Performance vs. time Linear case  ######################
#########################################################################
cfg_linear = cfg_N_20_graph.copy()
cfg_linear.update({
    "poly_coefficients": np.array([0, 1.0])
})
#########################################################################
############## - Performance vs. time Non-Linear case 1 #################
#########################################################################
cfg_non_linear_case1 = cfg_N_20_graph.copy()
cfg_non_linear_case1.update({
    "poly_coefficients": np.array([1.0, 1.0, 0.1, 1.0])
})
#########################################################################
############## - Performance vs. time - Non-Linear case 2 ###############
#########################################################################
cfg_non_linear_case2 = cfg_N_10_graph.copy()
cfg_non_linear_case2.update({
    "num_iterations": 1,
})
#########################################################################
################# - Performance vs. noise level  ########################
#########################################################################
cfg_non_linear_vs_snr = cfg_N_10_graph.copy()
for key1 in ["sigma_v", "sigma_w", "C_u_sqrt", "C_u", "C_w_sqrt", "C_w"]:
    del cfg_non_linear_vs_snr[key1]
cfg_non_linear_vs_snr.update({"sigma_w_list": np.logspace(-2, -0.5, 5),
})
#########################################################################
############## - Performance vs. rate of graph variations  ##############
#########################################################################
cfg_non_linear_vs_delta_n = cfg_N_10_graph.copy()
for key1 in ["delta_n",]:
    del cfg_non_linear_vs_delta_n[key1]
cfg_non_linear_vs_delta_n.update({
    "k": 3 * cfg_non_linear_vs_delta_n["n"],
    "delta_n_list": np.linspace(1,10,10).astype(int),
})
#########################################################################
############## - Performance vs. rate of graph variations  ##############
#########################################################################
cfg_non_linear_vs_k = cfg_N_10_graph.copy()
for key1 in ["k",]:
    del cfg_non_linear_vs_k[key1]
cfg_non_linear_vs_k.update({"k_list": np.geomspace(1, 32, num=6),
                            })
########################################################################
############# - Performance vs. sparsity level  ########################
########################################################################
cfg_non_linear_vs_sparsity = cfg_N_10_graph.copy()
for key1 in ["num_edges_stateinit"]:
    del cfg_non_linear_vs_sparsity[key1]
cfg_non_linear_vs_sparsity.update({
    "sparsity_list": np.linspace(10, 50, 5),
    "C_u_sqrt": np.dot(0.05 ** 0.5, np.eye(cfg_non_linear_vs_sparsity["m"])),
    "k": cfg_non_linear_vs_sparsity["n"],
})

#########################################################################
########################### - Run time vs. poly order  ##################
#########################################################################
cfg_non_linear_vs_filter_order = cfg_N_10_graph.copy()
for key1 in ["poly_coefficients"]:
    del cfg_non_linear_vs_filter_order[key1]
cfg_non_linear_vs_filter_order.update({
    "p_list": np.round(np.linspace(1, cfg_non_linear_vs_filter_order["n"]-1, 5)).astype(int)
})
