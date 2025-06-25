from EKF_modules import (ExtendedKalmanFilter, FastExtendedKalmanFilter, sparseKalmanFilter, sparseKalmanFilterISTA,
                         oraclKalmanFilt_paper, oraclKalmanFilt_paper_delayed,
                         oraclKalmanFilt_diagonalovariance_update, oraclKalmanFilt_nocovariance_update)
from change_detection_module import ChangeDetectionMethod

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