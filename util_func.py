import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle


matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc. depending on your setup


import os, psutil, time, math
from numpy.linalg import svd


def kalman_gain_imp(H, Sigma, W):
    S = (np.dot(H, np.dot(Sigma, H.T)) + np.dot(H, np.dot(Sigma, H.T)).T) / 2 + W
    # Calculate the Kalman Gain
    K = np.dot(np.dot(Sigma, H.T), pseudo_inv(S, 1e-3))
    I = np.eye(H.shape[1])
    Sigma = np.dot(np.dot((I - np.dot(K, H)), Sigma), (I - np.dot(K, H)).T) + np.dot(K, np.dot(W, K.T))
    return S, K, Sigma

def pseudo_inv(A, thr=None):
    """
    Return the Moore–Penrose pseudo-inverse of a matrix A.

    Parameters
    ----------
    A : (m, n) array_like
        Input matrix (real or complex).
    thr : float, optional
        Singular values <= thr are treated as zero.
        If None, an adaptive threshold max(m, n) * eps * max(s) is used,
        where eps is machine precision for A’s dtype and s are the singular values.

    Returns
    -------
    A_pinv : (n, m) ndarray
        The pseudo-inverse of A.
    """
    # Economy-size SVD (cheaper, gives U:(m×r), Vt:(r×n) where r = rank)
    U, s, Vt = svd(A, full_matrices=False)

    # Automatic tolerance if not provided
    if thr is None:
        thr = max(A.shape) * np.finfo(s.dtype).eps * s.max()

    # Invert the singular values above the threshold
    s_inv = np.where(s > (thr * s.max()), 1.0 / s, 0.0)

    # Re-compose: V * Σ⁻¹ * Uᵀ   (note: Vt is Vᵀ)
    A_pinv = (Vt.T * s_inv) @ U.T
    return A_pinv


def pick_worker_count(reserve_cores: int = 1,
                      idle_thresh: int = 20,
                      sample_secs: float = 0.2) -> int:
    """
    Return the number of workers that should be safe to spin up *right now*.

    reserve_cores  – always leave this many logical cores completely free
    idle_thresh    – treat a core as 'idle' if current util < this %
    sample_secs    – how long to measure utilisation
    """
    logical = os.cpu_count() or 1

    # single short sample – cheap enough for interactive jobs
    usage = psutil.cpu_percent(interval=sample_secs, percpu=True)
    idle_cores = sum(u < idle_thresh for u in usage)

    # leave a buffer so you don’t grab every last idle core
    safe = max(1, idle_cores - reserve_cores)
    return min(safe, logical - reserve_cores)

max_workers = pick_worker_count()



def aggregate_across_runs(runs, metric, method):
    """Stack the chosen metric across Monte-Carlo repetitions."""
    return np.stack([r[method][metric] for r in runs])      # shape: (R, T)


def stack_across_runs(runs, method):
    """Stack the chosen metric across Monte-Carlo repetitions."""
    return np.stack([r[method] for r in runs])

def compute_metric_summary(runs, metric, methods_to_plot=None):
    methods_list = methods_to_plot if methods_to_plot is not None else runs[0].keys()
    summary_table = {}
    for method in methods_list:
        if method in runs[0].keys():
            y = aggregate_across_runs(runs, metric, method).mean(axis=0)
            summary_table[method] = y
    return summary_table

LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
LINE_WIDTH = 2.5
MARKERS = ['o', 's', 'v', '^', 'D', 'x', '*', 'P', 'h', '+']  # extend as needed

def get_line_style(method, method_list=None):
    if method_list is None:
        return LINE_STYLES[hash(method) % len(LINE_STYLES)]
    index = method_list.index(method) % len(LINE_STYLES)
    return LINE_STYLES[index]


def get_marker(method, method_list=None):
    if method_list is None:
        return MARKERS[hash(method) % len(MARKERS)]
    index = method_list.index(method) % len(MARKERS)
    return MARKERS[index]


def plot_metric(time, runs, metric, labels=None, methods_to_plot=None, log_format=False, to_save=False, suffix=""):
    methods_dict = compute_metric_summary(runs, metric, methods_to_plot=methods_to_plot)
    plt.figure()
    methods_list = list(methods_dict.keys())
    for method in methods_list:
            y = methods_dict[method]
            linestyle = get_line_style(method, methods_list)
            # marker = get_marker(method, methods_list)
            label = method if labels is None else labels[method]
            if log_format:
                plt.plot(time, 10 * np.log10(y), label=label, linestyle=linestyle, linewidth=LINE_WIDTH)
            else:
                plt.plot(time, y, label=label, linestyle=linestyle, linewidth=LINE_WIDTH)
    plt.xlabel("l [time units]")
    postfix = " [dB]" if log_format else ""
    YLABEL = metric.upper() + postfix
    plt.ylabel(YLABEL)
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.legend()
    if to_save:
        plt.savefig(f"{metric}_vs_time_{suffix}.png", format='png')#, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def unit_func(x):
    return x

def mean_func(table):
    return table.mean(axis=1)

def plot_vs_parameter(parameter_values_list, runs, metric, aggregation_func=unit_func, labels=None,
                      methods_to_plot=None, log_format=False, x_label1="", to_save=False, suffix=""):
    mean_metric_along_trajectory = []
    for idx in range(len(parameter_values_list)):
        methods_dict = compute_metric_summary(runs[idx], metric)
        mean_metric_along_trajectory.append(methods_dict)
    methods_list = methods_to_plot if methods_to_plot is not None else list(mean_metric_along_trajectory[0].keys())
    plt.figure()
    for method in methods_list:
        if method in mean_metric_along_trajectory[0].keys():
            metric_along_trajectory_vs_parameter = np.stack([r[method] for r in mean_metric_along_trajectory]).squeeze()
            y = aggregation_func(metric_along_trajectory_vs_parameter)
            linestyle = get_line_style(method, methods_list)
            marker = get_marker(method, methods_list)
            label = method if labels is None else labels[method]
            if log_format:
                plt.plot(parameter_values_list, 10 * np.log10(y), label=label, linestyle=linestyle, marker=marker, linewidth=LINE_WIDTH)
            else:
                plt.plot(parameter_values_list, y, label=label, linestyle=linestyle, marker=marker, linewidth=LINE_WIDTH)
    plt.xlabel(x_label1)
    postfix = " [dB]" if log_format else ""
    YLABEL = metric.upper() + postfix
    plt.ylabel(YLABEL)
    plt.xlim(parameter_values_list[0], parameter_values_list[-1])
    plt.grid()
    plt.legend()
    if to_save:
        plt.savefig(f"{metric}_vs_{suffix}.png", format='png')#, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def create_table(runs, metric, methods_to_plot=None):
    methods_dict = compute_metric_summary(runs, metric, methods_to_plot=methods_to_plot)
    for method in methods_dict.keys():
        methods_dict[method] = methods_dict[method].mean()
    return methods_dict


def map_B_to_set(B):
    index_pairs = []

    # Loop through each column
    for col in range(B.shape[1]):
        # Find indices where the column is non-zero
        nonzero_indices = np.nonzero(B[:, col])[0]

        # Ensure there are exactly two non-zero indices
        if len(nonzero_indices) != 2:
            raise ValueError(f"Column {col} does not have exactly two nonzero entries.")

        # Append as poly_c tuple
        index_pairs.append(tuple(nonzero_indices))

    return index_pairs


def num_possible_edges(N):
    assert N >= 1, f"N must be >= 1, but N={N}"
    return int(N * (N - 1) / 2)


def chose_indices_without_repeating(max_num_edges, num_of_edges):
    assert max_num_edges >= num_of_edges, f"max_num_edges >= num_of_edges, but max_num_edges={max_num_edges} and num_of_edges={num_of_edges}"
    import random
    return random.sample(range(max_num_edges), num_of_edges)


def vector2diag(x):
    m = max(x.shape)
    gamma = np.zeros([m, m])
    np.fill_diagonal(gamma, x.reshape([1, m]))
    return gamma

def build_L(B, x):
    gamma = vector2diag(x)
    return np.dot(B, np.dot(gamma, np.transpose(B)))


def compute_poly(L, q, a):
    """
    Parameters
    ----------
    L : np.ndarray
        Laplacian matrix, shape (N, N).
    q : np.ndarray
        Input vector q_t, shape (N,).
    a : np.ndarray
        Polynomial coefficients of f(L): f(L)=sum_i a_i L^i, shape (m+1,).

    Returns
    -------
    H_col : np.ndarray
        The polynom after multiply by q, shape (N,).
    """
    # Dimensions
    N = L.shape[-1]
    # E_max = B.shape[1]
    # Maximum power needed
    if len(q.shape) == 3:
        batch_size = q.shape[0]
        signal_out = np.zeros([batch_size, N, 1])

    else:
        signal_out = np.zeros([N, 1])
    # if isinstance(q, torch.Tensor):
    #     signal_out = torch.from_numpy(signal_out).float()

    max_power = len(a)  # If poly_c = [a_0, a_1, ..., a_m], need L^m

    # Precompute powers of L
    # L_powers = [np.eye(N)]  # L^0 = I
    L_q = q
    for i_idx in range(max_power):
        signal_out = signal_out + a[i_idx] * L_q
        L_q = np.matmul(L, L_q)
    return signal_out



##  saving and editing functions
def update_performance_vs_time_data(orig_file_name, new_data_file_name, file_name_to_save):
    # Load
    with open(orig_file_name, "rb") as f:
        orig_data = pickle.load(f)

    # Load
    with open(new_data_file_name, "rb") as f:
        new_data = pickle.load(f)


        merged_dict_list = []
        for i in range(len(new_data)):
            new_dict = dict()
            new_dict.update(orig_data[i])
            new_dict.update(new_data[i])
            merged_dict_list.append(new_dict)
        with open(file_name_to_save, "wb") as f:
            pickle.dump(merged_dict_list, f)
        return merged_dict_list


def update_performance_vs_parameter_data(orig_file_name, new_data_file_name, file_name_to_save):
    with open(orig_file_name, "rb") as f:
        orig_data = pickle.load(f)
    # orig_data = orig_data[1:]
    with open(new_data_file_name, "rb") as f:
        new_data = pickle.load(f)
    merged_dict_list = []
    for parameter_val in range(len(new_data)):
        new_dict_list1 = []
        for iter_idx in range(len(new_data[parameter_val])):
            new_dict = dict()
            new_dict.update(orig_data[parameter_val][iter_idx])
            new_dict.update(new_data[parameter_val][iter_idx])
            new_dict_list1.append(new_dict)
        merged_dict_list.append(new_dict_list1)
    with open(file_name_to_save, "wb") as f:
        pickle.dump(merged_dict_list, f)
    return merged_dict_list