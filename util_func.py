import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pickle
import logging
import os, psutil, time
from numpy.linalg import svd

matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc. depending on your setup


#-------------------------   Basic util functions   -------------------------
def generate_q(n):
    return np.random.normal(size=(n, 1))


def generate_H(B, q):
    return np.dot(B, vector2diag(np.dot(B.T, q)))


def generate_y(H, x, C_w_sqrt):
    observation = np.dot(H, x) + np.dot(C_w_sqrt, np.random.normal(size=(H.shape[0], 1)))
    return observation


def state_evolvment(F, state, C_u):
    new_state = np.dot(F, state) + np.dot(C_u, np.random.normal(size=(state.size, 1)))
    return new_state


def truncate_matrix(x_est, thr):
    zero_idx = np.where(x_est < thr)
    x_est[zero_idx] = 0
    return x_est


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


def chose_indices_without_repeating(max_num_edges, num_of_edges):
    assert max_num_edges >= num_of_edges, f"max_num_edges >= num_of_edges, but max_num_edges={max_num_edges} and num_of_edges={num_of_edges}"
    import random
    return random.sample(range(max_num_edges), num_of_edges)


# -------------------------  Kalman filter util function  -------------------------
def kalman_gain_imp(H, Sigma, W):
    S = (np.dot(H, np.dot(Sigma, H.T)) + np.dot(H, np.dot(Sigma, H.T)).T) / 2 + W
    # Calculate the Kalman Gain
    K = np.dot(np.dot(Sigma, H.T), pseudo_inv(S, 1e-3))
    I = np.eye(H.shape[1])
    Sigma = np.dot(np.dot((I - np.dot(K, H)), Sigma), (I - np.dot(K, H)).T) + np.dot(K, np.dot(W, K.T))
    return S, K, Sigma



# -------------------------  Error metrics functions  -------------------------
def calc_mse(est, true_state):
    mse = np.dot((est - true_state).T, (est - true_state)) / len(true_state)
    normalized_mse = mse[0][0] / np.dot(true_state.T, true_state)
    return mse, normalized_mse


def calc_error_in_existing_edges(est, true_state):
    est_edge_det = set(np.where(est == 0)[0])
    true_edge_det = set(np.where(true_state == 0)[0])
    return len(est_edge_det.symmetric_difference(true_edge_det))


def calc_f1_score(est, true_state):
    est_edge_det = set(np.where(est >= 0.1)[0])
    true_edge_det = set(np.where(true_state >= 0.1)[0])
    tp = len(est_edge_det.intersection(true_edge_det))
    fp_fn = len(est_edge_det.symmetric_difference(true_edge_det))
    return 2 * tp / max(1, (2 * tp + fp_fn))


def calc_edge_identification_error_rate_score(est, true_state):
    est_edge_det = set(np.where(est >= 0.1)[0])
    true_edge_det = set(np.where(true_state >= 0.1)[0])
    edge_identification_error = len(est_edge_det.symmetric_difference(true_edge_det))
    eier = 100 * edge_identification_error / (2 * len(est))
    return eier, eier/ max((len(est) - len(true_edge_det)), 1)


# -------------------------  Data generation function  -------------------------
def sample_matrix(matrix, idx_rows, idx_columns=None):
    no_connection = np.setdiff1d(np.arange(matrix.shape[0]), idx_rows)
    if no_connection.size > 0:
        if matrix.shape[1] > 1:
            matrix[no_connection, :] = 0
            if not (idx_columns is None):
                no_connection = np.setdiff1d(np.arange(matrix.shape[1]), idx_columns)
            matrix[:, no_connection] = 0
        else:
            matrix[no_connection] = 0
    return matrix


def change_edge_set(state, p=1):
    indices = np.random.choice(len(state), size=p, replace=False)
    # Change the value at the chosen index (flip between 0 and 1)
    state[indices] = 1 - state[indices]
    # Append the new vector to the list
    updated_connections = np.where(state > 0)[0]
    updated_connections = np.sort(updated_connections)
    return updated_connections, state


def compute_measurements(L, C_w_sqrt, N, a):
    q = generate_q(N)
    f = compute_poly(L, q, a)
    observation = generate_y(np.eye(N), f, C_w_sqrt)
    return q, observation


def single_update_iteration(state, F, B, C_w, C_u, N, k):
    state = state_evolvment(F, state, C_u)
    q = generate_q(N)
    H = generate_H(B, q)
    observation = generate_y(H, state, C_w)
    return q, state, observation


def get_trajectory(trajectory_time, F, B, C_w_sqrt, C_u_sqrt, N, k, poly_c, new_edge_weight, num_edges_stateinit, delta_n):
    position = []
    measurements_q = []
    measurements_y = []
    m = num_possible_edges(N)
    idx_list = chose_indices_without_repeating(m, num_edges_stateinit)
    stateInit = np.zeros(m).reshape([m, 1])
    stateInit[idx_list] = new_edge_weight
    state = stateInit.copy()
    updated_connections_list = []
    data_len = trajectory_time.size
    for i in range(data_len):
        if (i % k == 0):
            binary_state = (state > 0).astype(float)
            updated_connections, new_binary_state = change_edge_set(np.copy(binary_state),delta_n)
            F_s = sample_matrix(np.copy(F), updated_connections)
            C_u_s_sqrt = sample_matrix(np.copy(C_u_sqrt), updated_connections)
            new_edges_indicator = ((new_binary_state - binary_state) > 0).astype(float)
            if np.sum(new_edges_indicator) > 0:
                print("new edge was added")
            state = generate_y(F_s, np.copy(state) + new_edge_weight * new_edges_indicator, C_u_s_sqrt)
        else:
            updated_connections = np.where(state > 0)[0]
            updated_connections = np.sort(updated_connections)
            C_u_s_sqrt = sample_matrix(np.copy(C_u_sqrt), updated_connections)
            F_s = sample_matrix(np.copy(F), updated_connections)
            state = generate_y(F_s, np.copy(state), C_u_s_sqrt)
        state = np.abs(state)
        q, observation = compute_measurements(build_L(B, np.copy(state)), C_w_sqrt, N, poly_c)
        updated_connections_list.append(updated_connections)
        state = np.abs(np.copy(state))
        measurements_q.append(q)
        position.append(np.copy(state))
        measurements_y.append(observation)
    return position, measurements_q, measurements_y, updated_connections_list, stateInit


# -------------------------  Graph Functions  -------------------------
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


def vector2diag(x):
    m = max(x.shape)
    gamma = np.zeros([m, m])
    np.fill_diagonal(gamma, x.reshape([1, m]))
    return gamma

def build_L(B, x):
    gamma = vector2diag(x)
    return np.dot(B, np.dot(gamma, np.transpose(B)))

def plot_state_as_graph(x, B, title_str):
    # print(z)
    L = build_L(B, x)
    # print(f"L={L}")
    Adj = np.diag(np.diag(L)) - L
    # print(Adj)
    print(Adj)
    G_t = nx.from_numpy_array(Adj)
    # G_t = nx.complete_graph(n, create_using=None)
    fig, ax = plt.subplots()
    # nx.draw(G_t, ax=ax, with_labels=True)
    # pos = nx.draw_circular(G_t)
    pos = nx.circular_layout(G_t)
    # print(pos)
    # nx.draw_networkx(G_t,pos)
    labels = nx.get_edge_attributes(G_t, 'weight')
    # print(labels)
    nx.draw(G_t, pos)
    nx.draw_networkx_edge_labels(G_t, pos, edge_labels=labels)
    # nx.draw(G, pos=pos)
    # # nx.draw_circular(G_t, with_labels=True, edge_labels=labels, width=4)
    ax.set_title(title_str)


# -------------------------  Polynomial functions  -------------------------
def compute_Jacobian_poly_dp(L, B, q, a, edges):
    """Efficient Jacobian for f(L)=Σ a_p L^p  (sparse version)."""
    # q = q.reshape(-1, 1)

    N, E_max = B.shape
    P = len(a)
    I = np.eye(N)

    # c_p = L^p q
    c = [q]
    for _ in range(1, P):
        c.append(L @ c[-1])

    # D_p recursion
    D = [np.zeros((N,N))] * P
    D[P - 1] = np.zeros((N, N))  # D_{P-1}=0

    for p in range(P - 2, -1, -1):  # p = P-2 … 0
        D[p] = L @ D[p + 1] + a[p + 1] * I

    H = np.zeros((N, B.shape[1]), dtype=L.dtype)
    for j in range(B.shape[1]):
        (n, m) = edges[j]
        col = np.zeros((N, 1), dtype=L.dtype)
        for p in range(P):
            temp_col = (c[p][n]-c[p][m]) * (D[p][:,n]-D[p][:,m])
            col += temp_col.reshape(-1,1)
        H[:, j] = col.ravel()
    return H


def compute_Jacobian_poly(L, B, q, a):
    """
    Compute the j-th column of the Jacobian H_t given by:
        [H_t]_j = sum_i a_i ( sum_{k=0}^{i-1} L^k (B E_j B^T) L^{i-k-1} q )

    Parameters
    ----------
    L : np.ndarray
        Laplacian matrix, shape (N, N).
    B : np.ndarray
        Incidence matrix, shape (N, E_max).
    q : np.ndarray
        Input vector q_t, shape (N,).
    a : np.ndarray
        Polynomial coefficients of f(L): f(L)=sum_i a_i L^i, shape (m+1,).

    Returns
    -------
    H_col : np.ndarray
        The j-th column of the Jacobian H_t, shape (N,).
    """
    # Dimensions
    N = L.shape[-1]
    E_max = B.shape[1]
    # Maximum power needed
    max_power = len(a) - 1  # If poly_c = [a_0, a_1, ..., a_m], need L^m

    # Precompute powers of L
    L_powers = [np.eye(N)]  # L^0 = I
    for _ in range(max_power):
        L_powers.append(np.matmul(L_powers[-1], L))
    H = np.zeros([N, E_max])
    for j in range(E_max):
        # E_j with poly_c single 1 at position (1,1), for example:
        E_j = np.zeros((E_max, E_max))
        E_j[j, j] = 1.0
        # Compute M_j = B E_j B^T
        M_j = np.matmul(B, np.matmul(E_j, B.T))

        # Initialize result vector
        H_col = np.zeros([N, 1])

        # Compute the sum for each i
        # Note: i starts from 1 since for i=0 there's no inner sum.
        # If polynomial is a_0 + a_1 L + ... + a_m L^m,
        # then i runs from 1 to m for the derivative terms.
        for i_idx in range(1, len(a)):  # i_idx corresponds to i in the formula
            ai = a[i_idx]
            if ai == 0:
                continue
            inner_sum = np.zeros([N, 1])

            # sum_{k=0}^{i-1} L^k (M_j) L^{i-k-1} q
            # i_idx - 1 is the highest exponent in the second part
            for k in range(i_idx):
                # L^k
                Lk = L_powers[k]
                # L^{i-k-1}
                L_i_k_1 = L_powers[i_idx - k - 1]

                # Compute L^k (M_j) L^{i-k-1} q
                # First compute temp = L^{i-k-1} q
                temp = np.matmul(L_i_k_1, q)
                # Then (M_j) temp
                temp = np.matmul(M_j, temp)
                # Then L^k temp
                temp = np.matmul(Lk, temp)

                inner_sum = inner_sum + temp

            # Multiply by a_i
            H_col += ai * inner_sum
        H[:, j:j + 1] = H_col

    return H


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



# -------------------------  saving and editing functions  -------------------------
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

# -------------------------  Simulation running functions  -------------------------
def one_method_evaluation(kf, measurements_q, measurements_y, position, updated_connections_list):
    mse_list = []
    normalized_mse_list = []
    F1_score_list = []
    EIER_list = []
    normalized_EIER_list = []
    times_list = []
    for i, (q, y, true_state, updated_connections) in enumerate(
            zip(measurements_q, measurements_y, position, updated_connections_list)):
        start = time.time()
        x_est = kf(q, y, updated_connections)
        end = time.time()
        elapsed = end - start
        mse, normalized_mse = calc_mse(x_est, true_state)
        logging.info(f"Iteration {i}: Execution time = {elapsed:.6f} seconds, mse={mse[0][0]}")
        mse_list.append(mse)
        normalized_mse_list.append(normalized_mse)
        F1_score_list.append(calc_f1_score(x_est, true_state))
        eier, normaized_eier = calc_edge_identification_error_rate_score(x_est, true_state)
        EIER_list.append(eier)
        normalized_EIER_list.append(normaized_eier)
        times_list.append(elapsed)
    return np.concatenate(mse_list), np.concatenate(normalized_mse_list), np.array(F1_score_list).reshape([i + 1, 1]), np.array(EIER_list).reshape(
        [i + 1, 1]), np.array(normalized_EIER_list).reshape([i + 1, 1]), times_list


# -------------------------  Running operation system functions  -------------------------
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


# -------------------------  Results processing functions  -------------------------
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


def create_table(runs, metric, methods_to_plot=None):
    methods_dict = compute_metric_summary(runs, metric, methods_to_plot=methods_to_plot)
    for method in methods_dict.keys():
        methods_dict[method] = methods_dict[method].mean()
    return methods_dict


# -------------------------  Plotting Functions  -------------------------
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


def plot_metric(time, runs, metric, labels=None, methods_to_plot=None, log_format=False, to_save=False, folder_name="",
                suffix=""):
    plt.rcParams.update({'font.size': 12})
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
    prefix = "N" if (metric == "mse") else ""
    YLABEL = prefix + metric.upper() + postfix
    plt.ylabel(YLABEL)
    plt.xlim(time[0], time[-1])
    plt.grid()
    plt.legend()
    if to_save:
        full_path = os.path.join(folder_name, f"{metric}_vs_time_{suffix}.png")
        plt.savefig(full_path, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def unit_func(x):
    return x

def mean_func(table):
    return table.mean(axis=1)

def plot_vs_parameter(parameter_values_list, runs, metric, aggregation_func=unit_func, labels=None,
                      methods_to_plot=None, log_format=False, log_x_axis=False, x_label1="", to_save=False,
                      folder_name="", suffix=""):
    plt.rcParams.update({'font.size': 16})
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
    if log_x_axis:
        plt.xscale('log', base=2)
    postfix = " [dB]" if log_format else ""
    prefix = "N" if (metric == "mse") else ""
    YLABEL = prefix + metric.upper() + postfix
    plt.ylabel(YLABEL)
    plt.xlim(parameter_values_list[0], parameter_values_list[-1])
    plt.grid()
    plt.legend()
    if to_save:
        full_path = os.path.join(folder_name, f"{metric}_vs_{suffix}.png")
        plt.savefig(full_path, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


# -------------------------  Post running functions  -------------------------
def update_performance_vs_time_data(orig_file_name, new_data_file_name, file_name_to_save):
    with open(orig_file_name, "rb") as f:
        orig_data = pickle.load(f)
    # orig_data = orig_data[1:]
    with open(new_data_file_name, "rb") as f:
        new_data = pickle.load(f)
    new_dict_list1 = []
    for iter_idx in range(len(new_data)):
        new_dict = dict()
        temp = orig_data[iter_idx]
        del temp["ekf"]
        new_dict.update(temp)
        new_dict.update(new_data[iter_idx])
        new_dict_list1.append(new_dict)
    with open(file_name_to_save, "wb") as f:
        pickle.dump(new_dict_list1, f)
    return new_dict_list1


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
