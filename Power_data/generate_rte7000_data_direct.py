import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import networkx as nx

from util_func import extract_x_to_match_B_from_L, build_L

# ── paths (bundled RTE7000 'bal' subgraph data) ─────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'rte7000_data')
BRANCH_PARAMS_BAL_PATH = os.path.join(DATA_DIR, 'branch_params_bal.json')
NODE_META_BAL_PATH = os.path.join(DATA_DIR, 'node_meta_bal.json')
SUBGRAPH_META_BAL_PATH = os.path.join(DATA_DIR, 'subgraph_meta_bal.json')
BOUNDARY_PARAMS_BAL_PATH = os.path.join(DATA_DIR, 'boundary_branches_bal.json')
GENERATORS_BAL_PATH = os.path.join(DATA_DIR, 'generators_bal.json')
_TOPOLOGY_CSV_BY_SIZE = {
    'bal': os.path.join(DATA_DIR, 'bal100_timeseries.csv'),
}

S_BASE_MVA = 100.0

_PATHS_BY_SIZE = {
    'bal': (BRANCH_PARAMS_BAL_PATH, NODE_META_BAL_PATH, SUBGRAPH_META_BAL_PATH,
            BOUNDARY_PARAMS_BAL_PATH, GENERATORS_BAL_PATH),
}


def load_static_data(size='bal'):
    branch_path, node_path, meta_path, _, _ = _PATHS_BY_SIZE[size]
    with open(branch_path) as f:
        branches = json.load(f)
    with open(node_path) as f:
        node_meta = json.load(f)
    with open(meta_path) as f:
        sub_meta = json.load(f)
    return branches, node_meta, sub_meta


def load_boundary_data(size='bal'):
    _, _, _, boundary_path, _ = _PATHS_BY_SIZE[size]
    with open(boundary_path) as f:
        return json.load(f)


def pick_slack_bus(branches, node_meta):
    """Slack = node with highest degree in the (in-service) subgraph."""
    deg = {bid: 0 for bid in node_meta}
    for br in branches:
        if br['connected1'] and br['connected2']:
            deg[br['bus1']] = deg.get(br['bus1'], 0) + 1
            deg[br['bus2']] = deg.get(br['bus2'], 0) + 1
    return max(deg, key=deg.get)


def load_real_topology_series(csv_path, branch_ids):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    missing = [b for b in branch_ids if b not in df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} branch ids missing from {csv_path}, e.g. {missing[:3]}")
    return df


# ── real topology + static weights, in memory (no AC/DC solve) ─────────────

def build_real_topology_arrays(size, T, topology_csv, topology_offset):
    if topology_csv is None:
        topology_csv = _TOPOLOGY_CSV_BY_SIZE.get(size)

    branches, node_meta, sub_meta = load_static_data(size=size)
    boundary = load_boundary_data(size=size)
    branch_ids = [b['id'] for b in branches]
    bus_ids_sorted = sorted(node_meta.keys())
    slack_bus = pick_slack_bus(branches, node_meta)

    N_buses = len(bus_ids_sorted)
    N_lines = len(branch_ids)

    real_topo = load_real_topology_series(topology_csv, branch_ids)
    available = max(0, len(real_topo) - topology_offset) if real_topo is not None else 0
    n_real = min(available, T)
    if n_real < T:
        print(f"  NOTE: only {n_real}/{T} steps have real recorded topology from offset "
              f"{topology_offset} -- remaining steps hold at the last known topology.")

    bus_name_to_row = {name: i for i, name in enumerate(bus_ids_sorted)}
    line_ids = np.array([[bus_name_to_row.get(b['bus1'], -1), bus_name_to_row.get(b['bus2'], -1)] for b in branches])
    line_params = np.array([[b['r_ohm'], b['x_ohm'], b['b_siemens'], 1.0] for b in branches])

    B_real = np.zeros((N_buses, N_lines))
    weights_real = np.zeros(N_lines)
    for i, br in enumerate(branches):
        u, v = line_ids[i]
        if u < 0 or v < 0:
            continue
        vn1 = node_meta[br['bus1']]['nominal_v']
        vn2 = node_meta[br['bus2']]['nominal_v']
        if abs(vn1 - vn2) < 1e-6:
            x_pu = max(abs(br['x_ohm']), 1e-9) * S_BASE_MVA / (vn1 ** 2)
        else:
            vbase = (vn1 + vn2) / 2.0
            x_pu = max(abs(br['x_ohm']) * S_BASE_MVA / (vbase ** 2), 1e-6)
        weights_real[i] = 1.0 / x_pu
        B_real[u, i] = 1.0
        B_real[v, i] = -1.0

    L_static = build_L(B_real, weights_real)
    B_complete = nx.incidence_matrix(nx.complete_graph(N_buses), oriented=True).todense()
    full_weights = extract_x_to_match_B_from_L(L_static, B_complete)

    current_topology = np.array([bool(br['connected1'] and br['connected2']) for br in branches], dtype=bool)
    slack_row = bus_name_to_row[slack_bus]

    line_topology = np.zeros((T, N_lines), dtype=np.int8)
    isolated_mask = np.zeros((T, N_buses), dtype=bool)
    for t in range(T):
        if t < n_real:
            row = real_topo.iloc[topology_offset + t]
            for i, bid in enumerate(branch_ids):
                val = row.get(bid, np.nan)
                current_topology[i] = bool(val) if not pd.isna(val) else current_topology[i]

        g = nx.Graph()
        g.add_nodes_from(range(N_buses))
        for i in range(N_lines):
            if current_topology[i]:
                u, v = line_ids[i]
                if u >= 0 and v >= 0:
                    g.add_edge(int(u), int(v))
        reachable = nx.node_connected_component(g, slack_row)
        isolated_mask[t, [b for b in range(N_buses) if b not in reachable]] = True

        line_topology[t] = current_topology.astype(np.int8)

    return {
        'bus_ids': np.array(bus_ids_sorted), 'line_ids': line_ids, 'line_params': line_params,
        'full_weights': full_weights, 'line_weights': weights_real, 'line_topology': line_topology,
        'slack_bus': slack_bus, 'N_buses': N_buses, 'N_lines': N_lines, 'n_real': n_real,
        'boundary_count': len(boundary), 'isolated_mask': isolated_mask,
    }


def build_cfg(real):
    n = real['N_buses']
    slack_idx = int(np.where(real['bus_ids'] == real['slack_bus'])[0][0])
    B = nx.incidence_matrix(nx.complete_graph(n), oriented=True).todense()
    edge_to_col = {e: i for i, e in enumerate(nx.complete_graph(n).edges())}
    m = len(edge_to_col)
    return {'n': n, 'm': m, 'B': B, 'full_weights': real['full_weights'],
            '_edge_to_col': edge_to_col, 'line_ids': real['line_ids'], 'slack_idx': slack_idx}


def _dc_estimation_residual(L, q, y, slack_idx):
    """DC (linearized) load-flow check for one time step"""
    zero_rows = np.where(np.all(L == 0, axis=1))[0]
    mask_rows = np.delete(np.arange(L.shape[1]), zero_rows)
    c = np.concat([zero_rows, np.array([slack_idx])], axis=0)
    mask_cols = np.delete(np.arange(L.shape[1]), c)
    L_sub = L[np.ix_(mask_rows, mask_cols)]
    DCestimation, *_ = np.linalg.lstsq(L_sub, y[mask_rows], rcond=None)
    consistency = np.sum(q[mask_cols] - q[slack_idx] - DCestimation)
    residual = y - L @ q
    return consistency, residual


def build_AC_model_dataset(real, cfg, seed=None):
    T = real['line_topology'].shape[0]
    N_buses = real['N_buses']

    rng = np.random.default_rng(seed)
    va_mc_ac_model = rng.normal(0.0, 0.1, size=(T, N_buses))
    va_mc_ac_model[:, cfg['slack_idx']] = 0

    lt = real['line_topology']
    line_weights_mc = real['line_weights'].copy()
    line_weights_mc[line_weights_mc > 120] = 120
    live_weight_sum = np.zeros((lt.shape[0], cfg['m']))
    full_weights_local = np.zeros((cfg['m'], 1))
    for k in range(real['line_ids'].shape[0]):
        u, v = int(real['line_ids'][k, 0]), int(real['line_ids'][k, 1])
        col = cfg['_edge_to_col'][(min(u, v), max(u, v))]
        live_weight_sum[:, col] += lt[:, k] * line_weights_mc[k]
        full_weights_local[col] += line_weights_mc[k]
    full_w_flat = full_weights_local.flatten()
    nonzero_cols = full_w_flat > 0
    conn_mc = np.zeros_like(live_weight_sum)
    conn_mc[:, nonzero_cols] = live_weight_sum[:, nonzero_cols] / full_w_flat[nonzero_cols]
    pos_mc = (conn_mc.T * cfg['full_weights']).T
    B = cfg['B']

    slack_idx_mc = cfg['slack_idx']
    dc_consistency_mc = np.zeros(T)
    dc_residuals_mc = np.zeros((T, N_buses))
    y_mc_ac_model = np.zeros((T, N_buses))
    for t in range(T):
        L_t = build_L(B, pos_mc[t, :])
        # True AC injection S = V*conj(Y_t @ V), |V|=1 pu
        V = np.exp(1j * va_mc_ac_model[t, :])
        S = V * np.conj(-1j * L_t @ V)
        y_mc_ac_model[t, :] = S.real + np.dot(0.1, rng.normal(size=(N_buses, )))
        dc_consistency_mc[t], dc_residuals_mc[t, :] = _dc_estimation_residual(
            L_t, va_mc_ac_model[t, :], y_mc_ac_model[t, :], slack_idx_mc)

    q_cov_ac_model_mc = np.cov(va_mc_ac_model.T)
    y_cov_ac_model_mc = np.cov(y_mc_ac_model.T)

    return {
        'bus_va_rad': va_mc_ac_model, 'bus_net_injection_mw': y_mc_ac_model,
        'pos_mc': pos_mc, 'B': np.asarray(B), 'line_weights': line_weights_mc,
        'full_weights': full_weights_local,
        'dc_residual_mean_consistency': float(np.mean(np.abs(dc_consistency_mc))),
        'dc_residual_mean_abs': float(np.mean(np.abs(dc_residuals_mc))),
        'q_cov_rank': int(np.linalg.matrix_rank(q_cov_ac_model_mc)),
        'y_cov_rank': int(np.linalg.matrix_rank(y_cov_ac_model_mc)),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='One-shot rte7000_ac_syn_dataset<N> generator')
    parser.add_argument('--subgraph', type=str, default='bal',
                         choices=['bal'])
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--topology_csv', type=str, default=None,
                         help='Defaults to the recorded series matching --subgraph, if one exists')
    parser.add_argument('--n_datasets', type=int, default=None,
                         help='How many T-step datasets to carve out of the real topology series. '
                              'Defaults to using all of it (n_rows // T).')
    parser.add_argument('--output_dir', type=str, default='./rte7000_ac_syn_dataset',
                         help='Prefix for each generated dataset folder -- dataset i is saved to '
                              '"{output_dir}{i}".')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    topology_overlap = args.T // 2
    topology_csv = args.topology_csv or _TOPOLOGY_CSV_BY_SIZE.get(args.subgraph)
    if topology_csv and os.path.exists(topology_csv):
        n_rows = sum(1 for _ in open(topology_csv)) - 1  # minus header
        n_datasets = args.n_datasets or max(1, n_rows // (args.T - topology_overlap))
        print(f"Real topology series has {n_rows} rows -> generating {n_datasets} "
              f"{args.T}-step dataset(s) (seed 0..{n_datasets - 1})")
    else:
        n_datasets = args.n_datasets or 1
        print(f"No real topology series for --subgraph {args.subgraph} -- generating {n_datasets} "
              f"dataset(s) held at the initial (real snapshot) topology throughout.")

    for seed_val in range(n_datasets):
        offset = seed_val * (args.T - topology_overlap)
        print('\n' + '=' * 70)
        logging.info(f"[{seed_val + 1}/{n_datasets}] T={args.T}, offset={offset}")

        real = build_real_topology_arrays(size=args.subgraph, T=args.T,
                                           topology_csv=topology_csv, topology_offset=offset)
        cfg = build_cfg(real)
        syn = build_AC_model_dataset(real, cfg, seed=seed_val)

        logging.info(
            f"  [{seed_val + 1}] DC residual check: mean|consistency|="
            f"{syn['dc_residual_mean_consistency']:.4g}, mean|residual|="
            f"{syn['dc_residual_mean_abs']:.4g}"
        )
        logging.info(
            f"  [{seed_val + 1}] rank(q_cov)={syn['q_cov_rank']}, "
            f"rank(y_cov)={syn['y_cov_rank']}  (out of {real['N_buses']})"
        )

        output_dir = f"{args.output_dir}{seed_val}"
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, "bus_va_rad.npy"), syn['bus_va_rad'])
        np.save(os.path.join(output_dir, "bus_net_injection_mw.npy"), syn['bus_net_injection_mw'])
        np.save(os.path.join(output_dir, "pos_mc.npy"), syn['pos_mc'])
        np.save(os.path.join(output_dir, "B.npy"), syn['B'])
        np.save(os.path.join(output_dir, "line_weights.npy"), syn['line_weights'])
        np.save(os.path.join(output_dir, "full_weights.npy"), syn['full_weights'])
        np.save(os.path.join(output_dir, "line_topology.npy"), real['line_topology'])
        np.save(os.path.join(output_dir, "line_ids.npy"), real['line_ids'])
        np.save(os.path.join(output_dir, "bus_ids.npy"), real['bus_ids'])

        meta = {
            'model': 'ac',
            'source': f'RTE7000 (OpenSynth/D-GITT-RTE7000-2021), size={args.subgraph} 225/380kV subgraph',
            'slack_bus': real['slack_bus'],
            'T': args.T, 'N_buses': real['N_buses'], 'N_lines': real['N_lines'],
            'N_boundary_ties': real['boundary_count'], 'N_generator_buses': 0,
            'n_steps_real_topology': real['n_real'], 'n_steps_ac_model_topology': args.T - real['n_real'],
            'topology_source': 'real (RTE7000 recorded)',
            'topology_offset': offset,
            'has_reactive_power': False, 'has_voltage_magnitude': False,
            'seed': seed_val,
            'ac_model_q': True,
            'ac_model_q_source': ('generated directly in-memory by generate_rte7000_data_direct.py '),
            'ac_model_q_note': ('bus_va_rad/bus_net_injection_mw are synthetic (i.i.d. Gaussian '
                                  'excitation, slack pinned to 0) -- line_topology/line_weights/'
                                  'full_weights/line_ids/bus_ids are the real RTE7000 ground truth.'),
            'units_note': ('bus_net_injection_mw.npy is PER-UNIT here, not MW, despite the filename '
                            '(kept matching the real contract on purpose so existing loaders work '
                            'unmodified) -- y = L_t @ q with no S_BASE_MVA scaling.'),
            'dc_residual_mean_consistency': syn['dc_residual_mean_consistency'],
            'dc_residual_mean_abs': syn['dc_residual_mean_abs'],
            'q_cov_rank': syn['q_cov_rank'], 'y_cov_rank': syn['y_cov_rank'],
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logging.info(f"  [{seed_val + 1}] Saved ac model dataset -> {output_dir}")

    print(f"\nAll {n_datasets} ac model dataset(s) saved under {args.output_dir}*")
