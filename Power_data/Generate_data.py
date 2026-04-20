"""
IEEE 57-Bus Time-Varying Dataset Generator
============================================
Generates a time series dataset from the IEEE 57-bus power system with:
- Time-varying loads (stochastic fluctuation)
- Periodic edge (line) topology changes
- Gaussian measurement noise on state variables

Outputs (each saved as a separate numpy array):
- bus_voltages:    (T, N_buses, 2)   -> [voltage magnitude (pu), voltage angle (deg)]
- bus_loads:       (T, N_buses, 2)   -> [active load P (MW), reactive load Q (MVAr)]
- line_topology:   (T, N_lines)      -> binary, 1=in service, 0=disconnected
- line_flows:      (T, N_lines, 2)   -> [P_from (MW), Q_from (MVAr)]  -- ground truth edge signals
- line_params:     (N_lines, 4)      -> static [r, x, b, length] per line
- bus_ids:         (N_buses,)        -> bus index
- line_ids:        (N_lines, 2)      -> [from_bus, to_bus] per line

Requirements:
    pip install pandapower numpy

Usage:
    python generate_ieee57_dataset.py \
        --T 500 \
        --noise 0.01 \
        --n_varying_edges 10 \
        --change_period 50 \
        --load_variability 0.15 \
        --seed 42 \
        --output_dir ./dataset
"""
import pandas as pd
pd.set_option('display.max_rows', 1000); pd.set_option('display.max_columns', 1000); pd.set_option('display.width', 1000)
import argparse
import os
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate IEEE 57-bus time-varying dataset")
    parser.add_argument("--T",                  type=int,   default=250,
                        help="Number of time samples")
    parser.add_argument("--noise",              type=float, default=0.01,
                        help="Std of Gaussian noise added to state variables (relative, e.g. 0.01 = 1%%)")
    parser.add_argument("--varying_edge_set", type=int, default=20,
                        help="Number of lines whose topology changes over time")
    parser.add_argument("--n_varying_edges",    type=int,   default=4,
                        help="Number of lines whose topology changes over time")
    parser.add_argument("--change_period",      type=int,   default=100,
                        help="Number of time steps between topology change events")
    parser.add_argument("--load_variability",   type=float, default=0.5,
                        help="Std of load fluctuation as fraction of nominal load (e.g. 0.5 = 50%%)")
    parser.add_argument("--seed",               type=int,   default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir",         type=str,   default="./ieee57_dataset",
                        help="Directory to save output .npy files")
    return parser.parse_args()


def add_noise(array: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Add relative Gaussian noise to an array."""
    return array + rng.normal(0, noise_std * np.abs(array) + 1e-9, array.shape)


# ── main generation ───────────────────────────────────────────────────────────

def generate_dataset(T, noise_std, varying_edge_set, n_varying_edges, change_period, load_variability, seed, output_dir):
    try:
        import pandapower as pp
        import pandapower.networks as pn
    except ImportError:
        raise ImportError("Please install pandapower:  pip install pandapower")

    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load IEEE 57-bus network ─────────────────────────────────────────────
    net = pn.case57()
    print(net.bus[['name', 'type']])

    pp.runpp(net, algorithm='nr', numba=False)   # initial solve to validate

    N_buses = len(net.bus)
    N_lines = len(net.line)

    print(f"IEEE 57-bus loaded:  {N_buses} buses,  {N_lines} lines")
    print(f"Generating {T} time steps ...")
    # ── Store static line parameters ──────────────────────────────────────────
    line_params = net.line[['r_ohm_per_km', 'x_ohm_per_km',
                             'c_nf_per_km',  'length_km']].values.astype(float)
    # (N_lines, 4)

    bus_ids  = net.bus.index.to_numpy()
    line_ids = net.line[['from_bus', 'to_bus']].values  # (N_lines, 2)
    Y_mat = net._ppc["internal"]["Ybus"].todense()

    # ── Choose which lines will vary ──────────────────────────────────────────
    # Find parallel lines and insert them to untouchable list
    line_ids_unique = net.line[['from_bus', 'to_bus']].copy()
    line_ids_unique['repetitions'] = line_ids_unique.groupby(['from_bus', 'to_bus'])['from_bus'].transform('count')
    duplicate_indices = line_ids_unique[line_ids_unique['repetitions'] > 1].index
    unique_lines = np.setdiff1d(np.arange(N_lines), duplicate_indices)
    import networkx as nx
    G = pp.topology.create_nxgraph(net)
    bridges = list(nx.bridges(G))  # lines whose removal disconnects the graph
    bridge_line_indices = []
    for (u, v) in bridges:
        mask = ((net.line['from_bus'] == u) & (net.line['to_bus'] == v)) | \
               ((net.line['from_bus'] == v) & (net.line['to_bus'] == u))
        idx = net.line.index[mask].tolist()
        bridge_line_indices.extend(idx)

    print("Bridge line indices:", bridge_line_indices)

    # Then exclude them from varying lines
    safe_lines = np.setdiff1d(unique_lines, bridge_line_indices)
    varying_edge_set = min(varying_edge_set, len(safe_lines))
    varying_line_idx = rng.choice(safe_lines, size=varying_edge_set, replace=False)
    print(f"Varying lines (indices): {sorted(varying_line_idx.tolist())}")

    # ── Nominal load values ───────────────────────────────────────────────────
    nominal_p = net.load['p_mw'].values.copy()    # (N_loads,)
    nominal_q = net.load['q_mvar'].values.copy()  # (N_loads,)
    load_bus  = net.load['bus'].values             # mapping: load → bus

    # ── Output arrays ─────────────────────────────────────────────────────────
    bus_voltages  = np.zeros((T, N_buses, 2))   # [vm_pu, va_degree]
    bus_loads     = np.zeros((T, N_buses, 2))   # [P_MW, Q_MVAr] per bus
    line_topology = np.zeros((T, N_lines),  dtype=np.int8)
    line_flows    = np.zeros((T, N_lines, 2))   # [p_from_mw, q_from_mvar]

    # Current topology state: all lines in service at t=0
    current_topology = np.ones(N_lines, dtype=bool)

    # ── Time loop ─────────────────────────────────────────────────────────────
    failed_steps = 0
    t = 0
    while t < T:
        flag = 0
        while flag == 0:
            # --- 1. Topology change event? ---------------------------------------
            if t % change_period == 0:
                # Randomly toggle a subset of the varying lines
                n_toggle = rng.integers(1, max(2, n_varying_edges))
                toggle_idx = rng.choice(varying_line_idx, size=n_toggle, replace=False)
                current_topology[toggle_idx] = ~current_topology[toggle_idx]

                # Safety: ensure at least 80% of lines remain in service
                if current_topology.sum() < int(0.80 * N_lines):
                    current_topology[toggle_idx] = ~current_topology[toggle_idx]  # revert

            # Apply topology to network
            net.line['in_service'] = current_topology.tolist()

            # --- 2. Stochastic load variation ------------------------------------
            load_factor_p = 1.0 + rng.normal(0, load_variability, size=len(nominal_p))
            load_factor_q = 1.0 + rng.normal(0, load_variability, size=len(nominal_q))
            load_factor_p = np.clip(load_factor_p, 0.1, 2.5)   # physical bounds
            load_factor_q = np.clip(load_factor_q, 0.1, 2.5)

            net.load['p_mw']   = nominal_p * load_factor_p
            net.load['q_mvar'] = nominal_q * load_factor_q

            # --- 3. Run power flow -----------------------------------------------
            try:
                pp.runpp(net, algorithm='nr', numba=False, max_iteration=50)
            except Exception:
                # Convergence failure: revert topology change and retry
                current_topology[toggle_idx if t % change_period == 0 else []] = \
                    ~current_topology[toggle_idx if t % change_period == 0 else []]
                net.line['in_service'] = current_topology.tolist()
                try:
                    pp.runpp(net, algorithm='nr', numba=False, max_iteration=100)
                except Exception:
                    failed_steps += 1
                    # Skip this step, retry with fresh loads
                    continue

            # --- 4. Extract results ----------------------------------------------
            # Bus state variables
            vm  = net.res_bus['vm_pu'].values.copy()      # voltage magnitude
            va_degree  = net.res_bus['va_degree'].values.copy()  # voltage angle
            va = np.deg2rad(va_degree)
            if not np.isnan(np.sum(vm)):
                flag = 1
            else:
                print(vm)



        # # Add measurement noise
        # vm_noisy = add_noise(vm, noise_std, rng)
        # va_noisy = add_noise(va, noise_std, rng)
        # bus_voltages[t, :, 0] = vm_noisy
        # bus_voltages[t, :, 1] = va_noisy
        bus_voltages[t, :, 0] = vm
        bus_voltages[t, :, 1] = va

        # Net nodal power injection (P_gen - P_load per bus)
        bus_loads[t, :, 0] = -add_noise(0.01 * net.res_bus['p_mw'].values.copy(), noise_std, rng)
        bus_loads[t, :, 1] = -add_noise(0.01 * net.res_bus['q_mvar'].values.copy(), noise_std, rng)

        # Line topology
        line_topology[t] = current_topology.astype(np.int8)

        # # Line flows (ground truth edge signals)
        # p_from = net.res_line['p_from_mw'].values.copy()
        # q_from = net.res_line['q_from_mvar'].values.copy()
        # # Zero out disconnected lines
        # p_from[~current_topology] = 0.0
        # q_from[~current_topology] = 0.0
        # line_flows[t, :, 0] = p_from
        # line_flows[t, :, 1] = q_from

        t += 1
        if t % 10 == 0:
            print(f"  t = {t}/{T}")
    #
    # Find indices to keep (first occurrence of each duplicate)
    keep_mask = line_ids_unique['repetitions'] == 1


    keep_indices = list(line_ids_unique[keep_mask].index)
    single_from_pair = []
    for _, group in line_ids_unique[line_ids_unique['repetitions'] > 1].iterrows():
        # Keep only first occurrence of each duplicate
        first_idx = line_ids_unique[
            (line_ids_unique['from_bus'] == group['from_bus']) &
            (line_ids_unique['to_bus'] == group['to_bus'])
            ].index[0]
        single_from_pair.append(first_idx)
    single_from_pair = list(set(single_from_pair))
    keep_indices = keep_indices+single_from_pair
    keep_indices = np.sort(np.array(keep_indices)).astype(int)

    # Apply to both
    line_ids_filtered = line_ids_unique.loc[keep_indices].drop(columns='repetitions').reset_index(drop=True)
    line_topology_filtered = line_topology[:, keep_indices]
    # ── Save arrays ───────────────────────────────────────────────────────────
    arrays = {
        "bus_voltages":  bus_voltages,    # (T, N_buses, 2)   noisy observations
        "bus_loads":     bus_loads,       # (T, N_buses, 2)   load per bus
        "line_topology": line_topology_filtered,   # (T, N_lines)      binary topology
        # "line_flows":    line_flows,      # (T, N_lines, 2)   GT edge signals
        # "line_params":   line_params,     # (N_lines, 4)      static edge params
        "bus_ids":       bus_ids,         # (N_buses,)
        "line_ids":      line_ids_filtered,        # (N_lines, 2)
        "Y_mat":         Y_mat,           # (N_buses,N_buses)
    }

    for name, arr in arrays.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
        if hasattr(arr, 'dtypes'):
            print(f"Saved  {path}   shape={arr.shape}  dtype={arr.dtypes.to_dict()}")
        else:
            print(f"Saved  {path}   shape={arr.shape}  dtype={arr.dtype}")


    # Save metadata
    meta = {
        "slack_bus":         int(net.ext_grid['bus'][0]),
        "T":                 T,
        "N_buses":           N_buses,
        "N_lines":           N_lines,
        "noise_std":         noise_std,
        "varying_edge_set":  varying_edge_set,
        "n_varying_edges":   n_varying_edges,
        "varying_line_idx":  varying_line_idx.tolist(),
        "change_period":     change_period,
        "load_variability":  load_variability,
        "seed":              seed,
        "failed_steps":      failed_steps,
        "arrays": {
            "bus_voltages":  "shape (T, N_buses, 2)  -> [vm_pu, va_degree] with noise",
            "bus_loads":     "shape (T, N_buses, 2)  -> [P_MW, Q_MVAr] per bus",
            "line_topology": "shape (T, N_lines)     -> 1=in_service, 0=disconnected",
            "line_flows":    "shape (T, N_lines, 2)  -> [p_from_MW, q_from_MVAr] GT edge signals",
            "line_params":   "shape (N_lines, 4)     -> [r_ohm/km, x_ohm/km, c_nF/km, length_km]",
            "bus_ids":       "shape (N_buses,)        -> pandapower bus indices",
            "line_ids":      "shape (N_lines, 2)      -> [from_bus, to_bus]",
        }
    }
    import json
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")
    print(f"\nDone!  Failed/skipped steps: {failed_steps}")
    print(f"Dataset ready in:  {os.path.abspath(output_dir)}")

    return arrays


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    for seed_val in range(100):
        generate_dataset(
            T                = args.T,
            noise_std        = args.noise,
            varying_edge_set = args.varying_edge_set,
            n_varying_edges  = args.n_varying_edges,
            change_period    = args.change_period,
            load_variability = args.load_variability,
            seed             = seed_val,
            output_dir       = f"{args.output_dir}{seed_val}",
        )