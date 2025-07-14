# PolyEKF: Sparsity-Aware Extended Kalman Filter for Tracking Dynamic Graphs
*Lital Dabush, Nir Shlezinger, and Tirza Routtenberg*

This repository contains the official implementation of the Sparsity-Aware Extended Kalman Filter (GSP-EKF) for tracking dynamic graph topologies, as proposed in the paper:

📄 L.Dabush, N. Shlezinger, and T. Routtenberg, ``Sparsity-Aware Extended Kalman Filter for Tracking Dynamic Graphs".

## 🧠 Overview

The proposed method formulates the problem of topology tracking as a sparse nonlinear state-space model under the Graph Signal Processing (GSP) framework. The key contributions include:

- **GSP-EKF**: An EKF that incorporates ℓ₁-regularized updates to promote sparsity in dynamic graphs.
- **Efficient Jacobian Computation**: A dynamic programming-based scheme that enables tractable evaluation of the Jacobian for polynomial graph filters.
- **Observability Analysis**: Theoretical insights into the impact of sparsity and graph dynamics on system observability.
- **Extensive Experiments**: Validated on both linear and nonlinear graph filters across various noise levels and topological change rates.


## 📂 Repository Structure
| File                             | Description                                                                                                                                                       |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`                        | Entry point for simulations. Runs Monte Carlo experiments for various scenarios (e.g., different noise levels, sparsity, etc.) and generates performance metrics. |
| `EKF_modules.py`                 | Contains implementations of the core tracking algorithms, including the proposed sparsity-aware EKF, Oracle-GSP-EKF, EKF and evaluation logic.                    |
| `change_detection_module.py`     | Implements the baseline graph change detection method from [1] using convex optimization with $\ell_1$ and nuclear norm regularization.                           |
| `constants.py`                   | Defines global configurations for different experimental setups (e.g., linear vs nonlinear, varying sparsity, SNR, etc.).                                         |
| `util_func.py`                   | Utility functions for computing metrics, generating plots, dynamic parallelization setup, and Jacobian evaluations.                                               |
| `ploting_results.py`             | Generates evaluation plots (e.g., MSE, F1-score) from saved simulation data.                                                                                      |


## 📘 Citation
If you find this work useful, please cite:

```bibtex
@article{dabush2025gspkf,
  title={Sparsity-Aware Extended Kalman Filter for Tracking Dynamic Graphs},
  author={Dabush, Lital and Shlezinger, Nir and Routtenberg, Tirza},
  year={2025}
}
```


## 📬 Contact
For questions or feedback, feel free to contact litaldab@post.bgu.ac.il.


## 📚 References

[1] Y. Hu and Z. Xiao, “Online directed graph estimation for dynamic network topology inference,” in *Proc. IEEE 98th Veh. Technol. Conf. (VTC-Fall)*, Hong Kong, China, 2023, pp. 1–5.
