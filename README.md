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

| File                           | Description                                                                                                                                                        |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `main.py`                     | Entry point for running Monte Carlo simulations and performance evaluation across various experimental setups (e.g., noise, sparsity, topology dynamics).         |
| `EKF_modules.py`              | Contains the implementation of filtering algorithms: standard EKF, fast EKF, sparsity-aware EKF (thresholding and ISTA-based), and Oracle-aided EKF variants.     |
| `change_detection_module.py`  | Implements the baseline convex graph change detection method from [1] with $\ell_1$ and nuclear norm regularization using CVXPY.                                          |
| `constants.py`                | Defines experimental configurations (linear and nonlinear cases) and maps method names to filtering modules via `METHOD_REGISTRY`.                                 |
| `util_func.py`                | Provides utility functions for Jacobian computation, polynomial filtering, simulation, Kalman gain, trajectory generation, evaluation metrics, and parallel setup.  |
| `ploting_results.py`          | Generates and saves performance plots (MSE, F1-score, EIER, runtime) across configurations and scenarios; handles merged and saved result data.                   |


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
