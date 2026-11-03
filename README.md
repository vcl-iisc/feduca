# FedUCA: Federated Learning by Utility-Constrained Stochastic Aggregation for Improving Rational Participation

FedUCA is a first-principles theoretical framework for managing endogenous participation in federated learning through non-monetary incentives. It formulates server-side optimization to balance incentive distribution against client participation dynamics, and provides theoretical convergence guarantees for stochastic aggregation while optimizing overall system performance.

## 1. Environment Setup

This code is adapted from FedDyn. Before running any scripts, initialize the local environment by creating the directories used for dataset storage and experiment logging. From the root of the project, run:

```bash
mkdir Data
mkdir Output
```

- `Data/`: Starts empty. The execution scripts automatically download and format the corresponding datasets into this directory.
- `Output/`: Destination for all experiment logs, baseline comparisons, and generated plots.

## 2. File Structure and Modules

### `incentive_cifar_10.py` / `incentive_cifar_100.py` / `incentive_fmnist.py`
Main execution scripts. Each one initializes the specified dataset (CIFAR-10, CIFAR-100, or Fashion-MNIST) and then runs both the standard baselines and the FedUCA framework in sequence.

### `utils_methods.py`
Core implementations of all baseline federated learning methods with rationality, covering both the baseline algorithms and the FedUCA framework.

### `utils_game.py`
Server-side optimization logic and the game-theoretic functions used to compute and distribute non-monetary incentives to clients.

## 3. Usage

Run the experiments by executing the script for the dataset you want to evaluate. For example, to evaluate the framework on CIFAR-10:

```bash
python incentive_cifar_10.py
```

Logs, baseline comparisons, and plots for the run are written to `Output/`.
