"""
Gaussian Process Regression with Rational Quadratic Kernel
Author: Enhanced implementation based on previous work
Date: 2025-11-26

This script implements Gaussian Process Regression using Rational Quadratic Kernel
for Machine Learning HW5.

Tasks:
1. Task 1: Apply GPR with initial parameters and visualize
2. Task 2: Optimize kernel parameters by minimizing negative marginal log-likelihood
"""

from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def rational_quadratic_kernel(x1, x2, sigma, alpha, length_scale):
    """
    Computes the Rational Quadratic kernel between two sets of points.

    The Rational Quadratic kernel is defined as:
    k(x, x') = σ² * (1 + ||x - x'||² / (2 * α * l²))^(-α)

    where:
    - σ (sigma): amplitude parameter
    - α (alpha): shape parameter controlling smoothness
    - l (length_scale): characteristic length scale

    Parameters:
    -----------
    x1, x2 : numpy.ndarray
        Input data, can be of different lengths
    sigma : float
        The amplitude parameter controlling the overall scale of the kernel
    alpha : float
        The shape parameter controlling the kernel's flexibility
    length_scale : float
        Length scale controls how far apart inputs must be to be considered correlated

    Returns:
    --------
    kernel : numpy.ndarray
        Kernel matrix K(x1, x2)
    """
    # Compute pairwise Euclidean distances between x1 and x2
    distance = cdist(x1, x2, 'euclidean')

    # Apply Rational Quadratic kernel formula
    kernel = (sigma**2) * (1 + distance**2 / (2 * alpha * length_scale**2))**(-alpha)

    return kernel


def gaussian_process_regression(X_train, Y_train, X_pred, sigma=1.0, alpha=1.0, length_scale=1.0, beta=5):
    """
    Perform Gaussian Process Regression to predict the distribution of f at X_pred.

    Given training data (X_train, Y_train) where Y_train = f(X_train) + ε,
    and ε ~ N(0, β^(-1)), this function computes the posterior predictive distribution
    p(f(X_pred) | X_train, Y_train).

    Formulas (from Kernel_GP_SVM.pdf page 48):
    - C(xn, xm) = k(xn, xm) + β^(-1) * δnm
    - μ(x*) = k(x, x*)^T * C^(-1) * y
    - σ²(x*) = k(x*, x*) + β^(-1) - k(x, x*)^T * C^(-1) * k(x, x*)

    Parameters:
    -----------
    X_train : numpy.ndarray, shape (n, 1)
        Training input data
    Y_train : numpy.ndarray, shape (n, 1)
        Training output data (noisy observations)
    X_pred : numpy.ndarray, shape (m, 1)
        Prediction points
    sigma : float
        Amplitude parameter for kernel
    alpha : float
        Shape parameter for kernel
    length_scale : float
        Length scale parameter for kernel
    beta : float
        Noise precision (inverse of observation noise variance)

    Returns:
    --------
    mu_s : numpy.ndarray, shape (m, 1)
        Mean of the posterior predictive distribution at X_pred
    cov_s : numpy.ndarray, shape (m, m)
        Covariance of the posterior predictive distribution at X_pred
    """
    # Compute covariance matrices
    # C = K(X, X) + (β^-1)I - Covariance of training observations
    C = rational_quadratic_kernel(X_train, X_train, sigma, alpha, length_scale) + \
        np.eye(len(X_train)) / beta

    # K_s = K(X, X*) - Cross-covariance between training and prediction points
    K_s = rational_quadratic_kernel(X_train, X_pred, sigma, alpha, length_scale)

    # K_ss = K(X*, X*) + (β^-1)I - Covariance of prediction points
    K_ss = rational_quadratic_kernel(X_pred, X_pred, sigma, alpha, length_scale) + \
           np.eye(len(X_pred)) / beta

    # Compute inverse of C
    C_inv = np.linalg.inv(C)

    # Compute the mean and covariance of the posterior distribution
    # μ(x*) = K(X, X*)^T * C^(-1) * Y
    mu_s = (K_s.T).dot(C_inv).dot(Y_train)

    # σ²(x*) = K(X*, X*) + β^(-1) - K(X, X*)^T * C^(-1) * K(X, X*)
    cov_s = K_ss - (K_s.T).dot(C_inv).dot(K_s)

    return mu_s, cov_s


def negative_log_likelihood(params, X_train, Y_train, beta=5):
    """
    Compute the negative log marginal likelihood for the GP with Rational Quadratic kernel.

    The marginal log-likelihood (from Kernel_GP_SVM.pdf page 52):
    ln p(y|θ) = -1/2 * ln|Cθ| - 1/2 * y^T * Cθ^(-1) * y - N/2 * ln(2π)

    We negate this to convert maximization problem to minimization problem
    for use with scipy.optimize.minimize.

    Parameters:
    -----------
    params : list or numpy.ndarray
        List of kernel parameters [sigma, alpha, length_scale]
    X_train : numpy.ndarray
        Training input data
    Y_train : numpy.ndarray
        Training output data
    beta : float
        Noise precision (inverse of observation noise variance)

    Returns:
    --------
    NLL : float
        Negative log marginal likelihood (scalar value)
    """
    sigma, alpha, length_scale = params

    # Compute kernel matrix
    K = rational_quadratic_kernel(X_train, X_train, sigma, alpha, length_scale)

    # Add noise term: C = K + β^(-1) * I
    C = K + np.eye(len(X_train)) / beta

    # Compute inverse of C
    C_inv = np.linalg.inv(C)

    # Compute negative log marginal likelihood
    # NLL = 0.5 * (ln|C| + y^T * C^(-1) * y + N * ln(2π))
    NLL = 0.5 * (np.log(np.linalg.det(C)) +
                 (Y_train.T).dot(C_inv).dot(Y_train) +
                 len(C) * np.log(2 * np.pi))

    # Extract scalar value from (1,1) matrix
    return NLL[0, 0]


def plot_gp_result(X, Y, X_pred, mu_s, cov_s, title, save_path=None):
    """
    Visualize Gaussian Process Regression results.

    Parameters:
    -----------
    X : numpy.ndarray
        Training input data points
    Y : numpy.ndarray
        Training output data points
    X_pred : numpy.ndarray
        Prediction points
    mu_s : numpy.ndarray
        Mean predictions
    cov_s : numpy.ndarray
        Covariance matrix of predictions
    title : str
        Plot title
    save_path : str, optional
        If provided, save the plot to this path
    """
    # Compute standard deviation from covariance diagonal
    std = np.sqrt(np.diag(cov_s))

    plt.figure(figsize=(12, 7))

    # Plot training data points
    plt.scatter(X, Y, color='red', s=50, zorder=3, label='Training Data', alpha=0.8)

    # Plot mean prediction
    plt.plot(X_pred, mu_s, color='blue', linewidth=2, label='Mean Prediction')

    # Plot 95% confidence interval (1.96 * std for normal distribution)
    plt.fill_between(X_pred.flatten(),
                     mu_s.flatten() - 1.96 * std,
                     mu_s.flatten() + 1.96 * std,
                     color='blue', alpha=0.2, label='95% Confidence Interval')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.xlim(-60, 60)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    """Main function to execute Gaussian Process Regression tasks."""

    print("="*80)
    print("Gaussian Process Regression with Rational Quadratic Kernel")
    print("="*80)

    # ==================== Load Data ====================
    print("\n[1/5] Loading data...")
    data = np.loadtxt('./ML_HW05/data/input.data')
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)
    print(f"Data loaded: X.shape = {X.shape}, Y.shape = {Y.shape}")

    # ==================== Task 1 ====================
    print("\n" + "="*80)
    print("TASK 1: Gaussian Process Regression with Initial Parameters")
    print("="*80)

    # Setup parameters
    X_pred = np.linspace(-60, 60, 1000).reshape(-1, 1)
    initial_params = [1.0, 1.0, 1.0]  # [sigma, alpha, length_scale]
    beta = 5

    print(f"\n[2/5] Running GP regression with initial parameters...")
    print(f"  - Sigma (σ) = {initial_params[0]}")
    print(f"  - Alpha (α) = {initial_params[1]}")
    print(f"  - Length Scale (l) = {initial_params[2]}")
    print(f"  - Beta (β) = {beta}")
    print(f"  - Prediction range: [-60, 60] with {len(X_pred)} points")

    # Perform Gaussian Process Regression
    mu_s, cov_s = gaussian_process_regression(X, Y, X_pred,
                                              sigma=initial_params[0],
                                              alpha=initial_params[1],
                                              length_scale=initial_params[2],
                                              beta=beta)

    # Compute negative log likelihood
    nll_initial = negative_log_likelihood(initial_params, X, Y, beta)

    print(f"\n[Task 1 Results]")
    print(f"  Initial Parameters: σ={initial_params[0]}, α={initial_params[1]}, l={initial_params[2]}")
    print(f"  Initial Negative Log-Likelihood: {nll_initial:.3f}")

    # Visualize Task 1 results
    plot_gp_result(X, Y, X_pred, mu_s, cov_s,
                   'Task 1: Gaussian Process Regression with Rational Quadratic Kernel\n' +
                   f'(σ={initial_params[0]}, α={initial_params[1]}, l={initial_params[2]})',
                   save_path='./task1_gp_initial.png')

    # ==================== Task 2 ====================
    print("\n" + "="*80)
    print("TASK 2: Optimizing Kernel Parameters")
    print("="*80)

    print(f"\n[3/5] Optimizing kernel parameters...")
    print(f"  Method: Minimizing negative marginal log-likelihood")
    print(f"  Initial parameters: {initial_params}")
    print(f"  Bounds: [0.01, ∞) for all parameters")
    print(f"  Max iterations: 1000")

    # Optimize kernel parameters
    result = minimize(negative_log_likelihood,
                     initial_params,
                     args=(X, Y, beta),
                     bounds=[(1e-2, None), (1e-2, None), (1e-2, None)],
                     options={'maxiter': 1000})

    optimized_params = result.x
    nll_optimized = negative_log_likelihood(optimized_params, X, Y, beta)

    print(f"\n[4/5] Optimization completed!")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Iterations: {result.nit}")

    print(f"\n[Task 2 Results]")
    print(f"  Optimized Parameters:")
    print(f"    - Sigma (σ) = {optimized_params[0]:.3f}")
    print(f"    - Alpha (α) = {optimized_params[1]:.3f}")
    print(f"    - Length Scale (l) = {optimized_params[2]:.3f}")
    print(f"  Optimized Negative Log-Likelihood: {nll_optimized:.3f}")
    print(f"  Improvement: {nll_initial - nll_optimized:.3f}")

    # Perform GP regression with optimized parameters
    mu_s_optimized, cov_s_optimized = gaussian_process_regression(
        X, Y, X_pred,
        sigma=optimized_params[0],
        alpha=optimized_params[1],
        length_scale=optimized_params[2],
        beta=beta)

    # Visualize Task 2 results
    plot_gp_result(X, Y, X_pred, mu_s_optimized, cov_s_optimized,
                   'Task 2: Optimized Gaussian Process Regression with Rational Quadratic Kernel\n' +
                   f'(σ={optimized_params[0]:.3f}, α={optimized_params[1]:.3f}, l={optimized_params[2]:.3f})',
                   save_path='./task2_gp_optimized.png')

    # ==================== Comparison ====================
    print("\n" + "="*80)
    print("COMPARISON: Task 1 vs Task 2")
    print("="*80)

    print(f"\n[5/5] Summary:")
    print(f"\n  Task 1 (Initial):")
    print(f"    Parameters: σ={initial_params[0]:.3f}, α={initial_params[1]:.3f}, l={initial_params[2]:.3f}")
    print(f"    Negative Log-Likelihood: {nll_initial:.3f}")

    print(f"\n  Task 2 (Optimized):")
    print(f"    Parameters: σ={optimized_params[0]:.3f}, α={optimized_params[1]:.3f}, l={optimized_params[2]:.3f}")
    print(f"    Negative Log-Likelihood: {nll_optimized:.3f}")

    print(f"\n  Improvement:")
    print(f"    NLL Reduction: {nll_initial - nll_optimized:.3f} ({((nll_initial - nll_optimized)/nll_initial * 100):.2f}%)")

    print("\n" + "="*80)
    print("Gaussian Process Regression Completed Successfully!")
    print("="*80)

    return {
        'initial_params': initial_params,
        'optimized_params': optimized_params,
        'nll_initial': nll_initial,
        'nll_optimized': nll_optimized
    }


if __name__ == "__main__":
    results = main()
