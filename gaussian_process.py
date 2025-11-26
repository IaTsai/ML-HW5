"""
Gaussian Process Regression with Rational Quadratic Kernel
===========================================================
Mathematical Background
-----------------------
The Rational Quadratic kernel is a scale mixture of RBF kernels with different length scales.
It can be interpreted as an infinite sum of RBF kernels with different characteristic length
scales, making it particularly useful for modeling functions with multiple length scales.

RQ Kernel Formula:
    k(x, x') = σ² * (1 + ||x - x'||² / (2αl²))^(-α)

where:
    - σ (sigma): Signal variance, controls the overall amplitude
    - α (alpha): Scale-mixture weight, controls smoothness (α→∞ becomes RBF)
    - l (length scale): Characteristic length scale of variation

Implementation Tasks
--------------------
Task 1: Apply GPR with initial parameters (σ=1.0, α=1.0, l=1.0)
    - Compute posterior predictive distribution
    - Visualize mean prediction and 95% confidence intervals
    - Calculate negative marginal log-likelihood

Task 2: Optimize kernel hyperparameters
    - Minimize negative marginal log-likelihood using L-BFGS-B
    - Compare optimized vs. initial predictions
    - Analyze parameter influence on model behavior

Notes
-----
This implementation uses:
- Exact GP inference (matrix inversion) suitable for small-to-medium datasets (n < 10000)
- Numerical stability: Jitter term (β^(-1)) added to diagonal for positive definiteness
- Optimization: L-BFGS-B with bounds to ensure parameter positivity
"""

from scipy.spatial.distance import cdist  # Efficient pairwise distance computation
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting
from scipy.optimize import minimize  # Hyperparameter optimization


def rational_quadratic_kernel(x1, x2, sigma, alpha, length_scale):
    """
    Compute the Rational Quadratic covariance kernel between two sets of input points.

    The Rational Quadratic (RQ) kernel is equivalent to a scale mixture (infinite sum) of
    RBF kernels with different characteristic length scales. It interpolates between the
    RBF kernel (α→∞) and linear kernel behavior for different length scales.

    Mathematical Formulation
    ------------------------
    k(x, x') = σ² * (1 + ||x - x'||² / (2αl²))^(-α)

    where ||x - x'|| is the Euclidean distance between points x and x'.

    Parameters
    ----------
    x1 : numpy.ndarray, shape (n_samples_1, n_features)
        First set of input points. Each row represents a data point.
    x2 : numpy.ndarray, shape (n_samples_2, n_features)
        Second set of input points. Can have different number of samples than x1.
    sigma : float, positive
        Signal standard deviation (amplitude parameter). Controls the average distance
        of the function output from its mean. Typical range: [0.1, 10.0]
    alpha : float, positive
        Shape parameter controlling the relative weighting of different length scales.
        - Small α (< 1): more weight on larger length scales (smoother variations)
        - Large α (> 100): approaches RBF kernel (single length scale)
        Typical range: [0.1, 100.0]
    length_scale : float, positive
        Characteristic length scale parameter. Determines the distance in input space
        over which function values are significantly correlated.
        Typical range: [0.1, 10.0]

    Returns
    -------
    kernel_matrix : numpy.ndarray, shape (n_samples_1, n_samples_2)
        Computed kernel (covariance) matrix K(x1, x2).
        Entry K[i,j] represents the covariance between x1[i] and x2[j].

    Notes
    -----
    - Computational complexity: O(n1 * n2 * d) where d is the feature dimension
    - The kernel is symmetric if x1 == x2: K(x, x') = K(x', x)
    - When α→∞, RQ kernel converges to RBF kernel: k(x,x') = σ² * exp(-||x-x'||²/(2l²))
    - The kernel is positive semi-definite, ensuring valid covariance matrices
    """
    # Step 1: Compute pairwise Euclidean distances using scipy's efficient implementation
    # distance[i,j] = ||x1[i] - x2[j]||_2
    distance = cdist(x1, x2, metric='euclidean')

    # Step 2: Apply Rational Quadratic kernel formula
    # Numerically stable computation: avoid overflow for small length_scale
    scaled_distance_sq = distance**2 / (2 * alpha * length_scale**2)

    # Kernel computation: k(x,x') = σ² * (1 + scaled_distance²)^(-α)
    kernel_matrix = (sigma**2) * np.power(1 + scaled_distance_sq, -alpha)

    return kernel_matrix


def gaussian_process_regression(X_train, Y_train, X_pred, sigma=1.0, alpha=1.0, length_scale=1.0, beta=5):
    """
    Perform Gaussian Process Regression to compute posterior predictive distribution.

    This function implements exact GP inference using the closed-form Bayesian update
    equations. Given noisy observations Y_train = f(X_train) + ε where ε ~ N(0, β^(-1)I),
    we compute the posterior distribution over function values at test points X_pred.

    Mathematical Framework
    ----------------------
    Prior: f ~ GP(0, k(·,·))
    Likelihood: y|f ~ N(f, β^(-1)I)
    Posterior: f*|X,y,X* ~ N(μ*, Σ*)

    where:
        C = K(X,X) + β^(-1)I                    [n × n] observation covariance
        μ* = K(X*,X)ᵀ C^(-1) y                  [m × 1] posterior mean
        Σ* = K(X*,X*) + β^(-1)I - K(X*,X)ᵀ C^(-1) K(X*,X)  [m × m] posterior covariance

    Parameters
    ----------
    X_train : numpy.ndarray, shape (n_samples, n_features)
        Training input locations. Each row is an observation.
    Y_train : numpy.ndarray, shape (n_samples, 1)
        Training output values (noisy observations of the latent function).
    X_pred : numpy.ndarray, shape (n_pred, n_features)
        Test input locations where predictions are desired.
    sigma : float, default=1.0
        Kernel amplitude parameter (signal standard deviation).
    alpha : float, default=1.0
        Kernel shape parameter (scale-mixture weight).
    length_scale : float, default=1.0
        Kernel characteristic length scale.
    beta : float, default=5
        Noise precision parameter (inverse of observation noise variance σ_n²).
        Equivalently, observation noise ~ N(0, β^(-1)).
        Typical range: [1, 100]. Higher β = lower noise assumption.

    Returns
    -------
    mu_pred : numpy.ndarray, shape (n_pred, 1)
        Posterior predictive mean at test locations X_pred.
        This is the expected value: E[f(X_pred) | X_train, Y_train]
    cov_pred : numpy.ndarray, shape (n_pred, n_pred)
        Posterior predictive covariance at test locations X_pred.
        Diagonal elements give pointwise predictive variance: Var[f(x*) | data]
        Off-diagonal elements give predictive covariances between test points.

    Notes
    -----
    - Computational complexity: O(n³) due to matrix inversion, where n = len(X_train)
    - For large datasets (n > 5000), consider sparse GP approximations (e.g., inducing points)
    - The jitter term β^(-1)I ensures numerical stability and positive definiteness
    - The posterior covariance is independent of observed Y values (only depends on X locations)
    - 95% credible intervals: μ ± 1.96 * sqrt(diag(Σ))

    Raises
    ------
    numpy.linalg.LinAlgError
        If the covariance matrix C is singular (extremely rare with jitter term).
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
    Compute negative log marginal likelihood (NLML) for GP hyperparameter optimization.

    The marginal likelihood p(y|X,θ) integrates out the latent function values f,
    providing a principled Bayesian approach to hyperparameter selection. Maximizing
    the marginal likelihood (or equivalently minimizing its negative) automatically
    balances model fit and complexity, preventing overfitting.

    Mathematical Formulation
    ------------------------
    Log marginal likelihood:
        log p(y|X,θ) = -½ log|C_θ| - ½ yᵀC_θ⁻¹y - (n/2)log(2π)

    where:
        - C_θ = K(X,X) + β⁻¹I is the marginal covariance of observations
        - θ = [σ, α, l] are the kernel hyperparameters
        - n is the number of training points

    Decomposition:
        log p(y|X,θ) = log p(y|f) + log p(f|X,θ) - log p(f|X,y,θ)
                     = data fit term + complexity penalty - posterior term

    The three terms in the formula have interpretations:
        1. log|C_θ|: Model complexity (penalizes very flexible models)
        2. yᵀC_θ⁻¹y: Data fit (how well model explains observations)
        3. n·log(2π): Normalization constant (independent of θ)

    Parameters
    ----------
    params : array-like, shape (3,)
        Kernel hyperparameters [sigma, alpha, length_scale] to optimize.
        Must be positive values.
    X_train : numpy.ndarray, shape (n_samples, n_features)
        Training input locations.
    Y_train : numpy.ndarray, shape (n_samples, 1)
        Training output values (noisy observations).
    beta : float, default=5
        Fixed noise precision parameter (not optimized).

    Returns
    -------
    nlml : float
        Negative log marginal likelihood value. Lower values indicate better
        hyperparameters that balance fit and complexity.

    Notes
    -----
    - This function is typically used as the objective for scipy.optimize.minimize
    - Gradient information is not provided (numerical gradients used by L-BFGS-B)
    - For numerical stability, consider log-det via Cholesky: log|C| = 2·sum(log(diag(L)))
    - The function assumes well-conditioned covariance matrices (ensured by jitter β⁻¹I)
    - Optimization landscape is typically non-convex with multiple local minima
    - Typical optimization: minimize(negative_log_likelihood, x0, bounds=[(1e-2, None)]*3)

    Computational Complexity
    ------------------------
    - O(n³) for matrix inversion (dominant cost)
    - O(n²) for kernel matrix computation
    - O(n²) for quadratic form yᵀC⁻¹y
    """
    # Unpack hyperparameters
    sigma, alpha, length_scale = params

    # Step 1: Compute kernel matrix K(X,X) using current hyperparameters
    # K[i,j] = k(x_i, x_j; θ)
    K = rational_quadratic_kernel(X_train, X_train, sigma, alpha, length_scale)

    # Step 2: Add observation noise to form marginal covariance
    # C = K(X,X) + σ_noise²·I = K + β⁻¹·I
    # The jitter term β⁻¹I ensures positive definiteness and numerical stability
    n_samples = len(X_train)
    C = K + np.eye(n_samples) / beta

    # Step 3: Compute matrix inverse (expensive O(n³) operation)
    # In production code, prefer Cholesky decomposition: C = LLᵀ for stability
    C_inv = np.linalg.inv(C)

    # Step 4: Compute the three terms of negative log marginal likelihood
    # Term 1: Complexity penalty - log determinant of C
    # Measures the "volume" of the hypothesis space
    log_det_C = np.log(np.linalg.det(C))

    # Term 2: Data fit term - quadratic form yᵀC⁻¹y
    # Measures how well the model explains the observations
    data_fit = (Y_train.T).dot(C_inv).dot(Y_train)

    # Term 3: Normalization constant (independent of hyperparameters)
    log_2pi_term = n_samples * np.log(2 * np.pi)

    # Compute NLML: -log p(y|X,θ) = ½(log|C| + yᵀC⁻¹y + n·log(2π))
    nlml = 0.5 * (log_det_C + data_fit + log_2pi_term)

    # Extract scalar from (1,1) matrix result
    return nlml[0, 0]


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
