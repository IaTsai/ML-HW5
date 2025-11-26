"""
Support Vector Machine Classification on MNIST Handwritten Digits
===================================================================

This module implements Support Vector Machine (SVM) multi-class classification on a subset of the MNIST handwritten digit dataset (digits 0-4). 
This implementation uses LIBSVM library and explores different kernel functions for non-linear classification.

Mathematical Background
-----------------------
Support Vector Machines find the optimal separating hyperplane by solving:

    min_{w,b} ½||w||² + C∑ξᵢ
    subject to: yᵢ(wᵀφ(xᵢ) + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

where:
    - w: weight vector in feature space
    - b: bias term
    - C: regularization parameter (controls overfitting vs. margin)
    - ξᵢ: slack variables (allow misclassification)
    - φ(x): feature map induced by kernel function k(x,x') = ⟨φ(x),φ(x')⟩

Kernel Functions Implemented
-----------------------------
1. **Linear Kernel**: k(x,x') = xᵀx'
   - Suitable for linearly separable data
   - Fastest training, no hyperparameters except C
   - Interpretable weights in original space

2. **Polynomial Kernel**: k(x,x') = (γxᵀx' + r)^d
   - Captures polynomial interactions up to degree d
   - Parameters: C (regularization), γ (scaling), d (degree), r (offset)
   - Can overfit if degree too high

3. **RBF (Gaussian) Kernel**: k(x,x') = exp(-γ||x-x'||²)
   - Universal approximator, handles non-linear boundaries
   - Parameters: C (regularization), γ (inverse bandwidth)
   - Most commonly used for complex classification

4. **Custom Kernel (Linear + RBF)**: k(x,x') = xᵀx' + exp(-γ||x-x'||²)
   - Combines linear and non-linear components
   - Can capture both global linear trends and local non-linearities
   - Requires precomputed kernel matrix format

Implementation Details
----------------------
Task 1: Baseline Evaluation
    - Train SVM with default parameters (C=1)
    - Test Linear, Polynomial (degree=3), and RBF kernels
    - Evaluate on hold-out test set

Task 2: Hyperparameter Optimization
    - Grid search over C, γ, and degree parameters
    - 5-fold cross-validation for model selection
    - Train final model with best hyperparameters
    - Expected: RBF with tuned parameters performs best

Task 3: Custom Kernel Design
    - Implement composite kernel (Linear + RBF)
    - Format as precomputed kernel matrix for LIBSVM (-t 4)
    - Grid search for optimal C and γ
    - Compare performance with standard kernels

Dataset Information
-------------------
- Training samples: 5000 images (1000 per class for digits 0-4)
- Test samples: 500 images (100 per class)
- Features: 784 dimensions (28×28 pixel intensities, flattened)
- Labels: {0, 1, 2, 3, 4}
- Preprocessing: Pixel values scaled to [0, 1]
"""

import gc  # Garbage collection for memory management
import numpy as np  # Numerical operations
from libsvm.svmutil import *  # LIBSVM: svm_train, svm_predict, svm_model
from scipy.spatial.distance import cdist  # Efficient pairwise distances
import pandas as pd  # CSV data loading
from datetime import datetime  # Timestamp logging


def reset_environment():
    """
    Reset environment to ensure reproducibility.

    This function:
    - Collects garbage to free memory
    - Sets random seed for reproducibility
    """
    gc.collect()
    np.random.seed(42)


def grid_search(X_train, Y_train, kernel_type, param_grid):
    """
    Perform grid search to find the best hyperparameters using 5-fold cross-validation.

    Parameters:
    -----------
    X_train : numpy.ndarray
        Training data features
    Y_train : numpy.ndarray
        Training data labels
    kernel_type : int
        Kernel type (0: linear, 1: polynomial, 2: RBF, 4: custom)
    param_grid : dict
        Dictionary containing parameter ranges to search
        Example: {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}

    Returns:
    --------
    best_params : dict
        Best parameters found
    best_acc : float
        Best cross-validation accuracy
    """
    best_acc = 0
    best_params = None
    total_combinations = len(param_grid['C']) * \
                        len(param_grid.get('gamma', [None])) * \
                        len(param_grid.get('degree', [None]))
    current = 0

    kernel_names = {0: 'Linear', 1: 'Polynomial', 2: 'RBF', 4: 'Custom'}
    print(f"\n  Grid search for {kernel_names.get(kernel_type, 'Unknown')} kernel:")
    print(f"  Total combinations to test: {total_combinations}")

    for C in param_grid['C']:
        for gamma in param_grid.get('gamma', [None]):
            for degree in param_grid.get('degree', [None]):
                current += 1

                # Build options string for libsvm
                options = f'-t {kernel_type} -c {C}'

                if gamma is not None:
                    options += f' -g {gamma}'
                if degree is not None:
                    options += f' -d {degree}'

                # Add 5-fold cross-validation and quiet mode
                options += ' -v 5 -q'

                # Print progress
                print(f"  [{current}/{total_combinations}] Testing: C={C}", end='')
                if gamma is not None:
                    print(f", gamma={gamma}", end='')
                if degree is not None:
                    print(f", degree={degree}", end='')

                # Train with custom kernel or standard kernel
                if kernel_type == 4:
                    # For custom kernel, need to format data with indices
                    formatted_X_train = np.hstack((
                        np.arange(1, X_train.shape[0] + 1).reshape(-1, 1),
                        custom_kernel(X_train, X_train, gamma)
                    ))
                    acc = svm_train(Y_train, formatted_X_train.tolist(), options)
                else:
                    acc = svm_train(Y_train, X_train, options)

                print(f" -> Accuracy: {acc:.2f}%")

                # Update best parameters if current is better
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'C': C}

                    if gamma is not None:
                        best_params['gamma'] = gamma
                    if degree is not None:
                        best_params['degree'] = degree

    return best_params, best_acc


def custom_kernel(x1, x2, gamma=0.01):
    """
    Compute custom composite kernel combining linear and RBF (Gaussian) kernels.

    This hybrid kernel captures both global linear structure and local non-linear patterns.
    The additive combination allows the SVM to learn from both kernel spaces simultaneously.

    Mathematical Formulation
    ------------------------
    k_custom(x, x') = k_linear(x, x') + k_rbf(x, x')
                    = xᵀx' + exp(-γ||x - x'||²)

    where:
    - First term (linear): captures global linear trends and correlations
    - Second term (RBF): captures local non-linear similarities
    - γ (gamma): controls the bandwidth of the RBF component

    Kernel Properties
    -----------------
    - Symmetric: k(x,x') = k(x',x)
    - Positive semi-definite (sum of valid kernels)
    - Universal approximator (due to RBF component)
    - Interpretable (linear component provides global structure)

    Design Rationale
    ----------------
    For MNIST digit classification:
    - Linear component: helps with linearly separable digit pairs (e.g., 0 vs 1)
    - RBF component: handles complex boundaries (e.g., 3 vs 8, 4 vs 9)
    - Equal weighting (1:1): simplest baseline for kernel combination
    - Could be improved with learnable weights: w₁·k_linear + w₂·k_rbf

    Parameters
    ----------
    x1 : numpy.ndarray, shape (n_samples_1, n_features)
        First set of data points (e.g., training samples or test samples).
    x2 : numpy.ndarray, shape (n_samples_2, n_features)
        Second set of data points (e.g., training samples).
        For training: x1 = x2 = X_train (square matrix)
        For testing: x1 = X_test, x2 = X_train (rectangular matrix)
    gamma : float, default=0.01
        RBF kernel bandwidth parameter. Controls the reach of influence:
        - Small γ (e.g., 0.001): smooth, wide influence (may underfit)
        - Medium γ (e.g., 0.01): balanced (often optimal)
        - Large γ (e.g., 0.1): tight influence (may overfit)
        Typical range for normalized MNIST: [0.001, 0.1]

    Returns
    -------
    kernel_matrix : numpy.ndarray, shape (n_samples_1, n_samples_2)
        Precomputed kernel matrix K[i,j] = k_custom(x1[i], x2[j]).
        This matrix is used with LIBSVM's precomputed kernel option (-t 4).

    Examples
    --------
    >>> X_train = np.random.rand(100, 784)  # 100 samples, 784 features
    >>> X_test = np.random.rand(20, 784)    # 20 test samples
    >>> K_train = custom_kernel(X_train, X_train, gamma=0.01)  # (100, 100)
    >>> K_test = custom_kernel(X_test, X_train, gamma=0.01)     # (20, 100)
    >>> print(f"Train kernel: {K_train.shape}, Test kernel: {K_test.shape}")

    Notes
    -----
    - Computational complexity: O(n1 * n2 * d) where d is feature dimension
    - For MNIST (d=784), this is more expensive than built-in LIBSVM kernels
    - Memory usage: O(n1 * n2) for storing full kernel matrix
    - Kernel values are not normalized (consider scaling if needed)
    - Alternative combinations: multiplicative (k₁ · k₂), weighted sum (w₁k₁ + w₂k₂)

    LIBSVM Usage
    ------------
    To use with LIBSVM precomputed kernel format:
    1. Compute kernel matrix: K = custom_kernel(X, X_train, gamma)
    2. Add sample indices: K_formatted = np.hstack([np.arange(1, n+1).reshape(-1,1), K])
    3. Train/predict: svm_train(Y, K_formatted.tolist(), '-t 4 -c 1')

    Performance Considerations
    --------------------------
    - Linear component scales O(d) per evaluation (efficient for high dimensions)
    - RBF component scales O(d) per evaluation but requires distance computation
    - Precomputation trades memory for speed during SVM training
    - For large datasets, consider approximate methods (e.g., Nyström approximation)

    See Also
    --------
    np.dot : Linear kernel computation (matrix multiplication)
    scipy.spatial.distance.cdist : Efficient pairwise distance computation
    """
    # Step 1: Compute linear kernel component
    # k_linear(x,x') = xᵀx' using efficient matrix multiplication
    # Result shape: (n_samples_1, n_samples_2)
    linear_kernel = np.dot(x1, x2.T)

    # Step 2: Compute RBF (Gaussian) kernel component
    # k_rbf(x,x') = exp(-γ * ||x - x'||²)
    # Using cdist with 'sqeuclidean' metric for efficiency (avoids sqrt)
    # squared_distances[i,j] = ||x1[i] - x2[j]||²
    squared_distances = cdist(x1, x2, metric='sqeuclidean')

    # Apply exponential with gamma parameter
    rbf_kernel = np.exp(-gamma * squared_distances)

    # Step 3: Combine kernels via addition (equal weighting)
    # This is a valid positive semi-definite kernel (closure under addition)
    kernel_matrix = linear_kernel + rbf_kernel

    return kernel_matrix


def format_kernel_params(kernel_name, params):
    """Format kernel parameters for display."""
    param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
    return f"{kernel_name} ({param_str})"


def main():
    """Main function to execute SVM classification tasks."""

    print("="*80)
    print("SVM on MNIST Dataset (Digits 0-4)")
    print("="*80)

    # ==================== Load Data ====================
    print("\n[Step 1/6] Loading data...")
    try:
        X_train = pd.read_csv('./ML_HW05/data/X_train.csv', header=None).values
        Y_train = pd.read_csv('./ML_HW05/data/Y_train.csv', header=None).values.ravel()
        X_test = pd.read_csv('./ML_HW05/data/X_test.csv', header=None).values
        Y_test = pd.read_csv('./ML_HW05/data/Y_test.csv', header=None).values.ravel()

        print(f"  Training data: X_train.shape = {X_train.shape}, Y_train.shape = {Y_train.shape}")
        print(f"  Test data: X_test.shape = {X_test.shape}, Y_test.shape = {Y_test.shape}")
        print(f"  Classes: {np.unique(Y_train)}")
    except Exception as e:
        print(f"  Error loading data: {e}")
        return

    # ==================== Task 1 ====================
    print("\n" + "="*80)
    print("TASK 1: Training and Evaluating with Different Kernels")
    print("="*80)

    reset_environment()
    print("\n[Step 2/6] Task 1 - Testing different kernels with default parameters...")

    kernels = {'Linear': 0, 'Polynomial': 1, 'RBF': 2}
    task1_results = {}

    for kernel_name, kernel_type in kernels.items():
        print(f"\n  Training with {kernel_name} kernel...")
        print(f"    Parameters: C=1 (default)")

        # Train model
        model = svm_train(Y_train, X_train, f'-t {kernel_type} -c 1 -q')

        # Evaluate model
        print(f"    Evaluating on test set...")
        p_label, p_acc, p_val = svm_predict(Y_test, X_test, model, '-q')

        task1_results[kernel_name] = p_acc[0]
        print(f"    {kernel_name} kernel accuracy: {p_acc[0]:.2f}%")
        print(f"    " + "-"*60)

    # Display Task 1 summary
    print(f"\n  [Task 1 Summary]")
    print(f"  {'-'*60}")
    for kernel_name, acc in task1_results.items():
        print(f"  {kernel_name:12s}: {acc:6.2f}%")
    print(f"  {'-'*60}")
    best_task1 = max(task1_results.items(), key=lambda x: x[1])
    print(f"  Best kernel: {best_task1[0]} ({best_task1[1]:.2f}%)")

    # ==================== Task 2 ====================
    print("\n" + "="*80)
    print("TASK 2: Grid Search for Best Parameters")
    print("="*80)

    reset_environment()
    print("\n[Step 3/6] Task 2 - Grid search with 5-fold cross-validation...")

    # Define parameter grids for each kernel
    param_grid_task2 = {
        'Linear': {'C': [0.1, 1, 10]},
        'Polynomial': {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1], 'degree': [2, 3]},
        'RBF': {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
    }

    task2_results = {}

    for kernel_name, kernel_type in kernels.items():
        print(f"\n  Performing grid search for {kernel_name} kernel...")

        # Perform grid search
        best_params_task2, best_acc_task2 = grid_search(
            X_train, Y_train, kernel_type, param_grid_task2[kernel_name]
        )

        task2_results[kernel_name] = {
            'params': best_params_task2,
            'cv_acc': best_acc_task2
        }

        print(f"\n  Best parameters for {kernel_name} kernel: {best_params_task2}")
        print(f"  Best 5-fold cross-validation accuracy: {best_acc_task2:.2f}%")

    # Test best RBF model on test set
    print(f"\n[Step 4/6] Testing best RBF model on test set...")
    print(f"  Training with RBF kernel (C=10, gamma=0.01)...")

    model_task2 = svm_train(Y_train, X_train, f'-t 2 -c 10 -g 0.01 -q')
    p_label, p_acc, p_val = svm_predict(Y_test, X_test, model_task2, '-q')

    task2_test_acc = p_acc[0]
    print(f"  RBF kernel test accuracy: {task2_test_acc:.2f}%")

    # Display Task 2 summary
    print(f"\n  [Task 2 Summary]")
    print(f"  {'-'*70}")
    print(f"  {'Kernel':<12} {'Best Params':<35} {'CV Acc':>8}")
    print(f"  {'-'*70}")
    for kernel_name, result in task2_results.items():
        params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
        print(f"  {kernel_name:<12} {params_str:<35} {result['cv_acc']:>7.2f}%")
    print(f"  {'-'*70}")
    print(f"  Best RBF model test accuracy: {task2_test_acc:.2f}%")

    # ==================== Task 3 ====================
    print("\n" + "="*80)
    print("TASK 3: Custom Kernel (Linear + RBF)")
    print("="*80)

    reset_environment()
    print("\n[Step 5/6] Task 3 - Using custom kernel with grid search...")

    print(f"\n  Custom kernel formula: k(x, x') = k_linear(x, x') + k_rbf(x, x')")
    print(f"  where:")
    print(f"    - k_linear(x, x') = x^T * x'")
    print(f"    - k_rbf(x, x') = exp(-gamma * ||x - x'||²)")

    param_grid_task3 = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}

    best_params_task3, best_acc_task3 = grid_search(
        X_train, Y_train, 4, param_grid_task3
    )

    print(f"\n  Best parameters for custom kernel: {best_params_task3}")
    print(f"  Best 5-fold cross-validation accuracy: {best_acc_task3:.2f}%")

    # Train final model with best parameters
    print(f"\n[Step 6/6] Training final custom kernel model...")
    formatted_X_train = np.hstack((
        np.arange(1, X_train.shape[0] + 1).reshape(-1, 1),
        custom_kernel(X_train, X_train, best_params_task3['gamma'])
    ))

    model_task3 = svm_train(
        Y_train,
        formatted_X_train.tolist(),
        f'-t 4 -c {best_params_task3["C"]} -g {best_params_task3["gamma"]} -q'
    )

    # Evaluate on test set
    print(f"  Evaluating custom kernel on test set...")
    formatted_X_test = np.hstack((
        np.arange(1, X_test.shape[0] + 1).reshape(-1, 1),
        custom_kernel(X_test, X_train, best_params_task3['gamma'])
    ))

    p_label, p_acc, p_val = svm_predict(Y_test, formatted_X_test.tolist(), model_task3, '-q')
    task3_test_acc = p_acc[0]
    print(f"  Custom kernel test accuracy: {task3_test_acc:.2f}%")

    # ==================== Final Summary ====================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\n  Task 1 - Default Parameters:")
    print(f"  {'-'*60}")
    for kernel_name, acc in task1_results.items():
        print(f"    {kernel_name:12s}: {acc:6.2f}%")

    print(f"\n  Task 2 - Optimized Parameters (Cross-Validation):")
    print(f"  {'-'*60}")
    for kernel_name, result in task2_results.items():
        params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
        print(f"    {kernel_name:12s} ({params_str}): {result['cv_acc']:6.2f}%")
    print(f"    RBF Test Accuracy: {task2_test_acc:.2f}%")

    print(f"\n  Task 3 - Custom Kernel:")
    print(f"  {'-'*60}")
    params_str = ", ".join([f"{k}={v}" for k, v in best_params_task3.items()])
    print(f"    Custom ({params_str}): CV={best_acc_task3:.2f}%, Test={task3_test_acc:.2f}%")

    print(f"\n  Overall Best Model:")
    print(f"  {'-'*60}")
    all_test_accs = {'Task1_RBF': task1_results.get('RBF', 0),
                     'Task2_RBF': task2_test_acc,
                     'Task3_Custom': task3_test_acc}
    best_overall = max(all_test_accs.items(), key=lambda x: x[1])
    print(f"    {best_overall[0]}: {best_overall[1]:.2f}%")

    print("\n" + "="*80)
    print("SVM Classification Completed Successfully!")
    print("="*80)

    return {
        'task1_results': task1_results,
        'task2_results': task2_results,
        'task2_test_acc': task2_test_acc,
        'task3_params': best_params_task3,
        'task3_cv_acc': best_acc_task3,
        'task3_test_acc': task3_test_acc
    }


if __name__ == "__main__":
    results = main()
