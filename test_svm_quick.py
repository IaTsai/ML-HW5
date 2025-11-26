"""
Quick test for SVM implementation (Task 1 only)
This runs faster for initial testing
"""

import gc
import numpy as np
from libsvm.svmutil import *
import pandas as pd


def reset_environment():
    gc.collect()
    np.random.seed(42)


print("="*80)
print("Quick SVM Test (Task 1 Only)")
print("="*80)

# Load data
print("\nLoading data...")
X_train = pd.read_csv('./ML_HW05/data/X_train.csv', header=None).values
Y_train = pd.read_csv('./ML_HW05/data/Y_train.csv', header=None).values.ravel()
X_test = pd.read_csv('./ML_HW05/data/X_test.csv', header=None).values
Y_test = pd.read_csv('./ML_HW05/data/Y_test.csv', header=None).values.ravel()

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

# Task 1 - Test different kernels
print("\n" + "="*80)
print("TASK 1: Testing Different Kernels")
print("="*80)

reset_environment()
kernels = {'Linear': 0, 'Polynomial': 1, 'RBF': 2}

for kernel_name, kernel_type in kernels.items():
    print(f"\n  Training with {kernel_name} kernel...")
    model = svm_train(Y_train, X_train, f'-t {kernel_type} -c 1 -q')

    print(f"  Evaluating with {kernel_name} kernel...")
    p_label, p_acc, p_val = svm_predict(Y_test, X_test, model, '-q')

    print(f"  {kernel_name} kernel accuracy: {p_acc[0]:.2f}%")
    print(f"  " + "-"*60)

print("\n" + "="*80)
print("Quick Test Completed!")
print("="*80)
