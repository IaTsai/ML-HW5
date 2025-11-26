"""
Run All Tasks for Machine Learning HW5
Author: Enhanced implementation
Date: 2025-11-26

This script runs both Gaussian Process and SVM tasks sequentially.
"""

import sys
import time
from datetime import datetime


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80)


def main():
    """Run all tasks."""
    start_time = time.time()

    print_header("Machine Learning HW5 - Gaussian Process & SVM")
    print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # ==================== Gaussian Process ====================
        print_header("Part I: Gaussian Process Regression")
        print("\nImporting gaussian_process module...")

        import gaussian_process as gp

        print("\nRunning Gaussian Process tasks...")
        gp_results = gp.main()

        print("\n✓ Gaussian Process completed successfully!")
        print(f"  - Initial NLL: {gp_results['nll_initial']:.3f}")
        print(f"  - Optimized NLL: {gp_results['nll_optimized']:.3f}")
        print(f"  - Improvement: {gp_results['nll_initial'] - gp_results['nll_optimized']:.3f}")

    except Exception as e:
        print(f"\n✗ Error in Gaussian Process: {e}")
        import traceback
        traceback.print_exc()

    # Wait a bit before starting SVM
    print("\n" + "-"*80)
    print("Preparing for SVM tasks...")
    time.sleep(2)

    try:
        # ==================== SVM ====================
        print_header("Part II: SVM on MNIST Dataset")
        print("\nImporting svm_mnist module...")

        import svm_mnist as svm

        print("\nRunning SVM tasks...")
        svm_results = svm.main()

        print("\n✓ SVM classification completed successfully!")
        print(f"  - Task 1 best: {max(svm_results['task1_results'].values()):.2f}%")
        print(f"  - Task 2 test acc: {svm_results['task2_test_acc']:.2f}%")
        print(f"  - Task 3 test acc: {svm_results['task3_test_acc']:.2f}%")

    except Exception as e:
        print(f"\n✗ Error in SVM: {e}")
        import traceback
        traceback.print_exc()

    # ==================== Summary ====================
    end_time = time.time()
    elapsed_time = end_time - start_time

    print_header("Execution Summary")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Execution Time: {elapsed_time/60:.2f} minutes ({elapsed_time:.1f} seconds)")

    print("\n" + "="*80)
    print("All tasks completed!".center(80))
    print("="*80)


if __name__ == "__main__":
    main()
