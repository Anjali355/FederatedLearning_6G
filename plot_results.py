"""
Plotting utilities for federated learning results
"""

import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_training_curves(log_file=None, log_csv=None, outdir="artifacts_flower"):
    """Plot training accuracy and loss curves from log file"""
    
    # Support both parameter names for backwards compatibility
    if log_csv:
        log_file = log_csv
    
    if not log_file or not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} not found, skipping plot")
        return
    
    df = pd.read_csv(log_file)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(df['round'], df['test_acc'], marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Global Model Test Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot loss
    ax2.plot(df['round'], df['test_loss'], marker='o', linewidth=2, color='orange', markersize=6)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Global Model Test Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    print(f"[INFO] Training curves saved to {os.path.join(outdir, 'training_curves.png')}")
    plt.close()


def plot_model_eval(model_path=None, X_test=None, y_test=None, outdir="artifacts_flower", results_file=None):
    """Plot final model evaluation results"""
    
    # Legacy support: if results_file is provided, use it
    if results_file and os.path.exists(results_file):
        df = pd.read_csv(results_file)
    elif model_path and X_test is not None and y_test is not None:
        # Evaluate model and create results
        print(f"[INFO] Model evaluation plot skipped (model evaluation not implemented)")
        return
    else:
        print(f"Warning: No valid inputs for model evaluation plot")
        return
    
    if results_file and os.path.exists(results_file):
        df = pd.read_csv(results_file)
    else:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot per-class accuracy
    classes = [col for col in df.columns if col.startswith('class_')]
    if classes:
        accuracies = [df[col].values[0] for col in classes]
        class_labels = [col.replace('class_', 'Class ') for col in classes]
        
        bars = ax.bar(class_labels, accuracies, alpha=0.7, edgecolor='black')
        
        # Color bars based on accuracy
        for bar, acc in zip(bars, accuracies):
            if acc >= 0.9:
                bar.set_color('green')
            elif acc >= 0.7:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Class Accuracy')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'model_evaluation.png'), dpi=150, bbox_inches='tight')
    print(f"[INFO] Model evaluation plot saved to {os.path.join(outdir, 'model_evaluation.png')}")
    plt.close()
