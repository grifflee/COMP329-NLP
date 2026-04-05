"""
=============================================================================
 COMPARISON SCRIPT — Side-by-Side Model Evaluation
=============================================================================

 Run this AFTER training all 5 models to compare their performance.
 It reads the saved metrics JSON files and generates:
   1. A comparison table
   2. A bar chart of test accuracies
   3. A summary of which model wins

 Usage:
   python compare_results.py
=============================================================================
"""

import os
import json
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_all_metrics():
    """Load metrics from all model JSON files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_files = sorted(glob.glob(os.path.join(script_dir, '*_metrics.json')))

    if not metrics_files:
        print("❌ No metrics files found! Train the models first.")
        print("   Run each model script: python 01_dense_feedforward.py, etc.")
        return []

    all_metrics = []
    for path in metrics_files:
        with open(path, 'r') as f:
            metrics = json.load(f)
            metrics['filename'] = os.path.basename(path)
            all_metrics.append(metrics)

    return all_metrics


def print_comparison_table(metrics_list):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print(" MODEL COMPARISON — IMDB Genre Classification (27 classes)")
    print("=" * 80)

    header = f"{'Model':<25} {'Test Acc':>10} {'Test Loss':>10} {'Train Acc':>10} {'Val Acc':>10} {'Epochs':>8}"
    print(header)
    print("-" * 80)

    for m in metrics_list:
        row = (
            f"{m['model']:<25} "
            f"{m['test_accuracy']:>10.4f} "
            f"{m['test_loss']:>10.4f} "
            f"{m.get('final_train_acc', 0):>10.4f} "
            f"{m.get('final_val_acc', 0):>10.4f} "
            f"{m.get('epochs_trained', m.get('epochs', '?')):>8}"
        )
        print(row)

    print("-" * 80)

    # Find best model
    best = max(metrics_list, key=lambda m: m['test_accuracy'])
    print(f"\n🏆 BEST MODEL: {best['model']} with {best['test_accuracy']:.4f} test accuracy")

    # Find most overfit model (largest gap between train and val accuracy)
    for m in metrics_list:
        gap = m.get('final_train_acc', 0) - m.get('final_val_acc', 0)
        if gap > 0.05:
            print(f"⚠️  {m['model']} shows overfitting (train-val gap: {gap:.4f})")


def plot_comparison(metrics_list):
    """Create comparison bar chart."""
    models = [m['model'] for m in metrics_list]
    test_accs = [m['test_accuracy'] for m in metrics_list]

    # Color scheme
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0']
    colors = colors[:len(models)]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(models, test_accs, color=colors, edgecolor='white', linewidth=2)

    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            bar.get_height() + 0.005,
            f'{acc:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12
        )

    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title('Neural Network Architecture Comparison\nIMDB Genre Classification (27 classes)',
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(test_accs) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    # Rotate labels for readability
    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'comparison_chart.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 Comparison chart saved to comparison_chart.png")
    plt.show()


def main():
    metrics_list = load_all_metrics()

    if not metrics_list:
        return

    print(f"\n✅ Found {len(metrics_list)} trained model(s)")

    print_comparison_table(metrics_list)
    plot_comparison(metrics_list)

    # Summary insights
    print("\n" + "=" * 80)
    print(" KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    📌 Dense Feedforward: Simplest model, bag-of-words baseline.
       Good starting point but ignores word order.

    📌 Simple RNN: Introduces word order but struggles with long descriptions
       due to vanishing gradients.

    📌 LSTM: Gates allow long-range memory. Bidirectional reading helps
       capture context from both ends of the description.

    📌 CNN: Detects local n-gram patterns. Often surprisingly competitive
       and much faster to train than RNNs.

    📌 Transformer: Self-attention lets every word attend to every other word.
       Most powerful but most complex. This is the architecture behind
       BERT, GPT, and most modern NLP systems.
    """)


if __name__ == '__main__':
    main()
