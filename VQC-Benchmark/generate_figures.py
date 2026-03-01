"""
VQC Benchmark: 논문용 시각화 생성 스크립트
Publication-quality figures with error bars for all 5 experiments
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figures", exist_ok=True)

# 스타일 설정
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

COLORS_VQC = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#17becf']
COLORS_CLASSICAL = ['#9467bd', '#8c564b', '#e377c2']


def _get_std(df, col):
    """_std 컬럼이 있으면 반환, 없으면 0."""
    std_col = f'{col}_std'
    if std_col in df.columns:
        return df[std_col].values
    return np.zeros(len(df))


# ── Figure 1: Experiment 1 - Model Comparison (Grouped Bar Chart with Error Bars) ──
def fig1_model_comparison():
    df = pd.read_csv("results/experiment1_model_comparison.csv")

    datasets = ["Iris", "Wine", "Breast Cancer"]
    models = df['model'].unique()

    vqc_models = [m for m in models if 'VQC' in m]
    classical_models = [m for m in models if 'VQC' not in m]
    ordered_models = list(vqc_models) + list(classical_models)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df['dataset'] == dataset]

        accuracies = []
        errors = []
        colors = []
        labels = []
        for m in ordered_models:
            row = subset[subset['model'] == m]
            if len(row) > 0:
                accuracies.append(row['test_accuracy'].values[0])
                std_col = 'test_accuracy_std'
                errors.append(row[std_col].values[0] if std_col in row.columns else 0)
                labels.append(m)
                if 'VQC' in m:
                    colors.append(COLORS_VQC[len([l for l in labels[:-1] if 'VQC' in l])])
                else:
                    colors.append(COLORS_CLASSICAL[len([l for l in labels[:-1] if 'VQC' not in l])])

        bars = ax.bar(range(len(labels)), accuracies, yerr=errors, capsize=3,
                      color=colors, edgecolor='black', linewidth=0.5,
                      error_kw={'linewidth': 1})
        ax.set_title(dataset, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

        for bar, acc, err in zip(bars, accuracies, errors):
            y_pos = bar.get_height() + err + 0.02
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{acc:.2f}', ha='center', va='bottom', fontsize=7)

    axes[0].set_ylabel('Test Accuracy')
    fig.suptitle('Figure 1: Model Comparison Across Datasets', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig1_model_comparison.png")
    plt.savefig("figures/fig1_model_comparison.pdf")
    plt.close()
    print("  Saved fig1_model_comparison")


# ── Figure 2: Experiment 1 - Training Time Comparison (Log Scale) ──
def fig2_training_time():
    df = pd.read_csv("results/experiment1_model_comparison.csv")

    datasets = ["Iris", "Wine", "Breast Cancer"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df['dataset'] == dataset].copy()

        vqc_mask = subset['model'].str.contains('VQC')
        vqc_data = subset[vqc_mask]
        classical_data = subset[~vqc_mask]

        all_data = pd.concat([vqc_data, classical_data])
        colors = ['#1f77b4' if 'VQC' in row['model'] else '#2ca02c'
                  for _, row in all_data.iterrows()]

        time_err = _get_std(all_data, 'train_time')
        bars = ax.barh(range(len(all_data)), all_data['train_time'].values,
                       xerr=time_err, capsize=3,
                       color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(all_data)))
        ax.set_yticklabels(all_data['model'].values, fontsize=8)
        ax.set_xscale('log')
        ax.set_title(dataset, fontweight='bold')
        ax.set_xlabel('Training Time (s, log scale)')

    fig.suptitle('Figure 2: Training Time Comparison', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig2_training_time.png")
    plt.savefig("figures/fig2_training_time.pdf")
    plt.close()
    print("  Saved fig2_training_time")


# ── Figure 3: Experiment 2 - Data Size vs Performance (with Error Bands) ──
def fig3_data_size():
    df = pd.read_csv("results/experiment2_data_size.csv")
    datasets = df['dataset'].unique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#9467bd', '#8c564b', '#e377c2']

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ds_df = df[df['dataset'] == dataset]
        models = ds_df['model'].unique()

        for i, model in enumerate(models):
            subset = ds_df[ds_df['model'] == model]
            x = subset['fraction'].values * 100
            y = subset['test_accuracy'].values
            yerr = _get_std(subset, 'test_accuracy')

            color = colors[i % len(colors)]
            ax.plot(x, y, marker=markers[i % len(markers)], label=model,
                    color=color, linewidth=2, markersize=8)
            if yerr.any():
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

        ax.set_xlabel('Training Data Fraction (%)')
        ax.set_title(dataset, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_ylim(0.3, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([20, 40, 60, 80, 100])

    axes[0].set_ylabel('Test Accuracy')
    fig.suptitle('Figure 3: Effect of Training Data Size on Performance',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig3_data_size.png")
    plt.savefig("figures/fig3_data_size.pdf")
    plt.close()
    print("  Saved fig3_data_size")


# ── Figure 4: Experiment 3 - Optimizer Comparison (with Error Bars) ──
def fig4_optimizer():
    df = pd.read_csv("results/experiment3_optimizer.csv")
    datasets = df['dataset'].unique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ds_df = df[df['dataset'] == dataset]

        optimizers = ds_df['optimizer'].values
        test_acc = ds_df['test_accuracy'].values
        train_acc = ds_df['train_accuracy'].values
        test_err = _get_std(ds_df, 'test_accuracy')
        train_err = _get_std(ds_df, 'train_accuracy')

        x = np.arange(len(optimizers))
        width = 0.35

        bars1 = ax.bar(x - width/2, train_acc, width, yerr=train_err, capsize=4,
                       label='Train', color='#1f77b4', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, test_acc, width, yerr=test_err, capsize=4,
                       label='Test', color='#ff7f0e', edgecolor='black', linewidth=0.5)

        ax.set_title(dataset, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([o.upper() for o in optimizers])
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)

        for bar, val, err in zip(bars1, train_acc, train_err):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + err + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        for bar, val, err in zip(bars2, test_acc, test_err):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + err + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    axes[0].set_ylabel('Accuracy')
    fig.suptitle('Figure 4: Optimizer Comparison', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig4_optimizer.png")
    plt.savefig("figures/fig4_optimizer.pdf")
    plt.close()
    print("  Saved fig4_optimizer")


# ── Figure 5: Experiment 4 - Circuit Depth (with Error Bars) ──
def fig5_circuit_depth():
    df = pd.read_csv("results/experiment4_circuit_depth.csv")
    datasets = df['dataset'].unique()

    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ds_df = df[df['dataset'] == dataset]

        reps = ds_df['reps'].values
        test_acc = ds_df['test_accuracy'].values
        train_acc = ds_df['train_accuracy'].values
        test_err = _get_std(ds_df, 'test_accuracy')
        train_err = _get_std(ds_df, 'train_accuracy')

        ax.errorbar(reps, train_acc, yerr=train_err, fmt='o-', label='Train',
                     color='#1f77b4', linewidth=2, markersize=8, capsize=4)
        ax.errorbar(reps, test_acc, yerr=test_err, fmt='s-', label='Test',
                     color='#ff7f0e', linewidth=2, markersize=8, capsize=4)
        ax.set_xlabel('Circuit Depth (reps)')
        ax.set_title(dataset, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xticks(reps)
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Accuracy')
    fig.suptitle('Figure 5: Effect of Circuit Depth', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig5_circuit_depth.png")
    plt.savefig("figures/fig5_circuit_depth.pdf")
    plt.close()
    print("  Saved fig5_circuit_depth")


# ── Figure 6: Summary Heatmap ──
def fig6_summary_heatmap():
    df = pd.read_csv("results/experiment1_model_comparison.csv")

    pivot = df.pivot_table(values='test_accuracy', index='model', columns='dataset')
    vqc_models = [m for m in pivot.index if 'VQC' in m]
    classical_models = [m for m in pivot.index if 'VQC' not in m]
    order = vqc_models + classical_models
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                linewidths=1, linecolor='white', vmin=0.3, vmax=1.0,
                cbar_kws={'label': 'Test Accuracy'})

    ax.set_title('Figure 6: Test Accuracy Heatmap (All Models x Datasets)',
                 fontweight='bold', pad=15)
    ax.set_ylabel('')
    ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig("figures/fig6_heatmap.png")
    plt.savefig("figures/fig6_heatmap.pdf")
    plt.close()
    print("  Saved fig6_heatmap")


# ── Figure 7: Experiment 5 - Noise Impact (with Error Bars) ──
def fig7_noise_impact():
    try:
        df = pd.read_csv("results/experiment5_noise.csv")
    except FileNotFoundError:
        print("  Skipped fig7_noise_impact (no data)")
        return

    if len(df) < 2:
        print("  Skipped fig7_noise_impact (insufficient data)")
        return

    datasets = df['dataset'].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        ds_df = df[df['dataset'] == dataset]

        error_rates = ds_df['error_rate'].values
        test_acc = ds_df['test_accuracy'].values
        train_acc = ds_df['train_accuracy'].values
        test_err = _get_std(ds_df, 'test_accuracy')
        train_err = _get_std(ds_df, 'train_accuracy')

        ax.errorbar(error_rates, train_acc, yerr=train_err, fmt='o-',
                     label='Train', color='#1f77b4', linewidth=2, markersize=8, capsize=4)
        ax.errorbar(error_rates, test_acc, yerr=test_err, fmt='s-',
                     label='Test', color='#ff7f0e', linewidth=2, markersize=8, capsize=4)
        ax.set_xlabel('Depolarizing Error Rate')
        ax.set_title(dataset, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xscale('symlog', linthresh=0.0005)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Accuracy')
    fig.suptitle('Figure 7: Impact of Quantum Noise on VQC Performance',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig7_noise_impact.png")
    plt.savefig("figures/fig7_noise_impact.pdf")
    plt.close()
    print("  Saved fig7_noise_impact")


# ── Figure 8: Experiment 6 - Convergence Curves ──
def fig8_convergence():
    import json
    try:
        with open("results/experiment6_loss_curves.json", "r") as f:
            curves_data = json.load(f)
        df = pd.read_csv("results/experiment6_convergence.csv")
    except FileNotFoundError:
        print("  Skipped fig8_convergence (no data)")
        return

    datasets = df['dataset'].unique()
    colors = {'200': '#1f77b4', '500': '#ff7f0e', '1000': '#2ca02c'}
    labels_map = {'200': 'maxiter=200', '500': 'maxiter=500', '1000': 'maxiter=1000'}

    fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        # Loss convergence curves for this dataset
        for maxiter_str in ['200', '500', '1000']:
            key = f"{dataset}_{maxiter_str}"
            if key not in curves_data:
                continue
            seed_curves = curves_data[key]
            color = colors[maxiter_str]
            min_len = min(len(c) for c in seed_curves)
            aligned = np.array([c[:min_len] for c in seed_curves])
            mean_curve = np.mean(aligned, axis=0)
            std_curve = np.std(aligned, axis=0)
            iters = np.arange(1, min_len + 1)

            ax.plot(iters, mean_curve, color=color, linewidth=1.5,
                     label=labels_map[maxiter_str])
            ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve,
                             alpha=0.15, color=color)

        ax.set_xlabel('Iteration')
        ax.set_title(dataset, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Objective Function Value')
    fig.suptitle('Figure 8: Convergence Analysis',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig8_convergence.png")
    plt.savefig("figures/fig8_convergence.pdf")
    plt.close()
    print("  Saved fig8_convergence")


# ── Figure 9: Experiment 7 - QSVC Comparison ──
def fig9_qsvc_comparison():
    try:
        df_qsvc = pd.read_csv("results/experiment7_qsvc.csv")
        df_exp1 = pd.read_csv("results/experiment1_model_comparison.csv")
    except FileNotFoundError:
        print("  Skipped fig9_qsvc_comparison (no data)")
        return

    # Combine QSVC results with experiment 1 results
    df_combined = pd.concat([df_exp1, df_qsvc], ignore_index=True)

    datasets = ["Iris", "Wine", "Breast Cancer"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df_combined[df_combined['dataset'] == dataset]

        # Order: VQC models, QSVC models, Classical models
        models = subset['model'].values
        vqc_models = [m for m in models if m.startswith('VQC')]
        qsvc_models = [m for m in models if m.startswith('QSVC')]
        classical_models = [m for m in models if not m.startswith('VQC') and not m.startswith('QSVC')]
        ordered = list(vqc_models) + list(qsvc_models) + list(classical_models)

        accuracies = []
        errors = []
        colors = []
        for m in ordered:
            row = subset[subset['model'] == m]
            if len(row) > 0:
                accuracies.append(row['test_accuracy'].values[0])
                std_col = 'test_accuracy_std'
                errors.append(row[std_col].values[0] if std_col in row.columns else 0)
                if m.startswith('VQC'):
                    colors.append('#1f77b4')
                elif m.startswith('QSVC'):
                    colors.append('#d62728')
                else:
                    colors.append('#2ca02c')

        bars = ax.bar(range(len(ordered)), accuracies, yerr=errors, capsize=3,
                      color=colors, edgecolor='black', linewidth=0.5,
                      error_kw={'linewidth': 1})
        ax.set_title(dataset, fontweight='bold')
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(ordered, rotation=45, ha='right', fontsize=7)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

        for bar, acc, err in zip(bars, accuracies, errors):
            y_pos = bar.get_height() + err + 0.02
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{acc:.2f}', ha='center', va='bottom', fontsize=6)

    axes[0].set_ylabel('Test Accuracy')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='VQC'),
                       Patch(facecolor='#d62728', label='QSVC'),
                       Patch(facecolor='#2ca02c', label='Classical')]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.suptitle('Figure 9: VQC vs QSVC vs Classical Models',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig9_qsvc_comparison.png")
    plt.savefig("figures/fig9_qsvc_comparison.pdf")
    plt.close()
    print("  Saved fig9_qsvc_comparison")


if __name__ == "__main__":
    print("Generating publication-quality figures...")
    fig1_model_comparison()
    fig2_training_time()
    fig3_data_size()
    fig4_optimizer()
    fig5_circuit_depth()
    fig6_summary_heatmap()
    fig7_noise_impact()
    fig8_convergence()
    fig9_qsvc_comparison()
    print("\nAll figures saved to figures/ directory")
