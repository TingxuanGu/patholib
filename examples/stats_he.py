#!/usr/bin/env python3
"""
Statistical analysis for H&E inflammation batch results — with outlier filtering.

Two-layer filtering:
  1. Quality filter: remove samples with total_nuclei < 5000 (insufficient tissue)
  2. IQR outlier detection: per-group 1.5×IQR on inflammatory_density

Reads JSON reports and performs:
- Descriptive statistics per group
- Normality testing (Shapiro-Wilk)
- Group comparisons (ANOVA/Kruskal-Wallis + post-hoc)
- Visualization (4 plots: density boxplot, score boxplot, density bar, cell type distribution)
- Excel summary export with 7 sheets including Excluded_Samples
"""

import os
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kruskal, f_oneway
from scikit_posthocs import posthoc_dunn
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = "/home/bio/桌面/Tingxuan Gu/analysis/WYJ HE-IHC results/WYJ HE/results"
OUTPUT_EXCEL = os.path.join(RESULTS_DIR, "WYJ_HE_Inflammation_Statistics.xlsx")
OUTPUT_FIGS_DIR = os.path.join(RESULTS_DIR, "figures")

GROUP_PATTERNS = {
    'con': r'^con-\d+',
    '4NQO': r'^4NQO-\d+$',
    '4NQO+Low-Se': r'^4NQO\+Low-Se-\d+$',
    '4NQO+Low-Se+L-MSC': r'^4NQO\+Low-Se\+L-MSC-\d+$',
    '4NQO+Low-Se+Se-Met': r'^4NQO\+Low-Se\+Se-Met-\d+$',
}

GROUP_ORDER = ['con', '4NQO', '4NQO+Low-Se', '4NQO+Low-Se+L-MSC', '4NQO+Low-Se+Se-Met']

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

MIN_NUCLEI = 5000  # Quality filter threshold (H&E has higher cell counts than IHC)
IQR_FACTOR = 1.5   # IQR multiplier for outlier detection


def assign_group(sample_name):
    for group, pattern in GROUP_PATTERNS.items():
        if re.match(pattern, sample_name):
            return group
    return 'Unknown'


def load_all_reports(results_dir):
    """Load all per-slide JSON reports."""
    data = []
    for json_file in sorted(Path(results_dir).glob("*_report.json")):
        sample_name = json_file.stem.replace('_report', '')
        group = assign_group(sample_name)
        with open(json_file, 'r') as f:
            r = json.load(f)
        data.append({
            'Sample': sample_name,
            'Group': group,
            'Inflammatory_density': r['inflammatory_density'],  # cells/mm²
            'Inflammation_score': r['inflammation_score'],      # 0-3
            'Total_nuclei': r['total_nuclei'],
            'Inflammatory_cells': r['inflammatory_cells'],
            'Parenchymal_cells': r['parenchymal_cells'],
            'Tissue_area_mm2': r['tissue_area_mm2'],
        })
    df = pd.DataFrame(data)
    df = df[df['Group'] != 'Unknown']
    return df


def filter_outliers(df):
    """Two-layer outlier filtering.

    Layer 1: Quality — remove samples with total_nuclei < MIN_NUCLEI.
    Layer 2: IQR — per-group 1.5×IQR on Inflammatory_density.

    Returns (filtered_df, excluded_df) where excluded_df records every
    removed sample with its exclusion reason.
    """
    excluded = []

    # --- Layer 1: Quality filter ---
    low_cell_mask = df['Total_nuclei'] < MIN_NUCLEI
    for _, row in df[low_cell_mask].iterrows():
        excluded.append({
            'Sample': row['Sample'],
            'Group': row['Group'],
            'Inflammatory_density': row['Inflammatory_density'],
            'Total_nuclei': row['Total_nuclei'],
            'Reason': f'Quality filter: total_nuclei ({int(row["Total_nuclei"])}) < {MIN_NUCLEI}',
            'Filter_layer': 1,
        })
    df_qual = df[~low_cell_mask].copy()

    # --- Layer 2: IQR outlier detection (per group) ---
    iqr_outlier_mask = pd.Series(False, index=df_qual.index)
    for group in GROUP_ORDER:
        gd = df_qual[df_qual['Group'] == group]['Inflammatory_density']
        if len(gd) < 3:
            continue
        q1 = gd.quantile(0.25)
        q3 = gd.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - IQR_FACTOR * iqr
        upper = q3 + IQR_FACTOR * iqr
        outliers = (gd < lower) | (gd > upper)
        for idx in gd[outliers].index:
            row = df_qual.loc[idx]
            excluded.append({
                'Sample': row['Sample'],
                'Group': row['Group'],
                'Inflammatory_density': row['Inflammatory_density'],
                'Total_nuclei': row['Total_nuclei'],
                'Reason': f'IQR outlier: Inflammatory_density {row["Inflammatory_density"]:.1f} outside [{lower:.1f}, {upper:.1f}]',
                'Filter_layer': 2,
            })
        iqr_outlier_mask |= outliers.reindex(iqr_outlier_mask.index, fill_value=False)

    df_filtered = df_qual[~iqr_outlier_mask].copy()
    excluded_df = pd.DataFrame(excluded) if excluded else pd.DataFrame(
        columns=['Sample', 'Group', 'Inflammatory_density', 'Total_nuclei', 'Reason', 'Filter_layer'])

    return df_filtered, excluded_df


def compute_group_stats(df):
    """Compute descriptive statistics per group."""
    rows = []
    for group in GROUP_ORDER:
        gd = df[df['Group'] == group]
        if len(gd) == 0:
            continue
        n = len(gd)
        total_nuclei = gd['Total_nuclei'].sum()
        inflammatory_cells = gd['Inflammatory_cells'].sum()
        parenchymal_cells = gd['Parenchymal_cells'].sum()
        
        rows.append({
            'Group': group,
            'N': n,
            'Inflammatory_density_mean': round(gd['Inflammatory_density'].mean(), 2),
            'Inflammatory_density_sd': round(gd['Inflammatory_density'].std(), 2),
            'Inflammatory_density_CV': round((gd['Inflammatory_density'].std() / gd['Inflammatory_density'].mean() * 100), 1) if gd['Inflammatory_density'].mean() > 0 else 0,
            'Inflammation_score_mean': round(gd['Inflammation_score'].mean(), 2),
            'Inflammation_score_sd': round(gd['Inflammation_score'].std(), 2),
            'Total_nuclei_mean': round(gd['Total_nuclei'].mean(), 0),
            'Inflammatory_pct': round((inflammatory_cells / total_nuclei) * 100, 2),
            'Parenchymal_pct': round((parenchymal_cells / total_nuclei) * 100, 2),
        })
    return pd.DataFrame(rows)


def test_normality(df):
    """Test normality using Shapiro-Wilk test."""
    results = []
    for group in GROUP_ORDER:
        gd = df[df['Group'] == group]
        if len(gd) < 3:
            continue
        d_stat, d_p = shapiro(gd['Inflammatory_density'])
        s_stat, s_p = shapiro(gd['Inflammation_score'])
        results.append({
            'Group': group,
            'N': len(gd),
            'Density_W': round(d_stat, 4),
            'Density_p': round(d_p, 4),
            'Density_normal': 'Yes' if d_p > 0.05 else 'No',
            'Score_W': round(s_stat, 4),
            'Score_p': round(s_p, 4),
            'Score_normal': 'Yes' if s_p > 0.05 else 'No',
        })
    return pd.DataFrame(results)


def compare_groups(df, metric='Inflammatory_density'):
    """Compare groups using ANOVA or Kruskal-Wallis."""
    groups = [df[df['Group'] == g][metric].values
              for g in GROUP_ORDER if g in df['Group'].values]

    all_normal = all(shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3)

    if all_normal:
        stat, p_value = f_oneway(*groups)
        test_name = 'One-way ANOVA'
    else:
        stat, p_value = kruskal(*groups)
        test_name = 'Kruskal-Wallis'

    return {
        'Metric': metric,
        'Test': test_name,
        'Statistic': round(stat, 4),
        'P_value': round(p_value, 6),
        'Significant': 'Yes' if p_value < 0.05 else 'No',
    }


def posthoc_analysis(df, metric='Inflammatory_density'):
    """Perform post-hoc pairwise comparisons using Dunn's test."""
    posthoc_df = posthoc_dunn(df, val_col=metric, group_col='Group', p_adjust='bonferroni')
    return posthoc_df, "Dunn's test (Bonferroni)"


def plot_inflammatory_density_boxplot(df, output_dir):
    """Plot inflammatory density boxplot with stripplot overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Group', y='Inflammatory_density', order=GROUP_ORDER, palette='Set2', ax=ax)
    sns.stripplot(data=df, x='Group', y='Inflammatory_density', order=GROUP_ORDER,
                  color='black', alpha=0.6, size=5, ax=ax)
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Inflammatory Cell Density (cells/mm²)', fontsize=12)
    ax.set_title('Inflammatory Cell Density by Group', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'inflammatory_density_boxplot.png'), dpi=300)
    plt.close(fig)


def plot_inflammation_score_boxplot(df, output_dir):
    """Plot inflammation score boxplot with stripplot overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Group', y='Inflammation_score', order=GROUP_ORDER, palette='Set2', ax=ax)
    sns.stripplot(data=df, x='Group', y='Inflammation_score', order=GROUP_ORDER,
                  color='black', alpha=0.6, size=5, ax=ax)
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Inflammation Score (0-3)', fontsize=12)
    ax.set_title('Inflammation Score by Group', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'inflammation_score_boxplot.png'), dpi=300)
    plt.close(fig)


def plot_inflammatory_density_barplot(df, output_dir):
    """Plot mean inflammatory density bar chart with error bars."""
    stats_agg = df.groupby('Group')['Inflammatory_density'].agg(['mean', 'std']).reindex(GROUP_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(stats_agg))
    ax.bar(x_pos, stats_agg['mean'], yerr=stats_agg['std'],
           capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Mean Inflammatory Cell Density (cells/mm²)', fontsize=12)
    ax.set_title('Mean Inflammatory Cell Density by Group', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_agg.index, rotation=15, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'inflammatory_density_barplot.png'), dpi=300)
    plt.close(fig)


def plot_cell_type_distribution(df, output_dir):
    """Plot 100% stacked bar chart of cell type distribution."""
    cell_type_pcts = []
    for group in GROUP_ORDER:
        gd = df[df['Group'] == group]
        if len(gd) == 0:
            continue
        total = gd['Total_nuclei'].sum()
        cell_type_pcts.append({
            'Group': group,
            'Inflammatory': (gd['Inflammatory_cells'].sum() / total) * 100,
            'Parenchymal': (gd['Parenchymal_cells'].sum() / total) * 100,
        })
    cell_df = pd.DataFrame(cell_type_pcts).set_index('Group')

    fig, ax = plt.subplots(figsize=(10, 6))
    cell_df.plot(kind='bar', stacked=True,
                 color=['#e65c5c', '#5c8ae6'],  # red for inflammatory, blue for parenchymal
                 edgecolor='black', linewidth=0.5, ax=ax)
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Cell Type Distribution by Group', fontsize=14, fontweight='bold')
    ax.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'cell_type_distribution.png'), dpi=300)
    plt.close(fig)


def main():
    print("=" * 60)
    print("H&E Inflammation Statistical Analysis")
    print("=" * 60)

    os.makedirs(OUTPUT_FIGS_DIR, exist_ok=True)

    # 1. Load
    print("\n[1/8] Loading JSON reports...")
    df_raw = load_all_reports(RESULTS_DIR)
    print(f"  Loaded {len(df_raw)} samples across {df_raw['Group'].nunique()} groups")
    for g in GROUP_ORDER:
        n = len(df_raw[df_raw['Group'] == g])
        print(f"    {g}: n={n}")

    # 2. Filter outliers
    print("\n[2/8] Filtering outliers...")
    print(f"  Layer 1: Quality filter (total_nuclei < {MIN_NUCLEI})")
    print(f"  Layer 2: IQR outlier detection ({IQR_FACTOR}xIQR per group)")
    df, excluded_df = filter_outliers(df_raw)

    if len(excluded_df) > 0:
        print(f"\n  Excluded {len(excluded_df)} sample(s):")
        for _, row in excluded_df.iterrows():
            print(f"    - {row['Sample']} ({row['Group']}): {row['Reason']}")
    else:
        print("  No samples excluded.")

    print(f"\n  Remaining: {len(df)} samples")
    for g in GROUP_ORDER:
        n = len(df[df['Group'] == g])
        print(f"    {g}: n={n}")

    # Check minimum group size
    for g in GROUP_ORDER:
        n = len(df[df['Group'] == g])
        if n < 3:
            print(f"  WARNING: {g} has fewer than 3 samples after filtering!")

    # 3. Descriptive stats
    print("\n[3/8] Descriptive statistics...")
    stats_df = compute_group_stats(df)
    print(stats_df.to_string(index=False))

    # 4. Normality
    print("\n[4/8] Normality testing (Shapiro-Wilk)...")
    normality_df = test_normality(df)
    print(normality_df.to_string(index=False))

    # 5. Group comparisons
    print("\n[5/8] Group comparisons...")
    density_test = compare_groups(df, 'Inflammatory_density')
    score_test = compare_groups(df, 'Inflammation_score')
    print(f"  Inflammatory_density: {density_test['Test']}, stat={density_test['Statistic']}, p={density_test['P_value']}")
    print(f"  Inflammation_score: {score_test['Test']}, stat={score_test['Statistic']}, p={score_test['P_value']}")

    # 6. Post-hoc
    print("\n[6/8] Post-hoc pairwise comparisons...")
    density_posthoc, dn = posthoc_analysis(df, 'Inflammatory_density')
    score_posthoc, sn = posthoc_analysis(df, 'Inflammation_score')
    print(f"  Inflammatory_density: {dn}")
    print(density_posthoc.to_string())
    print(f"\n  Inflammation_score: {sn}")
    print(score_posthoc.to_string())

    # 7. Figures
    print("\n[7/8] Generating visualizations...")
    plot_inflammatory_density_boxplot(df, OUTPUT_FIGS_DIR)
    plot_inflammation_score_boxplot(df, OUTPUT_FIGS_DIR)
    plot_inflammatory_density_barplot(df, OUTPUT_FIGS_DIR)
    plot_cell_type_distribution(df, OUTPUT_FIGS_DIR)
    print(f"  Saved 4 figures to {OUTPUT_FIGS_DIR}")

    # 8. Excel
    print("\n[8/8] Exporting to Excel...")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        # Sheet 1: Group Summary
        stats_df.to_excel(writer, sheet_name='Group_Summary', index=False)

        # Sheet 2: Per-sample raw data (filtered)
        df_export = df[['Sample', 'Group', 'Inflammatory_density', 'Inflammation_score',
                        'Total_nuclei', 'Inflammatory_cells', 'Parenchymal_cells',
                        'Tissue_area_mm2']].sort_values(['Group', 'Sample'])
        df_export.to_excel(writer, sheet_name='Raw_Data', index=False)

        # Sheet 3: Cell Type Distribution
        cell_rows = []
        for group in GROUP_ORDER:
            gd = df[df['Group'] == group]
            if len(gd) == 0:
                continue
            total = gd['Total_nuclei'].sum()
            inflammatory = gd['Inflammatory_cells'].sum()
            parenchymal = gd['Parenchymal_cells'].sum()
            cell_rows.append({
                'Group': group,
                'N': len(gd),
                'Total_nuclei': int(total),
                'Inflammatory_count': int(inflammatory),
                'Parenchymal_count': int(parenchymal),
                'Inflammatory_pct': round((inflammatory / total) * 100, 2),
                'Parenchymal_pct': round((parenchymal / total) * 100, 2),
            })
        pd.DataFrame(cell_rows).to_excel(writer, sheet_name='Cell_Type_Distribution', index=False)

        # Sheet 4: Statistical Tests
        normality_df.to_excel(writer, sheet_name='Statistical_Tests', index=False, startrow=0)
        test_rows = [density_test, score_test]
        pd.DataFrame(test_rows).to_excel(writer, sheet_name='Statistical_Tests',
                                         index=False, startrow=len(normality_df) + 3)

        # Sheet 5: Post-hoc Density
        density_posthoc.to_excel(writer, sheet_name='PostHoc_Density')

        # Sheet 6: Post-hoc Score
        score_posthoc.to_excel(writer, sheet_name='PostHoc_Score')

        # Sheet 7: Excluded samples
        excluded_df.to_excel(writer, sheet_name='Excluded_Samples', index=False)

    print(f"  Saved: {OUTPUT_EXCEL}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"  Original samples: {len(df_raw)}")
    print(f"  Excluded: {len(excluded_df)}")
    print(f"  Remaining: {len(df)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
