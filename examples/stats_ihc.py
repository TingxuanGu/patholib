#!/usr/bin/env python3
"""
Statistical analysis for GPX4 IHC batch results.

Reads JSON reports and performs:
- Descriptive statistics per group
- Normality testing (Shapiro-Wilk)
- Group comparisons (ANOVA/Kruskal-Wallis + post-hoc)
- Visualization (boxplots, bar charts)
- Excel summary export (4+ sheets)
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

RESULTS_DIR = "/home/bio/桌面/Tingxuan Gu/analysis/WYJ HE-IHC results/WYJ IHC/results"
OUTPUT_EXCEL = os.path.join(RESULTS_DIR, "GPX4_IHC_Summary_Statistics.xlsx")
OUTPUT_FIGS_DIR = os.path.join(RESULTS_DIR, "figures")

GROUP_PATTERNS = {
    'CON': r'^con-\d+',
    '4NQO': r'^4NQO-\d+$',
    '4NQO+Low-Se': r'^4NQO\+Low-Se-\d+$',
    '4NQO+Low-Se+L-MSC': r'^4NQO\+Low-Se\+L-MSC-\d+$',
    '4NQO+Low-Se+Se-Met': r'^4NQO\+Low-Se\+Se-Met-\d+$',
}

GROUP_ORDER = ['CON', '4NQO', '4NQO+Low-Se', '4NQO+Low-Se+L-MSC', '4NQO+Low-Se+Se-Met']

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300


def assign_group(sample_name):
    for group, pattern in GROUP_PATTERNS.items():
        if re.match(pattern, sample_name):
            return group
    return 'Unknown'


def load_all_reports(results_dir):
    data = []
    for json_file in sorted(Path(results_dir).glob("*_report.json")):
        sample_name = json_file.stem.replace('_report', '')
        group = assign_group(sample_name)
        with open(json_file, 'r') as f:
            r = json.load(f)
        data.append({
            'Sample': sample_name,
            'Group': group,
            'H_score': r['h_score'],
            'Positive_pct': r['positive_percentage'],
            'Total_cells': r['total_cells'],
            'Positive_cells': r['positive_cells'],
            'Negative_cells': r['negative_cells'],
            'Negative': r['grade_counts']['0'],
            'Weak': r['grade_counts']['1'],
            'Moderate': r['grade_counts']['2'],
            'Strong': r['grade_counts']['3'],
            'Tissue_area_mm2': r.get('tissue_area_mm2', 0),
        })
    df = pd.DataFrame(data)
    df = df[df['Group'] != 'Unknown']
    return df


def compute_group_stats(df):
    rows = []
    for group in GROUP_ORDER:
        gd = df[df['Group'] == group]
        if len(gd) == 0:
            continue
        n = len(gd)
        total_cells = gd['Total_cells'].sum()
        rows.append({
            'Group': group,
            'N': n,
            'H_score_mean': round(gd['H_score'].mean(), 1),
            'H_score_sd': round(gd['H_score'].std(), 1),
            'Positive_pct_mean': round(gd['Positive_pct'].mean(), 1),
            'Positive_pct_sd': round(gd['Positive_pct'].std(), 1),
            'Total_cells_mean': round(gd['Total_cells'].mean(), 0),
            'Total_cells_sd': round(gd['Total_cells'].std(), 0),
            'Negative_pct': round((gd['Negative'].sum() / total_cells) * 100, 1),
            'Weak_pct': round((gd['Weak'].sum() / total_cells) * 100, 1),
            'Moderate_pct': round((gd['Moderate'].sum() / total_cells) * 100, 1),
            'Strong_pct': round((gd['Strong'].sum() / total_cells) * 100, 1),
        })
    return pd.DataFrame(rows)


def test_normality(df):
    results = []
    for group in GROUP_ORDER:
        gd = df[df['Group'] == group]
        if len(gd) < 3:
            continue
        h_stat, h_p = shapiro(gd['H_score'])
        p_stat, p_p = shapiro(gd['Positive_pct'])
        results.append({
            'Group': group,
            'H_score_W': round(h_stat, 4),
            'H_score_p': round(h_p, 4),
            'H_score_normal': 'Yes' if h_p > 0.05 else 'No',
            'Positive_pct_W': round(p_stat, 4),
            'Positive_pct_p': round(p_p, 4),
            'Positive_pct_normal': 'Yes' if p_p > 0.05 else 'No',
        })
    return pd.DataFrame(results)


def compare_groups(df, metric='H_score'):
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


def posthoc_analysis(df, metric='H_score'):
    posthoc_df = posthoc_dunn(df, val_col=metric, group_col='Group', p_adjust='bonferroni')
    return posthoc_df, "Dunn's test (Bonferroni)"


def plot_hscore_boxplot(df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Group', y='H_score', order=GROUP_ORDER, palette='Set2', ax=ax)
    sns.stripplot(data=df, x='Group', y='H_score', order=GROUP_ORDER,
                  color='black', alpha=0.6, size=5, ax=ax)
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('H-score', fontsize=12)
    ax.set_title('GPX4 H-score by Group', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'hscore_boxplot.png'), dpi=300)
    plt.close(fig)


def plot_positive_rate_barplot(df, output_dir):
    stats_agg = df.groupby('Group')['Positive_pct'].agg(['mean', 'std']).reindex(GROUP_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(stats_agg))
    ax.bar(x_pos, stats_agg['mean'], yerr=stats_agg['std'],
           capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Positive Rate (%)', fontsize=12)
    ax.set_title('GPX4 Positive Rate by Group', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_agg.index, rotation=15, ha='right')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'positive_rate_barplot.png'), dpi=300)
    plt.close(fig)


def plot_grade_distribution(df, output_dir):
    grade_pcts = []
    for group in GROUP_ORDER:
        gd = df[df['Group'] == group]
        if len(gd) == 0:
            continue
        total = gd['Total_cells'].sum()
        grade_pcts.append({
            'Group': group,
            'Negative': (gd['Negative'].sum() / total) * 100,
            'Weak (+)': (gd['Weak'].sum() / total) * 100,
            'Moderate (++)': (gd['Moderate'].sum() / total) * 100,
            'Strong (+++)': (gd['Strong'].sum() / total) * 100,
        })
    grade_df = pd.DataFrame(grade_pcts).set_index('Group')

    fig, ax = plt.subplots(figsize=(10, 6))
    grade_df.plot(kind='bar', stacked=True,
                  color=['#d3d3d3', '#a8d5a2', '#ffcc66', '#e65c5c'],
                  edgecolor='black', linewidth=0.5, ax=ax)
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('GPX4 Staining Grade Distribution', fontsize=14, fontweight='bold')
    ax.legend(title='Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'grade_distribution.png'), dpi=300)
    plt.close(fig)


def main():
    print("=" * 60)
    print("GPX4 IHC Statistical Analysis")
    print("=" * 60)

    os.makedirs(OUTPUT_FIGS_DIR, exist_ok=True)

    # 1. Load
    print("\n[1/7] Loading JSON reports...")
    df = load_all_reports(RESULTS_DIR)
    print(f"  Loaded {len(df)} samples across {df['Group'].nunique()} groups")
    for g in GROUP_ORDER:
        n = len(df[df['Group'] == g])
        print(f"    {g}: n={n}")

    # 2. Descriptive stats
    print("\n[2/7] Descriptive statistics...")
    stats_df = compute_group_stats(df)
    print(stats_df.to_string(index=False))

    # 3. Normality
    print("\n[3/7] Normality testing (Shapiro-Wilk)...")
    normality_df = test_normality(df)
    print(normality_df.to_string(index=False))

    # 4. Group comparisons
    print("\n[4/7] Group comparisons...")
    hscore_test = compare_groups(df, 'H_score')
    posrate_test = compare_groups(df, 'Positive_pct')
    print(f"  H-score: {hscore_test['Test']}, stat={hscore_test['Statistic']}, p={hscore_test['P_value']}")
    print(f"  Positive%: {posrate_test['Test']}, stat={posrate_test['Statistic']}, p={posrate_test['P_value']}")

    # 5. Post-hoc
    print("\n[5/7] Post-hoc pairwise comparisons...")
    hscore_posthoc, hn = posthoc_analysis(df, 'H_score')
    posrate_posthoc, pn = posthoc_analysis(df, 'Positive_pct')
    print(f"  H-score: {hn}")
    print(hscore_posthoc.to_string())
    print(f"\n  Positive%: {pn}")
    print(posrate_posthoc.to_string())

    # 6. Figures
    print("\n[6/7] Generating visualizations...")
    plot_hscore_boxplot(df, OUTPUT_FIGS_DIR)
    plot_positive_rate_barplot(df, OUTPUT_FIGS_DIR)
    plot_grade_distribution(df, OUTPUT_FIGS_DIR)
    print(f"  Saved 3 figures to {OUTPUT_FIGS_DIR}")

    # 7. Excel
    print("\n[7/7] Exporting to Excel...")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        # Sheet 1: Group Summary
        stats_df.to_excel(writer, sheet_name='Group_Summary', index=False)

        # Sheet 2: Per-sample raw data
        df_export = df[['Sample', 'Group', 'H_score', 'Positive_pct', 'Total_cells',
                        'Positive_cells', 'Negative_cells',
                        'Negative', 'Weak', 'Moderate', 'Strong',
                        'Tissue_area_mm2']].sort_values(['Group', 'Sample'])
        df_export.to_excel(writer, sheet_name='Raw_Data', index=False)

        # Sheet 3: Grade distribution
        grade_rows = []
        for group in GROUP_ORDER:
            gd = df[df['Group'] == group]
            if len(gd) == 0:
                continue
            total = gd['Total_cells'].sum()
            grade_rows.append({
                'Group': group,
                'N': len(gd),
                'Total_cells': total,
                'Negative_count': int(gd['Negative'].sum()),
                'Weak_count': int(gd['Weak'].sum()),
                'Moderate_count': int(gd['Moderate'].sum()),
                'Strong_count': int(gd['Strong'].sum()),
                'Negative_pct': round((gd['Negative'].sum() / total) * 100, 2),
                'Weak_pct': round((gd['Weak'].sum() / total) * 100, 2),
                'Moderate_pct': round((gd['Moderate'].sum() / total) * 100, 2),
                'Strong_pct': round((gd['Strong'].sum() / total) * 100, 2),
            })
        pd.DataFrame(grade_rows).to_excel(writer, sheet_name='Grade_Distribution', index=False)

        # Sheet 4: Statistical tests
        test_rows = [hscore_test, posrate_test]
        normality_df.to_excel(writer, sheet_name='Statistical_Tests', index=False, startrow=0)
        pd.DataFrame(test_rows).to_excel(writer, sheet_name='Statistical_Tests',
                                         index=False, startrow=len(normality_df) + 3)

        # Sheet 5: Post-hoc H-score
        hscore_posthoc.to_excel(writer, sheet_name='PostHoc_Hscore')

        # Sheet 6: Post-hoc Positive rate
        posrate_posthoc.to_excel(writer, sheet_name='PostHoc_PosRate')

    print(f"  Saved: {OUTPUT_EXCEL}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
