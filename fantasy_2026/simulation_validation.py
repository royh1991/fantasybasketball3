#!/usr/bin/env python3
"""
Simulation Validation Script

Fetches historical box scores from ESPN API for all completed weeks,
runs simulations for each week, and compares predicted vs actual results.

Outputs a comprehensive markdown report with visualizations showing:
- Prediction accuracy per week and overall
- Whether predictions fell within 1σ or 2σ of simulated distributions
- Category-level accuracy analysis
- Calibration analysis (do 70% predictions win 70% of the time?)

Usage:
    python simulation_validation.py
    python simulation_validation.py --weeks 1-7
    python simulation_validation.py --skip-fetch  # Use existing data
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / 'projection_diagnostics'))

# Constants
DATA_DIR = Path(__file__).parent / 'data'
MATCHUPS_DIR = DATA_DIR / 'matchups'
REPORTS_DIR = Path(__file__).parent / 'simulation_reports'
SCRIPTS_DIR = Path(__file__).parent / 'scripts'

# Categories used in fantasy basketball
CATEGORIES = ['FG%', 'FT%', '3P%', '3PM', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'DD']


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate simulation accuracy across historical weeks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--weeks',
        type=str,
        default='1-8',
        help='Week range to validate (e.g., "1-7" or "1,3,5,7")'
    )

    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip fetching from ESPN API, use existing data'
    )

    parser.add_argument(
        '--skip-simulations',
        action='store_true',
        help='Skip running simulations, use existing simulation results'
    )

    return parser.parse_args()


def parse_week_range(week_str: str) -> List[int]:
    """Parse week range string into list of weeks."""
    weeks = []
    for part in week_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            weeks.extend(range(int(start), int(end) + 1))
        else:
            weeks.append(int(part))
    return sorted(set(weeks))


def fetch_week_data(week: int) -> bool:
    """Fetch box scores for a specific week from ESPN API."""
    print(f"  📥 Fetching Week {week} from ESPN API...")

    script_path = SCRIPTS_DIR / '3_get_matchups.py'

    try:
        result = subprocess.run(
            ['python', str(script_path), '--week', str(week), '--no-latest'],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print(f"     ✅ Week {week} fetched successfully")
            return True
        else:
            print(f"     ❌ Failed to fetch Week {week}")
            if result.stderr:
                print(f"        {result.stderr[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"     ❌ Timeout fetching Week {week}")
        return False
    except Exception as e:
        print(f"     ❌ Error: {e}")
        return False


def find_box_scores_file(week: int) -> Optional[Path]:
    """Find the box scores file for a specific week."""
    # Look for week-specific files first
    for f in sorted(MATCHUPS_DIR.glob(f'box_scores_week{week}_*.csv'), reverse=True):
        return f

    # Check if box_scores_latest.csv has this week
    latest = MATCHUPS_DIR / 'box_scores_latest.csv'
    if latest.exists():
        df = pd.read_csv(latest)
        if week in df['week'].values:
            return latest

    # Check other archived files
    for f in sorted(MATCHUPS_DIR.glob('box_scores_*.csv'), reverse=True):
        if f.name == 'box_scores_latest.csv':
            continue
        try:
            df = pd.read_csv(f)
            if week in df['week'].values:
                return f
        except:
            continue

    return None


def run_simulation_for_week(week: int) -> Optional[Path]:
    """Run simulation for a specific week and return results directory."""
    print(f"  🎲 Running simulation for Week {week}...")

    script_path = Path(__file__).parent / 'simulate_with_correct_data.py'

    try:
        result = subprocess.run(
            ['python', str(script_path), '--week', str(week)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).parent)
        )

        if result.returncode == 0:
            # Find the generated report directory
            report_dirs = sorted(REPORTS_DIR.glob(f'week{week}_report_*'), reverse=True)
            if report_dirs:
                print(f"     ✅ Simulation complete: {report_dirs[0].name}")
                return report_dirs[0]
            else:
                print(f"     ⚠️  Simulation ran but no report found")
                return None
        else:
            print(f"     ❌ Simulation failed for Week {week}")
            if result.stderr:
                print(f"        {result.stderr[:300]}")
            return None

    except subprocess.TimeoutExpired:
        print(f"     ❌ Timeout running simulation for Week {week}")
        return None
    except Exception as e:
        print(f"     ❌ Error: {e}")
        return None


def parse_box_score_stats(stat_str: str) -> Dict:
    """Parse the stat_0 column which contains a dictionary."""
    import ast
    try:
        stat_dict = ast.literal_eval(stat_str)
        return stat_dict.get('total', {})
    except:
        return {}


def calculate_actual_results(box_scores_df: pd.DataFrame, week: int) -> Dict:
    """Calculate actual matchup results from box scores."""
    week_data = box_scores_df[box_scores_df['week'] == week].copy()

    if len(week_data) == 0:
        return {}

    # Parse stats
    week_data['parsed_stats'] = week_data['stat_0'].apply(parse_box_score_stats)

    results = {}

    for matchup_name in week_data['matchup'].unique():
        matchup_data = week_data[week_data['matchup'] == matchup_name]

        home_data = matchup_data[matchup_data['team_side'] == 'home']
        away_data = matchup_data[matchup_data['team_side'] == 'away']

        if len(home_data) == 0 or len(away_data) == 0:
            continue

        home_team = home_data['team_name'].iloc[0]
        away_team = away_data['team_name'].iloc[0]

        # Aggregate stats for each team
        home_totals = aggregate_team_stats(home_data)
        away_totals = aggregate_team_stats(away_data)

        # Calculate category winners
        category_results = calculate_category_winners(home_totals, away_totals)

        home_cats = sum(1 for v in category_results.values() if v == 'home')
        away_cats = sum(1 for v in category_results.values() if v == 'away')
        ties = sum(1 for v in category_results.values() if v == 'tie')

        winner = 'home' if home_cats > away_cats else ('away' if away_cats > home_cats else 'tie')

        results[matchup_name] = {
            'home_team': home_team,
            'away_team': away_team,
            'home_cats': home_cats,
            'away_cats': away_cats,
            'ties': ties,
            'winner': winner,
            'winner_name': home_team if winner == 'home' else (away_team if winner == 'away' else 'TIE'),
            'category_results': category_results,
            'home_totals': home_totals,
            'away_totals': away_totals
        }

    return results


def aggregate_team_stats(team_data: pd.DataFrame) -> Dict:
    """Aggregate stats for a team from player-level data."""
    totals = {
        'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0,
        'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'DD': 0
    }

    for _, row in team_data.iterrows():
        stats = row['parsed_stats']
        if not stats:
            continue

        totals['FGM'] += stats.get('FGM', 0)
        totals['FGA'] += stats.get('FGA', 0)
        totals['FTM'] += stats.get('FTM', 0)
        totals['FTA'] += stats.get('FTA', 0)
        totals['FG3M'] += stats.get('3PM', 0)
        totals['FG3A'] += stats.get('3PA', 0)
        totals['PTS'] += stats.get('PTS', 0)
        totals['REB'] += stats.get('REB', 0)
        totals['AST'] += stats.get('AST', 0)
        totals['STL'] += stats.get('STL', 0)
        totals['BLK'] += stats.get('BLK', 0)
        totals['TOV'] += stats.get('TO', 0)
        totals['DD'] += stats.get('DD', 0)

    # Calculate percentages
    totals['FG_PCT'] = totals['FGM'] / totals['FGA'] if totals['FGA'] > 0 else 0
    totals['FT_PCT'] = totals['FTM'] / totals['FTA'] if totals['FTA'] > 0 else 0
    totals['FG3_PCT'] = totals['FG3M'] / totals['FG3A'] if totals['FG3A'] > 0 else 0

    return totals


def calculate_category_winners(home: Dict, away: Dict) -> Dict[str, str]:
    """Determine category winners."""
    results = {}

    # Percentage categories
    for cat, key in [('FG%', 'FG_PCT'), ('FT%', 'FT_PCT'), ('3P%', 'FG3_PCT')]:
        if home[key] > away[key]:
            results[cat] = 'home'
        elif away[key] > home[key]:
            results[cat] = 'away'
        else:
            results[cat] = 'tie'

    # Counting categories (higher is better)
    for cat, key in [('3PM', 'FG3M'), ('PTS', 'PTS'), ('REB', 'REB'),
                     ('AST', 'AST'), ('STL', 'STL'), ('BLK', 'BLK'), ('DD', 'DD')]:
        if home[key] > away[key]:
            results[cat] = 'home'
        elif away[key] > home[key]:
            results[cat] = 'away'
        else:
            results[cat] = 'tie'

    # Turnovers (lower is better)
    if home['TOV'] < away['TOV']:
        results['TO'] = 'home'
    elif away['TOV'] < home['TOV']:
        results['TO'] = 'away'
    else:
        results['TO'] = 'tie'

    return results


def calculate_category_win_pcts_from_sims(sims_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate home team win percentage for each category from simulation data."""
    if sims_df is None or len(sims_df) == 0:
        return {}

    category_win_pcts = {}
    n_sims = len(sims_df)

    # FG% - calculate from FGM/FGA
    home_fg_pct = sims_df['team_a_FGM'] / sims_df['team_a_FGA']
    away_fg_pct = sims_df['team_b_FGM'] / sims_df['team_b_FGA']
    category_win_pcts['FG%'] = (home_fg_pct > away_fg_pct).sum() / n_sims

    # FT% - calculate from FTM/FTA
    home_ft_pct = sims_df['team_a_FTM'] / sims_df['team_a_FTA']
    away_ft_pct = sims_df['team_b_FTM'] / sims_df['team_b_FTA']
    category_win_pcts['FT%'] = (home_ft_pct > away_ft_pct).sum() / n_sims

    # 3P% - calculate from FG3M/FG3A
    home_3p_pct = sims_df['team_a_FG3M'] / sims_df['team_a_FG3A']
    away_3p_pct = sims_df['team_b_FG3M'] / sims_df['team_b_FG3A']
    category_win_pcts['3P%'] = (home_3p_pct > away_3p_pct).sum() / n_sims

    # Counting stats (higher is better)
    for cat, col in [('3PM', 'FG3M'), ('PTS', 'PTS'), ('REB', 'REB'),
                     ('AST', 'AST'), ('STL', 'STL'), ('BLK', 'BLK'), ('DD', 'DD')]:
        home_col = f'team_a_{col}'
        away_col = f'team_b_{col}'
        category_win_pcts[cat] = (sims_df[home_col] > sims_df[away_col]).sum() / n_sims

    # Turnovers (lower is better)
    category_win_pcts['TO'] = (sims_df['team_a_TOV'] < sims_df['team_b_TOV']).sum() / n_sims

    return category_win_pcts


def load_simulation_results(report_dir: Path) -> Dict:
    """Load simulation results from a report directory."""
    results = {}

    # Load summary
    sim_dir = Path(__file__).parent / 'fixed_simulations'
    summary_file = sim_dir / 'all_matchups_summary.csv'

    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)

        for _, row in summary_df.iterrows():
            matchup_name = row['matchup']

            # Load detailed simulations
            safe_name = matchup_name.replace(' ', '_')
            sims_file = sim_dir / safe_name / 'all_simulations.csv'

            sims_df = None
            category_win_pcts = {}
            if sims_file.exists():
                sims_df = pd.read_csv(sims_file)
                category_win_pcts = calculate_category_win_pcts_from_sims(sims_df)

            results[matchup_name] = {
                'home_team': row['team_a_name'],
                'away_team': row['team_b_name'],
                'home_win_pct': row['team_a_win_pct'],
                'away_win_pct': row['team_b_win_pct'],
                'home_avg_cats': row['team_a_avg_cats_won'],
                'away_avg_cats': row['team_b_avg_cats_won'],
                'predicted_winner': 'home' if row['team_a_win_pct'] > row['team_b_win_pct'] else 'away',
                'confidence': max(row['team_a_win_pct'], row['team_b_win_pct']),
                'simulations': sims_df,
                'category_win_pcts': category_win_pcts
            }

    return results


def compare_results(actual: Dict, simulated: Dict) -> Dict:
    """Compare actual vs simulated results for a matchup."""
    comparison = {
        'actual_winner': actual['winner'],
        'actual_winner_name': actual['winner_name'],
        'actual_home_cats': actual['home_cats'],
        'actual_away_cats': actual['away_cats'],
        'predicted_winner': simulated['predicted_winner'],
        'predicted_winner_name': simulated['home_team'] if simulated['predicted_winner'] == 'home' else simulated['away_team'],
        'home_win_pct': simulated['home_win_pct'],
        'away_win_pct': simulated['away_win_pct'],
        'confidence': simulated['confidence'],
        'correct': actual['winner'] == simulated['predicted_winner'],
    }

    # Check if actual result within distribution
    if simulated['simulations'] is not None:
        sims = simulated['simulations']

        # Category distribution analysis
        home_cats_mean = sims['team_a_cats'].mean()
        home_cats_std = sims['team_a_cats'].std()
        away_cats_mean = sims['team_b_cats'].mean()
        away_cats_std = sims['team_b_cats'].std()

        comparison['home_cats_mean'] = home_cats_mean
        comparison['home_cats_std'] = home_cats_std
        comparison['away_cats_mean'] = away_cats_mean
        comparison['away_cats_std'] = away_cats_std

        # Z-scores
        if home_cats_std > 0:
            comparison['home_cats_zscore'] = (actual['home_cats'] - home_cats_mean) / home_cats_std
        else:
            comparison['home_cats_zscore'] = 0

        if away_cats_std > 0:
            comparison['away_cats_zscore'] = (actual['away_cats'] - away_cats_mean) / away_cats_std
        else:
            comparison['away_cats_zscore'] = 0

        # Within 1σ or 2σ?
        comparison['home_within_1std'] = abs(comparison['home_cats_zscore']) <= 1
        comparison['home_within_2std'] = abs(comparison['home_cats_zscore']) <= 2
        comparison['away_within_1std'] = abs(comparison['away_cats_zscore']) <= 1
        comparison['away_within_2std'] = abs(comparison['away_cats_zscore']) <= 2

    # Category-level analysis
    category_comparisons = []
    cat_win_pcts = simulated.get('category_win_pcts', {})
    actual_cat_results = actual.get('category_results', {})

    for cat in CATEGORIES:
        if cat in cat_win_pcts and cat in actual_cat_results:
            pred_home_win_pct = cat_win_pcts[cat]
            actual_winner = actual_cat_results[cat]  # 'home', 'away', or 'tie'

            # Predicted winner for this category
            pred_winner = 'home' if pred_home_win_pct > 0.5 else 'away'
            pred_confidence = max(pred_home_win_pct, 1 - pred_home_win_pct)

            # Was prediction correct?
            if actual_winner == 'tie':
                cat_correct = None  # Skip ties
            else:
                cat_correct = (pred_winner == actual_winner)

            category_comparisons.append({
                'category': cat,
                'pred_home_win_pct': pred_home_win_pct,
                'pred_winner': pred_winner,
                'pred_confidence': pred_confidence,
                'actual_winner': actual_winner,
                'correct': cat_correct
            })

    comparison['category_comparisons'] = category_comparisons

    return comparison


def create_validation_visualizations(all_results: Dict, output_dir: Path):
    """Create comprehensive validation visualizations."""

    # Collect all comparisons
    all_comparisons = []
    for week, week_results in all_results.items():
        for matchup, comp in week_results['comparisons'].items():
            comp['week'] = week
            comp['matchup'] = matchup
            all_comparisons.append(comp)

    if not all_comparisons:
        print("  ⚠️  No comparisons to visualize")
        return

    df = pd.DataFrame(all_comparisons)

    # Figure 1: Overall accuracy dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Accuracy by week
    ax = axes[0, 0]
    week_accuracy = df.groupby('week')['correct'].mean() * 100
    colors = ['#2ecc71' if acc >= 70 else '#f39c12' if acc >= 50 else '#e74c3c'
              for acc in week_accuracy.values]
    bars = ax.bar(week_accuracy.index, week_accuracy.values, color=colors, edgecolor='black')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    ax.axhline(y=week_accuracy.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Average ({week_accuracy.mean():.1f}%)')
    ax.set_xlabel('Week', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Prediction Accuracy by Week', fontweight='bold', fontsize=12)
    ax.legend()
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, week_accuracy.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', fontweight='bold')

    # 2. Overall accuracy pie
    ax = axes[0, 1]
    correct = df['correct'].sum()
    incorrect = len(df) - correct
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax.pie([correct, incorrect],
                                       labels=['Correct', 'Incorrect'],
                                       colors=colors, autopct='%1.1f%%',
                                       explode=(0.05, 0), startangle=90)
    ax.set_title(f'Overall Accuracy\n({correct}/{len(df)} correct)', fontweight='bold', fontsize=12)

    # 3. Confidence calibration
    ax = axes[0, 2]
    # Bin predictions by confidence
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    bin_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    expected = []
    actual_acc = []
    counts = []

    for (low, high), label in zip(bins, bin_labels):
        mask = (df['confidence'] >= low) & (df['confidence'] < high)
        if mask.sum() > 0:
            expected.append((low + high) / 2 * 100)
            actual_acc.append(df.loc[mask, 'correct'].mean() * 100)
            counts.append(mask.sum())
        else:
            expected.append((low + high) / 2 * 100)
            actual_acc.append(0)
            counts.append(0)

    x = np.arange(len(bin_labels))
    width = 0.35
    ax.bar(x - width/2, expected, width, label='Expected', color='#3498db', alpha=0.7)
    ax.bar(x + width/2, actual_acc, width, label='Actual', color='#2ecc71', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_ylabel('Win Rate (%)', fontweight='bold')
    ax.set_title('Calibration: Expected vs Actual Win Rate', fontweight='bold', fontsize=12)
    ax.legend()
    ax.set_ylim(0, 100)

    # Add count labels
    for i, count in enumerate(counts):
        ax.text(i, 5, f'n={count}', ha='center', fontsize=9)

    # 4. Z-score distribution (home team)
    ax = axes[1, 0]
    if 'home_cats_zscore' in df.columns:
        zscores = df['home_cats_zscore'].dropna()
        ax.hist(zscores, bins=20, color='#3498db', alpha=0.7, edgecolor='black', density=True)

        # Overlay normal distribution
        x_norm = np.linspace(-4, 4, 100)
        y_norm = (1 / np.sqrt(2 * np.pi)) * np.exp(-x_norm**2 / 2)
        ax.plot(x_norm, y_norm, 'r--', linewidth=2, label='Standard Normal')

        ax.axvline(x=0, color='green', linestyle='-', linewidth=2)
        ax.axvline(x=-1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        within_1std = (abs(zscores) <= 1).mean() * 100
        within_2std = (abs(zscores) <= 2).mean() * 100

        ax.set_xlabel('Z-Score (Categories Won)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title(f'Distribution of Prediction Errors\n(Within 1σ: {within_1std:.1f}%, Within 2σ: {within_2std:.1f}%)',
                     fontweight='bold', fontsize=12)
        ax.legend()

    # 5. Accuracy by confidence level
    ax = axes[1, 1]
    df['conf_bucket'] = pd.cut(df['confidence'], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               labels=['50-60%', '60-70%', '70-80%', '80-90%', '90-100%'])
    conf_accuracy = df.groupby('conf_bucket')['correct'].agg(['mean', 'count'])
    conf_accuracy['mean'] *= 100

    colors = ['#e74c3c' if acc < 50 else '#f39c12' if acc < 70 else '#2ecc71'
              for acc in conf_accuracy['mean'].values]
    bars = ax.bar(range(len(conf_accuracy)), conf_accuracy['mean'].values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(conf_accuracy)))
    ax.set_xticklabels(conf_accuracy.index, rotation=45)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy by Prediction Confidence', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 100)

    for bar, (_, row) in zip(bars, conf_accuracy.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{row["mean"]:.0f}%\n(n={int(row["count"])})', ha='center', fontsize=9)

    # 6. Cumulative accuracy over time
    ax = axes[1, 2]
    df_sorted = df.sort_values(['week', 'matchup'])
    df_sorted['cumulative_correct'] = df_sorted['correct'].cumsum()
    df_sorted['cumulative_total'] = range(1, len(df_sorted) + 1)
    df_sorted['cumulative_accuracy'] = df_sorted['cumulative_correct'] / df_sorted['cumulative_total'] * 100

    ax.plot(df_sorted['cumulative_total'], df_sorted['cumulative_accuracy'],
            'b-', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    ax.fill_between(df_sorted['cumulative_total'], 50, df_sorted['cumulative_accuracy'],
                    where=df_sorted['cumulative_accuracy'] >= 50, alpha=0.3, color='green')
    ax.fill_between(df_sorted['cumulative_total'], 50, df_sorted['cumulative_accuracy'],
                    where=df_sorted['cumulative_accuracy'] < 50, alpha=0.3, color='red')
    ax.set_xlabel('Matchups (Cumulative)', fontweight='bold')
    ax.set_ylabel('Cumulative Accuracy (%)', fontweight='bold')
    ax.set_title('Cumulative Accuracy Over Time', fontweight='bold', fontsize=12)
    ax.legend()
    ax.set_ylim(0, 100)

    plt.suptitle('Fantasy Basketball Simulation Validation Dashboard',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: validation_dashboard.png")
    plt.close()

    # Figure 2: Week-by-week detail
    n_weeks = len(all_results)
    n_cols = max(2, (n_weeks + 1) // 2)
    n_rows = 2 if n_weeks > 2 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_weeks == 1:
        axes = [axes] if not hasattr(axes, '__len__') else axes.flatten()
    else:
        axes = axes.flatten()

    for idx, (week, week_data) in enumerate(sorted(all_results.items())):
        if idx >= len(axes):
            break
        ax = axes[idx]

        week_df = df[df['week'] == week]

        # Show each matchup
        matchups = week_df['matchup'].values
        correct_vals = week_df['correct'].values
        confidences = week_df['confidence'].values

        y_pos = range(len(matchups))
        colors = ['#2ecc71' if c else '#e74c3c' for c in correct_vals]

        bars = ax.barh(y_pos, confidences * 100, color=colors, edgecolor='black', alpha=0.8)
        ax.axvline(x=50, color='gray', linestyle='--', linewidth=1)

        # Add matchup labels
        labels = []
        for m in matchups:
            parts = m.split(' vs ')
            if len(parts) == 2:
                labels.append(f"{parts[0][:15]}\nvs\n{parts[1][:15]}")
            else:
                labels.append(m[:20])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Confidence (%)', fontweight='bold')
        ax.set_title(f'Week {week}\n({sum(correct_vals)}/{len(correct_vals)} correct)',
                     fontweight='bold', fontsize=11)
        ax.set_xlim(0, 100)

        # Add legend
        legend_elements = [mpatches.Patch(color='#2ecc71', label='Correct'),
                          mpatches.Patch(color='#e74c3c', label='Incorrect')]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    # Hide unused axes
    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Week-by-Week Prediction Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'weekly_details.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: weekly_details.png")
    plt.close()

    # Figure 3: Category-level calibration
    category_df = create_category_calibration_visualization(all_results, output_dir)

    return df, category_df


def create_category_calibration_visualization(all_results: Dict, output_dir: Path) -> pd.DataFrame:
    """Create category-level calibration analysis and visualization."""

    # Collect all category-level comparisons
    all_cat_comparisons = []
    for week, week_results in all_results.items():
        for matchup, comp in week_results['comparisons'].items():
            for cat_comp in comp.get('category_comparisons', []):
                cat_comp['week'] = week
                cat_comp['matchup'] = matchup
                all_cat_comparisons.append(cat_comp)

    if not all_cat_comparisons:
        print("  ⚠️  No category comparisons to visualize")
        return pd.DataFrame()

    cat_df = pd.DataFrame(all_cat_comparisons)

    # Filter out ties for accuracy calculations
    cat_df_no_ties = cat_df[cat_df['correct'].notna()].copy()

    if len(cat_df_no_ties) == 0:
        print("  ⚠️  No non-tie category comparisons")
        return cat_df

    # Figure 3: Category-level dashboard (3x2 layout)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 3a. Accuracy by category
    ax = axes[0, 0]
    cat_accuracy = cat_df_no_ties.groupby('category')['correct'].mean() * 100
    cat_accuracy = cat_accuracy.reindex(CATEGORIES)
    colors = ['#2ecc71' if acc >= 60 else '#f39c12' if acc >= 50 else '#e74c3c'
              for acc in cat_accuracy.values]
    bars = ax.bar(range(len(cat_accuracy)), cat_accuracy.values, color=colors, edgecolor='black')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    ax.axhline(y=cat_accuracy.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Average ({cat_accuracy.mean():.1f}%)')
    ax.set_xticks(range(len(cat_accuracy)))
    ax.set_xticklabels(CATEGORIES, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Category Prediction Accuracy', fontweight='bold', fontsize=12)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, cat_accuracy.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')

    # 3b. Category calibration (expected vs actual by confidence bin)
    ax = axes[0, 1]
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    bin_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    expected = []
    actual_acc = []
    counts = []

    for (low, high), label in zip(bins, bin_labels):
        mask = (cat_df_no_ties['pred_confidence'] >= low) & (cat_df_no_ties['pred_confidence'] < high)
        if mask.sum() > 0:
            expected.append((low + high) / 2 * 100)
            actual_acc.append(cat_df_no_ties.loc[mask, 'correct'].mean() * 100)
            counts.append(mask.sum())
        else:
            expected.append((low + high) / 2 * 100)
            actual_acc.append(0)
            counts.append(0)

    x = np.arange(len(bin_labels))
    width = 0.35
    ax.bar(x - width/2, expected, width, label='Expected', color='#3498db', alpha=0.7)
    ax.bar(x + width/2, actual_acc, width, label='Actual', color='#2ecc71', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.set_ylabel('Win Rate (%)', fontweight='bold')
    ax.set_title('Category Calibration: Expected vs Actual', fontweight='bold', fontsize=12)
    ax.legend()
    ax.set_ylim(0, 100)
    for i, count in enumerate(counts):
        ax.text(i, 5, f'n={count}', ha='center', fontsize=9)

    # 3c. Sample size by category
    ax = axes[0, 2]
    cat_counts = cat_df_no_ties.groupby('category').size()
    cat_counts = cat_counts.reindex(CATEGORIES)
    ax.bar(range(len(cat_counts)), cat_counts.values, color='#9b59b6', edgecolor='black')
    ax.set_xticks(range(len(cat_counts)))
    ax.set_xticklabels(CATEGORIES, rotation=45, ha='right')
    ax.set_ylabel('Sample Size', fontweight='bold')
    ax.set_title('Predictions per Category (excl. ties)', fontweight='bold', fontsize=12)
    for i, val in enumerate(cat_counts.values):
        ax.text(i, val + 1, str(val), ha='center', fontsize=9)

    # 3d. Accuracy by week across all categories
    ax = axes[1, 0]
    week_cat_accuracy = cat_df_no_ties.groupby('week')['correct'].mean() * 100
    colors = ['#2ecc71' if acc >= 60 else '#f39c12' if acc >= 50 else '#e74c3c'
              for acc in week_cat_accuracy.values]
    bars = ax.bar(week_cat_accuracy.index, week_cat_accuracy.values, color=colors, edgecolor='black')
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random')
    ax.axhline(y=week_cat_accuracy.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Avg ({week_cat_accuracy.mean():.1f}%)')
    ax.set_xlabel('Week', fontweight='bold')
    ax.set_ylabel('Category Accuracy (%)', fontweight='bold')
    ax.set_title('Category Accuracy by Week', fontweight='bold', fontsize=12)
    ax.legend()
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, week_cat_accuracy.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')

    # 3e. Heatmap: Accuracy by Category x Week
    ax = axes[1, 1]
    pivot = cat_df_no_ties.pivot_table(
        index='category',
        columns='week',
        values='correct',
        aggfunc='mean'
    ) * 100
    pivot = pivot.reindex(CATEGORIES)

    # Ensure numeric dtype for imshow
    pivot_values = pivot.values.astype(float)
    im = ax.imshow(pivot_values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'W{w}' for w in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Week', fontweight='bold')
    ax.set_ylabel('Category', fontweight='bold')
    ax.set_title('Category Accuracy Heatmap', fontweight='bold', fontsize=12)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot_values[i, j]
            if not np.isnan(val):
                color = 'white' if val < 40 or val > 80 else 'black'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Accuracy %')

    # 3f. Overall category summary
    ax = axes[1, 2]
    ax.axis('off')

    total_cat_predictions = len(cat_df_no_ties)
    total_cat_correct = cat_df_no_ties['correct'].sum()
    overall_cat_accuracy = total_cat_correct / total_cat_predictions * 100

    # Find best/worst categories
    best_cat = cat_accuracy.idxmax()
    worst_cat = cat_accuracy.idxmin()

    summary_text = f"""
    CATEGORY-LEVEL SUMMARY
    ══════════════════════════════

    Total Category Predictions: {total_cat_predictions}
    Correct Predictions: {int(total_cat_correct)}
    Overall Accuracy: {overall_cat_accuracy:.1f}%

    Best Category: {best_cat} ({cat_accuracy[best_cat]:.1f}%)
    Worst Category: {worst_cat} ({cat_accuracy[worst_cat]:.1f}%)

    Categories Above 60%: {(cat_accuracy >= 60).sum()}/11
    Categories Above 50%: {(cat_accuracy >= 50).sum()}/11

    Weeks Analyzed: {cat_df_no_ties['week'].nunique()}
    Matchups Analyzed: {cat_df_no_ties['matchup'].nunique()}
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Category-Level Calibration Analysis\n(11 categories × weeks × matchups)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'category_calibration.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: category_calibration.png")
    plt.close()

    return cat_df


def generate_markdown_report(all_results: Dict, df: pd.DataFrame, output_dir: Path, category_df: pd.DataFrame = None):
    """Generate comprehensive markdown validation report."""

    report = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Header
    report.append("# Fantasy Basketball Simulation Validation Report\n\n")
    report.append(f"**Generated:** {timestamp}\n\n")
    report.append("---\n\n")

    # Executive Summary
    report.append("## Executive Summary\n\n")

    total_matchups = len(df)
    total_correct = df['correct'].sum()
    overall_accuracy = total_correct / total_matchups * 100 if total_matchups > 0 else 0

    report.append(f"| Metric | Value |\n")
    report.append(f"|--------|-------|\n")
    report.append(f"| **Total Matchups Analyzed** | {total_matchups} |\n")
    report.append(f"| **Correct Predictions** | {total_correct} |\n")
    report.append(f"| **Overall Accuracy** | **{overall_accuracy:.1f}%** |\n")
    report.append(f"| **Weeks Analyzed** | {df['week'].nunique()} |\n")

    # Assess performance
    if overall_accuracy >= 70:
        assessment = "🟢 **EXCELLENT** - Model significantly outperforms random chance"
    elif overall_accuracy >= 60:
        assessment = "🟡 **GOOD** - Model shows meaningful predictive power"
    elif overall_accuracy >= 50:
        assessment = "🟠 **MARGINAL** - Model slightly better than random"
    else:
        assessment = "🔴 **POOR** - Model underperforms random chance"

    report.append(f"| **Assessment** | {assessment} |\n")
    report.append("\n")

    # Add within std analysis
    if 'home_within_1std' in df.columns:
        within_1std = ((df['home_within_1std'] == True) | (df['away_within_1std'] == True)).mean() * 100
        within_2std = ((df['home_within_2std'] == True) | (df['away_within_2std'] == True)).mean() * 100
        report.append(f"| **Results within 1σ** | {within_1std:.1f}% (expected: 68%) |\n")
        report.append(f"| **Results within 2σ** | {within_2std:.1f}% (expected: 95%) |\n")

    report.append("\n---\n\n")

    # Validation Dashboard
    report.append("## Validation Dashboard\n\n")
    report.append("![Validation Dashboard](validation_dashboard.png)\n\n")
    report.append("**Dashboard Components:**\n")
    report.append("1. **Accuracy by Week** - How well did we predict each week?\n")
    report.append("2. **Overall Accuracy** - Total correct vs incorrect predictions\n")
    report.append("3. **Calibration** - Do 70% predictions win 70% of the time?\n")
    report.append("4. **Z-Score Distribution** - Are prediction errors normally distributed?\n")
    report.append("5. **Accuracy by Confidence** - Do high-confidence picks win more?\n")
    report.append("6. **Cumulative Accuracy** - How has accuracy evolved?\n")
    report.append("\n---\n\n")

    # Weekly Details
    report.append("## Weekly Breakdown\n\n")
    report.append("![Weekly Details](weekly_details.png)\n\n")

    # Table of weekly results
    report.append("### Week-by-Week Summary\n\n")
    report.append("| Week | Matchups | Correct | Accuracy | Avg Confidence |\n")
    report.append("|------|----------|---------|----------|----------------|\n")

    for week in sorted(df['week'].unique()):
        week_df = df[df['week'] == week]
        n_matchups = len(week_df)
        n_correct = week_df['correct'].sum()
        accuracy = n_correct / n_matchups * 100
        avg_conf = week_df['confidence'].mean() * 100

        acc_emoji = "✅" if accuracy >= 70 else "⚠️" if accuracy >= 50 else "❌"
        report.append(f"| Week {week} | {n_matchups} | {n_correct} | {acc_emoji} {accuracy:.1f}% | {avg_conf:.1f}% |\n")

    report.append("\n---\n\n")

    # Detailed matchup results
    report.append("## Detailed Matchup Results\n\n")

    for week in sorted(all_results.keys()):
        week_data = all_results[week]
        report.append(f"### Week {week}\n\n")

        report.append("| Matchup | Predicted | Actual | Confidence | Result | Within 1σ |\n")
        report.append("|---------|-----------|--------|------------|--------|----------|\n")

        for matchup, comp in week_data['comparisons'].items():
            pred = comp['predicted_winner_name'][:20]
            actual = comp['actual_winner_name'][:20]
            conf = comp['confidence'] * 100
            result = "✅" if comp['correct'] else "❌"

            within_1std = "✓" if comp.get('home_within_1std', False) or comp.get('away_within_1std', False) else "✗"

            # Shorten matchup name
            parts = matchup.split(' vs ')
            short_matchup = f"{parts[0][:12]} vs {parts[1][:12]}" if len(parts) == 2 else matchup[:25]

            report.append(f"| {short_matchup} | {pred} | {actual} | {conf:.1f}% | {result} | {within_1std} |\n")

        report.append("\n")

    report.append("---\n\n")

    # Calibration Analysis
    report.append("## Calibration Analysis\n\n")
    report.append("Calibration measures whether our confidence levels are accurate. ")
    report.append("If we predict 80% confidence, the team should win ~80% of those matchups.\n\n")

    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    bin_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

    report.append("| Confidence Bin | Expected Win Rate | Actual Win Rate | Sample Size | Calibration |\n")
    report.append("|----------------|-------------------|-----------------|-------------|-------------|\n")

    for (low, high), label in zip(bins, bin_labels):
        mask = (df['confidence'] >= low) & (df['confidence'] < high)
        n = mask.sum()
        if n > 0:
            expected = (low + high) / 2 * 100
            actual = df.loc[mask, 'correct'].mean() * 100
            diff = actual - expected

            if abs(diff) < 10:
                cal = "🟢 Good"
            elif abs(diff) < 20:
                cal = "🟡 Fair"
            else:
                cal = "🔴 Poor"

            report.append(f"| {label} | {expected:.1f}% | {actual:.1f}% | {n} | {cal} |\n")
        else:
            report.append(f"| {label} | - | - | 0 | N/A |\n")

    report.append("\n---\n\n")

    # Category-Level Analysis
    report.append("## Category-Level Calibration\n\n")
    report.append("This section analyzes prediction accuracy at the individual category level ")
    report.append("(11 categories × weeks × matchups).\n\n")
    report.append("![Category Calibration](category_calibration.png)\n\n")

    if category_df is not None and len(category_df) > 0:
        cat_df_no_ties = category_df[category_df['correct'].notna()].copy()

        if len(cat_df_no_ties) > 0:
            # Overall category stats
            total_cat_pred = len(cat_df_no_ties)
            total_cat_correct = cat_df_no_ties['correct'].sum()
            cat_overall_acc = total_cat_correct / total_cat_pred * 100

            report.append(f"### Category-Level Summary\n\n")
            report.append(f"| Metric | Value |\n")
            report.append(f"|--------|-------|\n")
            report.append(f"| **Total Category Predictions** | {total_cat_pred} |\n")
            report.append(f"| **Correct Predictions** | {int(total_cat_correct)} |\n")
            report.append(f"| **Overall Category Accuracy** | **{cat_overall_acc:.1f}%** |\n")
            report.append("\n")

            # Per-category accuracy
            report.append("### Accuracy by Category\n\n")
            report.append("| Category | Correct | Total | Accuracy | Assessment |\n")
            report.append("|----------|---------|-------|----------|------------|\n")

            cat_accuracy = cat_df_no_ties.groupby('category')['correct'].agg(['sum', 'count', 'mean'])
            cat_accuracy['mean'] *= 100

            for cat in CATEGORIES:
                if cat in cat_accuracy.index:
                    row = cat_accuracy.loc[cat]
                    acc = row['mean']
                    if acc >= 60:
                        assessment = "✅ Good"
                    elif acc >= 50:
                        assessment = "⚠️ Fair"
                    else:
                        assessment = "❌ Poor"
                    report.append(f"| {cat} | {int(row['sum'])} | {int(row['count'])} | {acc:.1f}% | {assessment} |\n")

            report.append("\n")

            # Category calibration table
            report.append("### Category Calibration by Confidence\n\n")
            report.append("| Confidence | Expected | Actual | Count | Calibration |\n")
            report.append("|------------|----------|--------|-------|-------------|\n")

            bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            bin_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

            for (low, high), label in zip(bins, bin_labels):
                mask = (cat_df_no_ties['pred_confidence'] >= low) & (cat_df_no_ties['pred_confidence'] < high)
                n = mask.sum()
                if n > 0:
                    expected = (low + high) / 2 * 100
                    actual = cat_df_no_ties.loc[mask, 'correct'].mean() * 100
                    diff = actual - expected
                    if abs(diff) < 10:
                        cal = "🟢 Good"
                    elif abs(diff) < 20:
                        cal = "🟡 Fair"
                    else:
                        cal = "🔴 Poor"
                    report.append(f"| {label} | {expected:.1f}% | {actual:.1f}% | {n} | {cal} |\n")
                else:
                    report.append(f"| {label} | - | - | 0 | N/A |\n")

            report.append("\n")

    report.append("---\n\n")

    # Insights and Recommendations
    report.append("## Insights and Recommendations\n\n")

    # High confidence analysis
    high_conf = df[df['confidence'] >= 0.8]
    if len(high_conf) > 0:
        high_conf_acc = high_conf['correct'].mean() * 100
        report.append(f"### High Confidence Predictions (≥80%)\n")
        report.append(f"- **Count:** {len(high_conf)} matchups\n")
        report.append(f"- **Accuracy:** {high_conf_acc:.1f}%\n")
        if high_conf_acc >= 80:
            report.append(f"- **Assessment:** ✅ High confidence predictions are reliable\n")
        elif high_conf_acc >= 60:
            report.append(f"- **Assessment:** ⚠️ High confidence predictions are somewhat reliable\n")
        else:
            report.append(f"- **Assessment:** ❌ High confidence predictions may be overconfident\n")
        report.append("\n")

    # Close matchup analysis
    close = df[df['confidence'] < 0.6]
    if len(close) > 0:
        close_acc = close['correct'].mean() * 100
        report.append(f"### Close Matchups (<60% confidence)\n")
        report.append(f"- **Count:** {len(close)} matchups\n")
        report.append(f"- **Accuracy:** {close_acc:.1f}%\n")
        report.append(f"- These are essentially coin-flip predictions where variance dominates\n")
        report.append("\n")

    # Category analysis
    report.append("### Variance Analysis\n")
    if 'home_cats_zscore' in df.columns:
        mean_abs_zscore = df['home_cats_zscore'].abs().mean()
        report.append(f"- **Mean Absolute Z-Score:** {mean_abs_zscore:.2f}\n")
        if mean_abs_zscore < 1:
            report.append(f"- **Assessment:** ✅ Predictions are well-calibrated (mean |z| < 1)\n")
        elif mean_abs_zscore < 1.5:
            report.append(f"- **Assessment:** ⚠️ Predictions have moderate variance\n")
        else:
            report.append(f"- **Assessment:** ❌ Predictions have high variance - model may need adjustment\n")

    report.append("\n---\n\n")

    # Footer
    report.append("## Methodology\n\n")
    report.append("1. **Data Collection:** Box scores fetched from ESPN Fantasy API for each completed week\n")
    report.append("2. **Simulation:** 500 Monte Carlo simulations per matchup using Bayesian player models\n")
    report.append("3. **Comparison:** Predicted winner compared against actual matchup outcome\n")
    report.append("4. **Z-Score Analysis:** Actual categories won compared to simulated distribution\n")
    report.append("\n")
    report.append("*Generated by Fantasy 2026 Simulation Validation System*\n")

    # Write report
    with open(output_dir / 'VALIDATION_REPORT.md', 'w') as f:
        f.writelines(report)

    print(f"  Saved: VALIDATION_REPORT.md")


def main():
    """Main execution."""
    args = parse_args()
    weeks = parse_week_range(args.weeks)

    print("="*80)
    print("FANTASY BASKETBALL SIMULATION VALIDATION")
    print("="*80)
    print(f"Weeks to validate: {weeks}")
    print(f"Skip fetch: {args.skip_fetch}")
    print(f"Skip simulations: {args.skip_simulations}")
    print("="*80)

    # Create output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = REPORTS_DIR / f'validation_report_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/\n")

    # Step 1: Fetch data for each week
    if not args.skip_fetch:
        print("Step 1: Fetching historical box scores from ESPN API...")
        for week in weeks:
            fetch_week_data(week)
        print()
    else:
        print("Step 1: Skipping fetch (using existing data)\n")

    # Step 2: Run simulations for each week
    all_results = {}

    print("Step 2: Running simulations and comparing results...")
    for week in weeks:
        print(f"\n  📊 Processing Week {week}...")

        # Find box scores
        box_file = find_box_scores_file(week)
        if box_file is None:
            print(f"     ⚠️  No box scores found for Week {week}, skipping")
            continue

        # Load actual results
        box_scores_df = pd.read_csv(box_file)
        actual_results = calculate_actual_results(box_scores_df, week)

        if not actual_results:
            print(f"     ⚠️  No actual results found for Week {week}, skipping")
            continue

        # Run simulation (or load existing)
        if not args.skip_simulations:
            report_dir = run_simulation_for_week(week)
        else:
            report_dirs = sorted(REPORTS_DIR.glob(f'week{week}_report_*'), reverse=True)
            report_dir = report_dirs[0] if report_dirs else None

        if report_dir is None:
            print(f"     ⚠️  No simulation results for Week {week}, skipping")
            continue

        # Load simulation results
        sim_results = load_simulation_results(report_dir)

        # Compare
        comparisons = {}
        for matchup_name, actual in actual_results.items():
            if matchup_name in sim_results:
                comparisons[matchup_name] = compare_results(actual, sim_results[matchup_name])
            else:
                print(f"     ⚠️  Matchup not in simulation: {matchup_name[:30]}...")

        week_correct = sum(1 for c in comparisons.values() if c['correct'])
        week_total = len(comparisons)
        if week_total > 0:
            print(f"     ✅ Week {week}: {week_correct}/{week_total} correct ({week_correct/week_total*100:.1f}%)")
        else:
            print(f"     ⚠️  Week {week}: No matchups to compare")

        all_results[week] = {
            'actual': actual_results,
            'simulated': sim_results,
            'comparisons': comparisons
        }

    if not all_results:
        print("\n❌ No results to analyze!")
        return

    # Step 3: Generate visualizations
    print("\n" + "="*80)
    print("Step 3: Generating visualizations...")
    df, category_df = create_validation_visualizations(all_results, output_dir)

    # Step 4: Generate report
    print("\nStep 4: Generating validation report...")
    generate_markdown_report(all_results, df, output_dir, category_df)

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)

    total_matchups = len(df)
    total_correct = df['correct'].sum()
    overall_accuracy = total_correct / total_matchups * 100

    print(f"\n📊 Overall Results:")
    print(f"   Total Matchups: {total_matchups}")
    print(f"   Correct Predictions: {total_correct}")
    print(f"   Overall Accuracy: {overall_accuracy:.1f}%")

    # Category-level summary
    if category_df is not None and len(category_df) > 0:
        cat_df_no_ties = category_df[category_df['correct'].notna()]
        if len(cat_df_no_ties) > 0:
            cat_correct = cat_df_no_ties['correct'].sum()
            cat_total = len(cat_df_no_ties)
            cat_accuracy = cat_correct / cat_total * 100
            print(f"\n📊 Category-Level Results:")
            print(f"   Total Category Predictions: {cat_total}")
            print(f"   Correct Predictions: {int(cat_correct)}")
            print(f"   Category Accuracy: {cat_accuracy:.1f}%")

    print(f"\n📁 Output Directory: {output_dir}/")
    print(f"\n📈 Generated Files:")
    print(f"   - VALIDATION_REPORT.md")
    print(f"   - validation_dashboard.png")
    print(f"   - weekly_details.png")
    print(f"   - category_calibration.png")
    print(f"\n🎯 Open the report: open {output_dir}/VALIDATION_REPORT.md")


if __name__ == '__main__':
    main()
