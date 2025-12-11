"""
Fixed Matchup Simulation - Uses Correct Data Source

This script simulates fantasy basketball matchups with flexible week selection.

Data source: box_scores_latest.csv (actual game results) + matchups_latest.csv
Modeling: BayesianProjectionModel (updated 2025-12-10)

Key improvements in new model:
- Separate 2P% and 3P% modeling (FG% derived, not sampled)
- Position-based priors (centers ~58% 2P%, guards ~47%)
- Last season only as prior (not all historical data)
- Different prior strengths for shooting (30 games) vs counting stats (10 games)

Modes:
- Historical: Use actual games from completed weeks (--week 6)
- Mid-week: Use actual games played + simulate remaining (default, assumes 4 games/player)

Note on ESPN Week Numbering:
- ESPN API returns an internal "matchup_period" number (e.g., 50)
- This is different from the fantasy week number displayed in the UI (e.g., Week 8)
- Our CSV files store both: 'week' column = internal ESPN period, 'matchup_period' = fantasy week
- When using --week, specify the FANTASY week (e.g., --week 8), not the internal number
"""

import pandas as pd
import sys
import json
import ast
import argparse
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

sys.path.append('/Users/rhu/fantasybasketball3')
sys.path.append('/Users/rhu/fantasybasketball3/fantasy_2026/projection_diagnostics')
from bayesian_projection_model import BayesianProjectionModel

# Week date ranges for automatic remaining games calculation
WEEK_DATES = {
    1: ("2024-10-22", "2024-10-27"),
    2: ("2024-10-28", "2024-11-03"),
    3: ("2024-11-04", "2024-11-10"),
    4: ("2024-11-11", "2024-11-17"),
    5: ("2024-11-18", "2024-11-24"),
    6: ("2024-11-25", "2024-12-01"),
    7: ("2024-12-02", "2024-12-08"),
    8: ("2024-12-09", "2024-12-15"),
    9: ("2024-12-16", "2024-12-22"),
    10: ("2024-12-23", "2024-12-29"),
    11: ("2024-12-30", "2025-01-05"),
    12: ("2025-01-06", "2025-01-12"),
    13: ("2025-01-13", "2025-01-19"),
    14: ("2025-01-20", "2025-01-26"),
    15: ("2025-01-27", "2025-02-02"),
    16: ("2025-02-03", "2025-02-09"),
}

# Set matplotlib style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def get_latest_week_from_matchups() -> Tuple[int, int]:
    """
    Get the latest week numbers from matchups_latest.csv.

    Returns:
        Tuple of (espn_week, fantasy_week) where:
        - espn_week: Internal ESPN matchup_period number (stored in 'week' column)
        - fantasy_week: User-facing fantasy week (stored in 'matchup_period' column)
    """
    matchups_file = '/Users/rhu/fantasybasketball3/fantasy_2026/data/matchups/matchups_latest.csv'
    matchups = pd.read_csv(matchups_file)
    espn_week = int(matchups['week'].max())
    # matchup_period column contains the actual fantasy week number
    fantasy_week = int(matchups['matchup_period'].max()) if 'matchup_period' in matchups.columns else espn_week
    return espn_week, fantasy_week


def fetch_historical_box_scores(fantasy_week: int) -> bool:
    """
    Fetch historical box scores from ESPN API for a specific fantasy week.

    Args:
        fantasy_week: The fantasy week number (e.g., 6, 7, 8)

    Returns:
        True if fetch was successful, False otherwise
    """
    print(f"\n📥 Fetching box scores for Fantasy Week {fantasy_week} from ESPN API...")

    # Check if ESPN credentials are set
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('ESPN_SWID')

    if not espn_s2 or not swid:
        print("  ❌ ERROR: ESPN credentials not found in environment variables")
        print("  Please set ESPN_S2 and ESPN_SWID to fetch historical data")
        return False

    # Run the 3_get_matchups.py script with the --week argument
    script_path = '/Users/rhu/fantasybasketball3/fantasy_2026/scripts/3_get_matchups.py'

    try:
        result = subprocess.run(
            ['python', script_path, '--week', str(fantasy_week)],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        if result.returncode == 0:
            print(f"  ✅ Successfully fetched box scores for Week {fantasy_week}")
            return True
        else:
            print(f"  ❌ Failed to fetch box scores:")
            print(f"     {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("  ❌ Timeout: ESPN API request took too long")
        return False
    except Exception as e:
        print(f"  ❌ Error fetching box scores: {e}")
        return False


def find_box_scores_for_week(fantasy_week: int) -> Tuple[Optional[str], int]:
    """
    Find box scores data for a specific fantasy week.

    Searches through available box score files (both latest and archived)
    to find data for the requested week.

    Args:
        fantasy_week: The fantasy week number to find

    Returns:
        Tuple of (file_path, espn_week) where:
        - file_path: Path to CSV file containing the week's data, or None if not found
        - espn_week: The ESPN internal week number (for filtering the CSV)
    """
    matchups_dir = Path('/Users/rhu/fantasybasketball3/fantasy_2026/data/matchups')

    # First check matchups_latest.csv to understand the week mapping
    matchups_file = matchups_dir / 'matchups_latest.csv'
    if matchups_file.exists():
        matchups_df = pd.read_csv(matchups_file)
        if 'matchup_period' in matchups_df.columns:
            # Find the ESPN week number for this fantasy week
            week_match = matchups_df[matchups_df['matchup_period'] == fantasy_week]
            if len(week_match) > 0:
                espn_week = int(week_match['week'].iloc[0])
            else:
                # Fantasy week not in current matchups - might be in archived files
                espn_week = None
        else:
            # Old format - 'week' is the fantasy week
            espn_week = fantasy_week
    else:
        espn_week = fantasy_week

    # Search for data in box_scores_latest.csv
    latest_file = matchups_dir / 'box_scores_latest.csv'
    if latest_file.exists() and espn_week is not None:
        df = pd.read_csv(latest_file)
        if espn_week in df['week'].values:
            return str(latest_file), espn_week

    # Search archived box score files
    archived_files = sorted(matchups_dir.glob('box_scores_*.csv'), reverse=True)
    for f in archived_files:
        if f.name == 'box_scores_latest.csv':
            continue

        try:
            df = pd.read_csv(f)
            # Check if this file contains our week
            # Try both the ESPN week number and fantasy week number
            if espn_week is not None and espn_week in df['week'].values:
                return str(f), espn_week
            if fantasy_week in df['week'].values:
                return str(f), fantasy_week
        except Exception as e:
            continue

    return None, espn_week if espn_week else fantasy_week


def calculate_remaining_games_from_dates(week: int, default_games: int = 4) -> int:
    """
    Calculate remaining games for a week based on today's date.

    Formula: round(remaining_days / 2)

    Args:
        week: Week number
        default_games: Default to return if week dates not available or week ended

    Returns:
        Number of games to simulate (minimum 0, maximum default_games)
    """
    if week not in WEEK_DATES:
        print(f"    ℹ️  Week {week} not in WEEK_DATES mapping, using default: {default_games} games")
        return default_games

    start_str, end_str = WEEK_DATES[week]
    end_date = datetime.strptime(end_str, '%Y-%m-%d').date()
    today = datetime.now().date()

    # If week hasn't ended yet, calculate remaining
    if today <= end_date:
        remaining_days = (end_date - today).days + 1  # +1 to include today
        remaining_games = round(remaining_days / 2.0)
        remaining_games = max(0, min(remaining_games, default_games))  # Clamp between 0 and default
        print(f"    📅 Week {week} ends {end_str} ({remaining_days} days remaining → {remaining_games} games)")
        return remaining_games
    else:
        # Week already ended
        print(f"    📅 Week {week} already ended on {end_str}, using historical mode")
        return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Simulate fantasy basketball matchups with flexible week selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulate current week (mid-week projection with actual + simulated games)
  python simulate_with_correct_data.py

  # Simulate specific historical week (backwards projection)
  python simulate_with_correct_data.py --week 6

  # Simulate with custom games-per-player assumption
  python simulate_with_correct_data.py --games-per-player 3
        """
    )

    parser.add_argument(
        '--week',
        type=int,
        default=None,
        help='Week number to simulate (default: latest week from matchups_latest.csv)'
    )

    parser.add_argument(
        '--games-per-player',
        type=int,
        default=4,
        help='Expected games per player for mid-week projections (default: 4)'
    )

    parser.add_argument(
        '--use-date-calc',
        action='store_true',
        help='Calculate remaining games automatically from week dates (remaining_days / 2)'
    )

    parser.add_argument(
        '--no-auto-fetch',
        action='store_true',
        help='Do not automatically fetch historical box scores from ESPN API'
    )

    return parser.parse_args()


def parse_box_score_stats(stat_str: str) -> Dict:
    """Parse the stat_0 column which contains a dictionary."""
    try:
        stat_dict = ast.literal_eval(stat_str)
        return stat_dict.get('total', {})
    except:
        return {}


def load_data(fantasy_week: int = None, auto_fetch: bool = True):
    """
    Load all required data.

    Args:
        fantasy_week: The fantasy week to load box scores for. If None, uses latest.
        auto_fetch: If True and box scores not found locally, fetch from ESPN API.

    Returns:
        Tuple of (box_scores, historical, mapping, espn_proj, espn_week)
        - box_scores: DataFrame with game data
        - historical: DataFrame with historical game logs
        - mapping: DataFrame with player name mappings
        - espn_proj: DataFrame with ESPN projections
        - espn_week: The ESPN internal week number used for filtering
    """
    print("Loading data...")

    # Determine which box scores file to load
    if fantasy_week is not None:
        # Find box scores for specific week
        box_scores_file, espn_week = find_box_scores_for_week(fantasy_week)

        if box_scores_file is None:
            if auto_fetch:
                # Try to fetch from ESPN API
                print(f"\n⚠️  No local box scores found for Fantasy Week {fantasy_week}")
                success = fetch_historical_box_scores(fantasy_week)
                if success:
                    # Retry finding the file after fetch
                    box_scores_file, espn_week = find_box_scores_for_week(fantasy_week)
                    if box_scores_file is None:
                        print(f"  ❌ Still no data found after fetch attempt")
                        box_scores = pd.DataFrame()
                    else:
                        box_scores = pd.read_csv(box_scores_file)
                        print(f"  Box scores: {len(box_scores)} records from {Path(box_scores_file).name}")
                else:
                    box_scores = pd.DataFrame()
            else:
                print(f"  ⚠️  No box scores found for Week {fantasy_week} (auto_fetch=False)")
                box_scores = pd.DataFrame()
        else:
            box_scores = pd.read_csv(box_scores_file)
            print(f"  Box scores: {len(box_scores)} records from {Path(box_scores_file).name} (Week filter: {espn_week})")
    else:
        # Load latest box scores
        box_scores = pd.read_csv('/Users/rhu/fantasybasketball3/fantasy_2026/data/matchups/box_scores_latest.csv')
        espn_week = int(box_scores['week'].max()) if len(box_scores) > 0 else None
        print(f"  Box scores: {len(box_scores)} records")

    # Load historical game logs for modeling
    historical = pd.read_csv('/Users/rhu/fantasybasketball3/fantasy_2026/data/historical_gamelogs/historical_gamelogs_latest.csv')
    print(f"  Historical: {len(historical)} game logs")

    # Load player mapping
    mapping = pd.read_csv('/Users/rhu/fantasybasketball3/fantasy_2026/data/mappings/player_mapping_latest.csv')
    mapping = mapping[['espn_name', 'nba_api_name']].drop_duplicates()
    mapping.columns = ['espn_name', 'nba_name']
    print(f"  Player mapping: {len(mapping)} players")

    # Load ESPN projections (fallback for players without historical data)
    espn_proj = pd.read_csv('/Users/rhu/fantasybasketball3/data/fantasy_basketball_clean2.csv')
    espn_proj = espn_proj.rename(columns={'PLAYER': 'player_name'})
    print(f"  ESPN projections: {len(espn_proj)} players")

    return box_scores, historical, mapping, espn_proj, espn_week


def get_player_position(roster_df: pd.DataFrame, player_name: str) -> str:
    """Get player position from roster data."""
    if roster_df is None:
        return 'SF'
    player_row = roster_df[roster_df['player_name'] == player_name]
    if len(player_row) == 0:
        return 'SF'
    position = player_row.iloc[0]['position']
    if position in ['PG', 'SG', 'SF', 'PF', 'C']:
        return position
    return 'SF'


def fit_player_models(historical: pd.DataFrame, espn_proj: pd.DataFrame,
                      roster_df: pd.DataFrame = None) -> Dict:
    """
    Fit BayesianProjectionModel for all players in historical data.

    Uses the new model with:
    - Separate 2P% and 3P% modeling (FG% derived)
    - Position-based priors
    - Last season only as prior
    """
    print("\nFitting player models (BayesianProjectionModel)...")

    player_models = {}

    # Get unique players
    unique_players = historical['PLAYER_NAME'].unique()

    fitted_count = 0
    for player_name in unique_players:
        # Get position for this player
        position = get_player_position(roster_df, player_name) if roster_df is not None else 'SF'

        model = BayesianProjectionModel(position=position)
        success = model.fit_player(historical, player_name, position=position)

        if success:
            player_models[player_name] = model
            fitted_count += 1

    print(f"  Fitted {fitted_count} player models from historical data")
    return player_models


def get_matchup_players_from_box_scores(box_scores: pd.DataFrame, matchup_name: str,
                                        matchups_df: pd.DataFrame, week: int,
                                        games_per_player: int = 4,
                                        roster_df: pd.DataFrame = None) -> Tuple[Dict, Dict, str, str, str]:
    """
    Extract players and game counts for a matchup.

    Returns:
        home_players: Dict of {player_name: games_to_simulate}
        away_players: Dict of {player_name: games_to_simulate}
        home_team_name: str
        away_team_name: str
        mode: 'historical' or 'mid-week'
    """
    # Check if we have box score data for this week
    if len(box_scores) == 0 or 'week' not in box_scores.columns:
        week_data = pd.DataFrame()
        matchup_data = pd.DataFrame()
    else:
        week_data = box_scores[box_scores['week'] == week]
        matchup_data = week_data[week_data['matchup'] == matchup_name].copy() if len(week_data) > 0 else pd.DataFrame()

    # Get team names - prefer box_scores, then matchups_df, then parse matchup name
    home_team_name = None
    away_team_name = None

    # First try: get from box_scores (actual data)
    if len(matchup_data) > 0:
        home_team_name = matchup_data[matchup_data['team_side'] == 'home']['team_name'].iloc[0]
        away_team_name = matchup_data[matchup_data['team_side'] == 'away']['team_name'].iloc[0]
    else:
        # Second try: parse from matchup name (format: "Team A vs Team B")
        parts = matchup_name.split(' vs ')
        if len(parts) == 2:
            potential_home = parts[0].strip()
            potential_away = parts[1].strip()

            # Third try: verify against matchups_df if available
            if not matchups_df.empty and 'week' in matchups_df.columns:
                # Try to find matching row by team names
                week_matchups = matchups_df[matchups_df['week'] == week]
                for _, row in week_matchups.iterrows():
                    if (row['home_team_name'] == potential_home and row['away_team_name'] == potential_away):
                        home_team_name = row['home_team_name']
                        away_team_name = row['away_team_name']
                        break

            # If not found in matchups_df, use parsed names
            if home_team_name is None:
                home_team_name = potential_home
                away_team_name = potential_away
        else:
            # Fallback if parsing fails
            home_team_name = "Home Team"
            away_team_name = "Away Team"

    # Determine mode
    if len(matchup_data) > 0:
        # We have actual game data - this is historical or partial mid-week
        matchup_data['parsed_stats'] = matchup_data['stat_0'].apply(parse_box_score_stats)
        matchup_data['games_played'] = matchup_data['parsed_stats'].apply(lambda x: x.get('GP', 0))

        # Get actual games played
        home_players_actual = {}
        for _, row in matchup_data[matchup_data['team_side'] == 'home'].iterrows():
            player_name = row['player_name']
            games = int(row['games_played'])
            if games > 0:
                home_players_actual[player_name] = games

        away_players_actual = {}
        for _, row in matchup_data[matchup_data['team_side'] == 'away'].iterrows():
            player_name = row['player_name']
            games = int(row['games_played'])
            if games > 0:
                away_players_actual[player_name] = games

        # Determine if this is historical or mid-week mode
        # Historical mode: Use actual games when week has complete data
        # Mid-week mode: Simulate remaining games up to games_per_player

        # Check if we should use historical mode (actual games only)
        # This happens when:
        # 1. Week is explicitly specified and has complete data, OR
        # 2. All players have reached the expected games_per_player
        max_home = max(home_players_actual.values()) if home_players_actual else 0
        max_away = max(away_players_actual.values()) if away_players_actual else 0

        # For historical analysis, use actual games
        # A week is "historical" if most players have similar game counts (week ended naturally)
        avg_home = sum(home_players_actual.values()) / len(home_players_actual) if home_players_actual else 0
        avg_away = sum(away_players_actual.values()) / len(away_players_actual) if away_players_actual else 0
        avg_games = (avg_home + avg_away) / 2

        # If average games is significantly below games_per_player AND we have data,
        # it means the week naturally had fewer games (historical)
        # Use historical mode if all players have roughly the same game count (week ended)
        use_historical = (max_home >= 2 and max_away >= 2 and
                         max_home <= max_away + 1 and max_away <= max_home + 1)

        if use_historical or (max_home >= games_per_player and max_away >= games_per_player):
            # Historical mode - use actual games only
            mode = 'historical'
            return home_players_actual, away_players_actual, home_team_name, away_team_name, mode
        else:
            # Mid-week mode - calculate remaining games
            mode = 'mid-week'
            home_players = {}
            for player, games in home_players_actual.items():
                remaining = max(0, games_per_player - games)
                if remaining > 0:
                    home_players[player] = remaining

            away_players = {}
            for player, games in away_players_actual.items():
                remaining = max(0, games_per_player - games)
                if remaining > 0:
                    away_players[player] = remaining

            # Include players who haven't played yet from roster
            if roster_df is not None and not roster_df.empty:
                home_roster = roster_df[roster_df['fantasy_team_name'] == home_team_name]
                away_roster = roster_df[roster_df['fantasy_team_name'] == away_team_name]

                # Add home players with 0 games
                for _, row in home_roster.iterrows():
                    player_name = row['player_name']
                    injury_status = row.get('injury_status', 'ACTIVE')

                    # Skip if player already in actual data or is OUT
                    if player_name not in home_players_actual and injury_status != 'OUT':
                        home_players[player_name] = games_per_player

                # Add away players with 0 games
                for _, row in away_roster.iterrows():
                    player_name = row['player_name']
                    injury_status = row.get('injury_status', 'ACTIVE')

                    if player_name not in away_players_actual and injury_status != 'OUT':
                        away_players[player_name] = games_per_player

            print(f"    📊 Mid-week mode: {len(home_players_actual)} home players ({sum(home_players_actual.values())} games played, {sum(home_players.values())} to simulate)")
            print(f"                     {len(away_players_actual)} away players ({sum(away_players_actual.values())} games played, {sum(away_players.values())} to simulate)")

            return home_players, away_players, home_team_name, away_team_name, mode
    else:
        # No data yet - full projection mode (simulate all games using roster)
        mode = 'mid-week'

        if roster_df is None or roster_df.empty:
            print(f"    ⚠️  No game data for Week {week} and no roster provided")
            return {}, {}, home_team_name, away_team_name, mode

        # Get players from roster for each team
        home_roster = roster_df[roster_df['fantasy_team_name'] == home_team_name]
        away_roster = roster_df[roster_df['fantasy_team_name'] == away_team_name]

        home_players = {}
        for _, row in home_roster.iterrows():
            player_name = row['player_name']
            # Only include active players (not injured)
            if row.get('injury_status', 'ACTIVE') != 'OUT':
                home_players[player_name] = games_per_player

        away_players = {}
        for _, row in away_roster.iterrows():
            player_name = row['player_name']
            if row.get('injury_status', 'ACTIVE') != 'OUT':
                away_players[player_name] = games_per_player

        print(f"    📊 Full projection mode: {len(home_players)} home players × {games_per_player} games = {len(home_players) * games_per_player} total")
        print(f"                           {len(away_players)} away players × {games_per_player} games = {len(away_players) * games_per_player} total")

        return home_players, away_players, home_team_name, away_team_name, mode


def map_player_name(espn_name: str, mapping: pd.DataFrame, all_model_names: set) -> str:
    """Map ESPN name to NBA API name."""
    # Try mapping file first
    match = mapping[mapping['espn_name'].str.lower() == espn_name.lower()]
    if len(match) > 0:
        return match.iloc[0]['nba_name']

    # Fallback: check if name exists directly in models
    if espn_name in all_model_names:
        return espn_name

    # Last resort: return original
    return espn_name


def simulate_matchup(home_players: Dict[str, int], away_players: Dict[str, int],
                    player_models: Dict, mapping: pd.DataFrame,
                    player_positions: pd.DataFrame,
                    n_simulations: int = 500) -> Tuple[pd.DataFrame, Dict]:
    """Simulate matchup using player models."""

    results = []
    category_wins = {'A': {}, 'B': {}, 'TIE': {}}
    unmapped_players = set()
    replacement_players = set()
    all_model_names = set(player_models.keys())

    for sim_num in range(n_simulations):
        # Simulate Team A (home)
        team_a_totals = {
            'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0,
            'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'DD': 0
        }

        for player_name, n_games in home_players.items():
            # Map to NBA name
            nba_name = map_player_name(player_name, mapping, all_model_names)

            if nba_name not in player_models:
                # Create replacement-level model on first simulation
                if sim_num == 0:
                    # Get position from player_positions
                    pos_match = player_positions[player_positions['player_name'] == player_name]
                    position = pos_match.iloc[0]['position'] if len(pos_match) > 0 else 'SF'

                    # Create replacement-level model using BayesianProjectionModel
                    model = BayesianProjectionModel(position=position)
                    model.fit_replacement_level(position)
                    player_models[player_name] = model
                    replacement_players.add(f"{player_name} ({position})")

                model = player_models[player_name]
            else:
                model = player_models[nba_name]

            # Simulate n_games for this player
            for _ in range(n_games):
                game = model.simulate_game()
                for stat in team_a_totals.keys():
                    team_a_totals[stat] += game.get(stat, 0)

        # Simulate Team B (away)
        team_b_totals = {
            'FGM': 0, 'FGA': 0, 'FTM': 0, 'FTA': 0, 'FG3M': 0, 'FG3A': 0,
            'PTS': 0, 'REB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'DD': 0
        }

        for player_name, n_games in away_players.items():
            nba_name = map_player_name(player_name, mapping, all_model_names)

            if nba_name not in player_models:
                # Create replacement-level model on first simulation
                if sim_num == 0:
                    # Get position from player_positions
                    pos_match = player_positions[player_positions['player_name'] == player_name]
                    position = pos_match.iloc[0]['position'] if len(pos_match) > 0 else 'SF'

                    # Create replacement-level model using BayesianProjectionModel
                    model = BayesianProjectionModel(position=position)
                    model.fit_replacement_level(position)
                    player_models[player_name] = model
                    replacement_players.add(f"{player_name} ({position})")

                model = player_models[player_name]
            else:
                model = player_models[nba_name]

            for _ in range(n_games):
                game = model.simulate_game()
                for stat in team_b_totals.keys():
                    team_b_totals[stat] += game.get(stat, 0)

        # Calculate category winners
        categories = calculate_category_winner(team_a_totals, team_b_totals)

        # Count wins
        a_cats = sum(1 for v in categories.values() if v == 'A')
        b_cats = sum(1 for v in categories.values() if v == 'B')
        ties = sum(1 for v in categories.values() if v == 'TIE')

        winner = 'A' if a_cats > b_cats else ('B' if b_cats > a_cats else 'TIE')

        # Track results
        results.append({
            'sim_num': sim_num,
            'team_a_cats': a_cats,
            'team_b_cats': b_cats,
            'ties': ties,
            'winner': winner,
            **{f'team_a_{k}': v for k, v in team_a_totals.items()},
            **{f'team_b_{k}': v for k, v in team_b_totals.items()}
        })

        # Track category wins
        for cat, win in categories.items():
            category_wins[win][cat] = category_wins[win].get(cat, 0) + 1

    if replacement_players:
        print(f"    ℹ️  Using replacement-level for {len(replacement_players)} rookies:")
        for player in sorted(replacement_players):
            print(f"             {player}")

    return pd.DataFrame(results), category_wins


def calculate_category_winner(team_a: Dict, team_b: Dict) -> Dict[str, str]:
    """Determine which team wins each category."""
    categories = {}

    # Percentage categories
    for pct_stat, makes, attempts in [('FG%', 'FGM', 'FGA'), ('FT%', 'FTM', 'FTA'), ('3P%', 'FG3M', 'FG3A')]:
        a_pct = team_a[makes] / team_a[attempts] if team_a[attempts] > 0 else 0
        b_pct = team_b[makes] / team_b[attempts] if team_b[attempts] > 0 else 0

        if a_pct > b_pct:
            categories[pct_stat] = 'A'
        elif b_pct > a_pct:
            categories[pct_stat] = 'B'
        else:
            categories[pct_stat] = 'TIE'

    # Counting categories (higher is better)
    for stat in ['FG3M', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'DD']:
        if team_a[stat] > team_b[stat]:
            categories[stat] = 'A'
        elif team_b[stat] > team_a[stat]:
            categories[stat] = 'B'
        else:
            categories[stat] = 'TIE'

    # Turnovers (lower is better)
    if team_a['TOV'] < team_b['TOV']:
        categories['TO'] = 'A'
    elif team_b['TOV'] < team_a['TOV']:
        categories['TO'] = 'B'
    else:
        categories['TO'] = 'TIE'

    return categories


def load_all_simulation_data():
    """Load all simulation results and summaries."""
    base_dir = Path('/Users/rhu/fantasybasketball3/fantasy_2026/fixed_simulations')

    # Load overall summary
    summary_df = pd.read_csv(base_dir / 'all_matchups_summary.csv')

    # Load individual matchup data
    matchup_data = {}
    for matchup_dir in base_dir.iterdir():
        if matchup_dir.is_dir():
            summary_file = matchup_dir / 'summary.json'
            sims_file = matchup_dir / 'all_simulations.csv'

            if summary_file.exists() and sims_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                sims = pd.read_csv(sims_file)

                matchup_data[summary['matchup']] = {
                    'summary': summary,
                    'simulations': sims
                }

    return summary_df, matchup_data


def load_actual_results(week: int):
    """Load actual matchup results from box_scores."""
    box_scores = pd.read_csv('/Users/rhu/fantasybasketball3/fantasy_2026/data/matchups/box_scores_latest.csv')
    week_data = box_scores[box_scores['week'] == week]

    actual_results = {}
    for matchup in week_data['matchup'].unique():
        matchup_data = week_data[week_data['matchup'] == matchup]

        # Get team names
        home_team = matchup_data[matchup_data['team_side'] == 'home']['team_name'].iloc[0]
        away_team = matchup_data[matchup_data['team_side'] == 'away']['team_name'].iloc[0]

        actual_results[matchup] = {
            'home_team': home_team,
            'away_team': away_team
        }

    return actual_results


def create_overview_visualizations(summary_df, actual_results, output_dir, week: int):
    """Create overview visualizations."""

    # Figure with 4 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Win Probabilities Bar Chart
    ax1 = fig.add_subplot(gs[0, :])

    matchups_list = []
    team_a_probs = []
    team_b_probs = []
    colors_a = []
    colors_b = []

    for _, row in summary_df.iterrows():
        matchup_name = row['matchup']
        team_a_name = row['team_a_name']
        team_b_name = row['team_b_name']

        matchups_list.append(f"{team_a_name}\nvs\n{team_b_name}")
        team_a_probs.append(row['team_a_win_pct'] * 100)
        team_b_probs.append(row['team_b_win_pct'] * 100)

        # Color based on confidence
        if row['team_a_win_pct'] > 0.7:
            colors_a.append('#2ecc71')  # Green for favorite
            colors_b.append('#e74c3c')  # Red for underdog
        elif row['team_b_win_pct'] > 0.7:
            colors_a.append('#e74c3c')
            colors_b.append('#2ecc71')
        else:
            colors_a.append('#3498db')  # Blue for competitive
            colors_b.append('#e67e22')  # Orange for competitive

    x = np.arange(len(matchups_list))
    width = 0.35

    bars1 = ax1.barh(x - width/2, team_a_probs, width, label='Home Team', color=colors_a, alpha=0.8)
    bars2 = ax1.barh(x + width/2, team_b_probs, width, label='Away Team', color=colors_b, alpha=0.8)

    ax1.set_ylabel('Matchup', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Win Probability (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Week {week} Matchup Win Probabilities', fontsize=16, fontweight='bold', pad=20)
    ax1.set_yticks(x)
    ax1.set_yticklabels(matchups_list, fontsize=9)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        width1 = bar1.get_width()
        width2 = bar2.get_width()
        ax1.text(width1 + 1, bar1.get_y() + bar1.get_height()/2,
                f'{width1:.1f}%', ha='left', va='center', fontsize=8, fontweight='bold')
        ax1.text(width2 + 1, bar2.get_y() + bar2.get_height()/2,
                f'{width2:.1f}%', ha='left', va='center', fontsize=8, fontweight='bold')

    # 2. Average Categories Won
    ax2 = fig.add_subplot(gs[1, 0])

    team_a_cats = summary_df['team_a_avg_cats_won'].values
    team_b_cats = summary_df['team_b_avg_cats_won'].values

    x_cats = np.arange(len(summary_df))
    width_cats = 0.35

    ax2.bar(x_cats - width_cats/2, team_a_cats, width_cats, label='Home Team', color='#3498db', alpha=0.8)
    ax2.bar(x_cats + width_cats/2, team_b_cats, width_cats, label='Away Team', color='#e67e22', alpha=0.8)
    ax2.axhline(y=5.5, color='red', linestyle='--', linewidth=2, label='Win Threshold (6 cats)')

    ax2.set_ylabel('Avg Categories Won', fontsize=11, fontweight='bold')
    ax2.set_title('Average Categories Won (out of 11)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_cats)
    ax2.set_xticklabels([f"M{i+1}" for i in range(len(summary_df))], fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Competitiveness Scores
    ax3 = fig.add_subplot(gs[1, 1])

    # Calculate competitiveness (1 - abs(win_pct_diff))
    competitiveness = []
    for _, row in summary_df.iterrows():
        diff = abs(row['team_a_win_pct'] - row['team_b_win_pct'])
        comp_score = (1 - diff) * 100
        competitiveness.append(comp_score)

    colors_comp = []
    for score in competitiveness:
        if score > 60:
            colors_comp.append('#2ecc71')  # Green - competitive
        elif score > 30:
            colors_comp.append('#f39c12')  # Yellow - moderate
        else:
            colors_comp.append('#e74c3c')  # Red - mismatch

    bars = ax3.barh(range(len(competitiveness)), competitiveness, color=colors_comp, alpha=0.8)
    ax3.set_xlabel('Competitiveness Score (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Matchup Competitiveness\n(100% = perfectly even)', fontsize=13, fontweight='bold')
    ax3.set_yticks(range(len(summary_df)))
    ax3.set_yticklabels([f"Matchup {i+1}" for i in range(len(summary_df))], fontsize=9)

    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, competitiveness)):
        width = bar.get_width()
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='Competitive (>60%)'),
        Patch(facecolor='#f39c12', alpha=0.8, label='Moderate (30-60%)'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Mismatch (<30%)')
    ]
    ax3.legend(handles=legend_elements, loc='lower right', fontsize=8)
    ax3.grid(axis='x', alpha=0.3)

    # 4. Game Count Comparison
    ax4 = fig.add_subplot(gs[1, 2])

    team_a_games = summary_df['team_a_total_games'].values
    team_b_games = summary_df['team_b_total_games'].values

    scatter_colors = []
    for i in range(len(team_a_games)):
        if abs(team_a_games[i] - team_b_games[i]) <= 3:
            scatter_colors.append('#2ecc71')  # Even games
        else:
            scatter_colors.append('#e67e22')  # Uneven games

    ax4.scatter(team_a_games, team_b_games, s=200, alpha=0.6, c=scatter_colors)

    # Add diagonal line (equal games)
    max_games = max(team_a_games.max(), team_b_games.max())
    ax4.plot([0, max_games], [0, max_games], 'k--', alpha=0.3, linewidth=2, label='Equal Games')

    # Label each point with matchup number
    for i, (x, y) in enumerate(zip(team_a_games, team_b_games)):
        ax4.annotate(f'M{i+1}', (x, y), fontsize=9, fontweight='bold',
                    ha='center', va='center')

    ax4.set_xlabel('Home Team Total Games', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Away Team Total Games', fontsize=11, fontweight='bold')
    ax4.set_title('Game Count Comparison', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # 5. Win Probability Distribution
    ax5 = fig.add_subplot(gs[2, :])

    all_win_probs = []
    for _, row in summary_df.iterrows():
        all_win_probs.extend([row['team_a_win_pct'], row['team_b_win_pct']])

    ax5.hist(all_win_probs, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax5.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='50% (Even Matchup)')
    ax5.axvline(x=np.mean(all_win_probs), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(all_win_probs):.2%}')

    ax5.set_xlabel('Win Probability', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Distribution of Win Probabilities Across All Teams', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    plt.suptitle(f'Week {week} Fantasy Basketball - Matchup Simulation Overview',
                fontsize=20, fontweight='bold', y=0.995)

    plt.savefig(output_dir / 'overview_visualizations.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: overview_visualizations.png")
    plt.close()


def create_matchup_detail_viz(matchup_name, matchup_data, output_dir):
    """Create detailed visualizations for a single matchup."""

    summary = matchup_data['summary']
    sims = matchup_data['simulations']

    team_a_name = summary['team_a_name']
    team_b_name = summary['team_b_name']

    # Create figure with 3x4 grid for 11 categories + summary
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    categories = [
        ('FG%', 'FGM', 'FGA'),
        ('FT%', 'FTM', 'FTA'),
        ('3P%', 'FG3M', 'FG3A'),
        ('FG3M', 'FG3M', None),
        ('PTS', 'PTS', None),
        ('REB', 'REB', None),
        ('AST', 'AST', None),
        ('STL', 'STL', None),
        ('BLK', 'BLK', None),
        ('TO', 'TOV', None),
        ('DD', 'DD', None)
    ]

    for idx, (cat_name, stat_a, stat_b) in enumerate(categories):
        ax = axes[idx]

        # Calculate values
        if cat_name in ['FG%', 'FT%', '3P%']:
            team_a_vals = sims[f'team_a_{stat_a}'] / sims[f'team_a_{stat_b}']
            team_b_vals = sims[f'team_b_{stat_a}'] / sims[f'team_b_{stat_b}']
        else:
            team_a_vals = sims[f'team_a_{stat_a}']
            team_b_vals = sims[f'team_b_{stat_a}']

        # Calculate statistics
        a_mean = team_a_vals.mean()
        a_median = team_a_vals.median()
        a_std = team_a_vals.std()
        b_mean = team_b_vals.mean()
        b_median = team_b_vals.median()
        b_std = team_b_vals.std()

        # Plot distributions
        ax.hist(team_a_vals, bins=30, alpha=0.5, color='#3498db', label=team_a_name[:20], density=True, edgecolor='darkblue', linewidth=0.5)
        ax.hist(team_b_vals, bins=30, alpha=0.5, color='#e67e22', label=team_b_name[:20], density=True, edgecolor='darkorange', linewidth=0.5)

        # Add mean lines
        ax.axvline(a_mean, color='#2c3e50', linestyle='--', linewidth=2.5, label=f'{team_a_name[:10]} μ={a_mean:.1f}')
        ax.axvline(b_mean, color='#d35400', linestyle='--', linewidth=2.5, label=f'{team_b_name[:10]} μ={b_mean:.1f}')

        # Add median lines (dotted)
        ax.axvline(a_median, color='#2c3e50', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(b_median, color='#d35400', linestyle=':', linewidth=1.5, alpha=0.7)

        # Calculate win percentage
        if cat_name == 'TO':
            team_a_wins = (team_a_vals < team_b_vals).sum()
        else:
            team_a_wins = (team_a_vals > team_b_vals).sum()

        win_pct_a = team_a_wins / len(team_a_vals) * 100
        win_pct_b = 100 - win_pct_a

        # Add shaded region for std dev
        ax.axvspan(a_mean - a_std, a_mean + a_std, alpha=0.1, color='#3498db')
        ax.axvspan(b_mean - b_std, b_mean + b_std, alpha=0.1, color='#e67e22')

        # Title with win percentages and stats
        title_text = f'{cat_name}\n{team_a_name[:15]}: {win_pct_a:.1f}% | {team_b_name[:15]}: {win_pct_b:.1f}%'
        if cat_name in ['FG%', 'FT%', '3P%']:
            title_text += f'\nμ: {a_mean:.3f} vs {b_mean:.3f} | σ: {a_std:.3f} vs {b_std:.3f}'
        else:
            title_text += f'\nμ: {a_mean:.1f} vs {b_mean:.1f} | σ: {a_std:.1f} vs {b_std:.1f}'

        ax.set_title(title_text, fontsize=9, fontweight='bold')
        ax.set_xlabel(cat_name, fontsize=9, fontweight='bold')
        ax.set_ylabel('Density', fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3)

    # Use last subplot for summary stats
    ax = axes[-1]
    ax.axis('off')

    summary_text = f"""
╔══════════════════════════════════════╗
║        MATCHUP SUMMARY               ║
╚══════════════════════════════════════╝

{team_a_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Win Probability:    {summary['team_a_win_pct']:.1%}
Wins (of 500):      {summary['team_a_wins']}
Avg Cats Won:       {summary['team_a_avg_cats_won']:.2f} / 11
Players:            {summary['team_a_players']}
Total Games:        {summary['team_a_total_games']}

{team_b_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Win Probability:    {summary['team_b_win_pct']:.1%}
Wins (of 500):      {summary['team_b_wins']}
Avg Cats Won:       {summary['team_b_avg_cats_won']:.2f} / 11
Players:            {summary['team_b_players']}
Total Games:        {summary['team_b_total_games']}

Simulation Details
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Simulations:  {summary['n_simulations']}
Ties:               {summary['ties']}

Legend:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
― ― (dashed):  Mean (μ)
··· (dotted):   Median
░░ (shaded):    ±1 Std Dev (σ)
    """

    ax.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
           verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, edgecolor='black', linewidth=2))

    plt.suptitle(f'Category Distributions: {team_a_name} vs {team_b_name}',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    safe_name = matchup_name.replace(' ', '_').replace('vs', 'vs')
    plt.savefig(output_dir / f'{safe_name}_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {safe_name}_distributions.png")
    plt.close()


def generate_markdown_report(summary_df, matchup_data, actual_results, output_dir, run_timestamp, week: int):
    """Generate comprehensive markdown report."""

    report = []
    report.append(f"# Week {week} Fantasy Basketball - Comprehensive Matchup Analysis\n\n")

    # Metadata table
    report.append("## Report Metadata\n\n")
    report.append("| Attribute | Value |\n")
    report.append("|-----------|-------|\n")
    report.append(f"| **Generated** | {run_timestamp} |\n")
    report.append(f"| **Simulations Per Matchup** | 500 |\n")
    report.append(f"| **Total Matchups** | {len(summary_df)} |\n")
    report.append(f"| **Week** | {week} |\n")
    report.append(f"| **Data Source** | box_scores_latest.csv |\n")
    report.append(f"| **Model** | Bayesian (Beta-Binomial + Poisson) |\n")
    report.append(f"| **Historical Data** | 2019-2024 seasons |\n")
    report.append(f"| **Evolution Rate** | 0.5 |\n")
    report.append("\n---\n")

    # Overview Image
    report.append("## Overview Dashboard\n\n")
    report.append("### Complete Matchup Overview\n")
    report.append("![Overview](overview_visualizations.png)\n\n")

    report.append("**Dashboard Components:**\n")
    report.append("1. **Win Probabilities** - Predicted win % for each team (Green=favorite, Red=underdog)\n")
    report.append("2. **Average Categories Won** - Expected categories won out of 11 (dashed line = 6 needed to win)\n")
    report.append("3. **Competitiveness Scores** - How evenly matched (Green=>60%, Yellow=30-60%, Red=<30%)\n")
    report.append("4. **Game Count Comparison** - Scheduling fairness (diagonal = equal games)\n")
    report.append("5. **Win Probability Distribution** - Overall confidence spread\n")
    report.append("\n---\n")

    # Statistical Summary
    report.append("## Statistical Summary\n\n")
    report.append("| Metric | Value |\n")
    report.append("|--------|-------|\n")
    report.append(f"| Total Matchups | {len(summary_df)} |\n")
    report.append(f"| Mean Win Probability Spread | {(summary_df['team_a_win_pct'] - summary_df['team_b_win_pct']).abs().mean():.1%} |\n")
    report.append(f"| Median Win Probability | {summary_df[['team_a_win_pct', 'team_b_win_pct']].values.flatten().mean():.1%} |\n")
    report.append(f"| Competitive Matchups (>40% both teams) | {((summary_df['team_a_win_pct'] > 0.4) & (summary_df['team_b_win_pct'] > 0.4)).sum()} |\n")
    report.append(f"| High Confidence Predictions (>80%) | {((summary_df['team_a_win_pct'] > 0.8) | (summary_df['team_b_win_pct'] > 0.8)).sum()} |\n")
    report.append(f"| Average Games Per Team | {summary_df[['team_a_total_games', 'team_b_total_games']].values.mean():.1f} |\n")
    report.append(f"| Average Players Per Team | {summary_df[['team_a_players', 'team_b_players']].values.mean():.1f} |\n")
    report.append("\n---\n")

    # Individual Matchup Analysis
    report.append("## Individual Matchup Analysis\n\n")

    for idx, row in summary_df.iterrows():
        matchup_name = row['matchup']
        team_a = row['team_a_name']
        team_b = row['team_b_name']

        report.append(f"### {idx + 1}. {team_a} vs {team_b}\n\n")

        # Determine competitiveness
        diff = abs(row['team_a_win_pct'] - row['team_b_win_pct'])
        if diff < 0.3:
            comp_label = "🟢 COMPETITIVE"
            comp_desc = "Close matchup - expect nail-biter"
        elif diff < 0.5:
            comp_label = "🟡 MODERATE"
            comp_desc = "Slight favorite exists"
        else:
            comp_label = "🔴 MISMATCH"
            comp_desc = "Clear favorite - likely blowout"

        report.append(f"**Competitiveness:** {comp_label} - {comp_desc}\n\n")

        # Summary Table
        report.append("#### Matchup Summary\n\n")
        report.append("| Metric | {} | {} |\n".format(team_a, team_b))
        report.append("|--------|{}|{}|\n".format("-" * max(15, len(team_a)), "-" * max(15, len(team_b))))
        report.append(f"| **Win Probability** | **{row['team_a_win_pct']:.1%}** | **{row['team_b_win_pct']:.1%}** |\n")
        report.append(f"| Wins (out of 500) | {row['team_a_wins']} | {row['team_b_wins']} |\n")
        report.append(f"| Ties | {row['ties']} | {row['ties']} |\n")
        report.append(f"| Avg Categories Won | {row['team_a_avg_cats_won']:.2f} / 11 | {row['team_b_avg_cats_won']:.2f} / 11 |\n")
        report.append(f"| Players | {row['team_a_players']} | {row['team_b_players']} |\n")
        report.append(f"| Total Games | {row['team_a_total_games']} | {row['team_b_total_games']} |\n")

        # Add game count balance
        game_diff = abs(row['team_a_total_games'] - row['team_b_total_games'])
        if game_diff <= 3:
            balance = "Even schedules"
        else:
            balance = f"{'Home' if row['team_a_total_games'] > row['team_b_total_games'] else 'Away'} has {game_diff} more games"
        report.append(f"| **Schedule Balance** | {balance} | {balance} |\n")
        report.append("\n")

        # Get detailed category stats if available
        if matchup_name in matchup_data:
            sims = matchup_data[matchup_name]['simulations']

            report.append("#### Category-by-Category Breakdown\n\n")
            report.append("| Category | {} Mean ± SD | {} Mean ± SD | Win % | Win % |\n".format(team_a[:15], team_b[:15]))
            report.append("|----------|{}|{}|-------|-------|\n".format("-" * 20, "-" * 20))

            categories = [
                ('FG%', 'FGM', 'FGA', False),
                ('FT%', 'FTM', 'FTA', False),
                ('3P%', 'FG3M', 'FG3A', False),
                ('3PM', 'FG3M', None, True),
                ('PTS', 'PTS', None, True),
                ('REB', 'REB', None, True),
                ('AST', 'AST', None, True),
                ('STL', 'STL', None, True),
                ('BLK', 'BLK', None, True),
                ('TO', 'TOV', None, False),  # Lower is better
                ('DD', 'DD', None, True)
            ]

            for cat_name, stat_a, stat_b, higher_better in categories:
                if cat_name in ['FG%', 'FT%', '3P%']:
                    team_a_vals = sims[f'team_a_{stat_a}'] / sims[f'team_a_{stat_b}']
                    team_b_vals = sims[f'team_b_{stat_a}'] / sims[f'team_b_{stat_b}']
                    fmt = '.3f'
                else:
                    team_a_vals = sims[f'team_a_{stat_a}']
                    team_b_vals = sims[f'team_b_{stat_a}']
                    fmt = '.1f'

                a_mean = team_a_vals.mean()
                a_std = team_a_vals.std()
                b_mean = team_b_vals.mean()
                b_std = team_b_vals.std()

                # Calculate win percentages
                if cat_name == 'TO':  # Lower is better
                    a_win_pct = (team_a_vals < team_b_vals).sum() / len(team_a_vals) * 100
                else:  # Higher is better
                    a_win_pct = (team_a_vals > team_b_vals).sum() / len(team_a_vals) * 100

                b_win_pct = 100 - a_win_pct

                report.append(f"| **{cat_name}** | {a_mean:{fmt}} ± {a_std:{fmt}} | {b_mean:{fmt}} ± {b_std:{fmt}} | {a_win_pct:.1f}% | {b_win_pct:.1f}% |\n")

            report.append("\n")

        # Visualization
        report.append("#### Full Category Distributions\n\n")
        safe_name = matchup_name.replace(' ', '_').replace('vs', 'vs')
        report.append(f"![{matchup_name} Distributions]({safe_name}_distributions.png)\n\n")

        report.append("**Visualization Guide:**\n")
        report.append("- Blue histogram = {}, Orange histogram = {}\n".format(team_a, team_b))
        report.append("- Dashed lines (--) = Mean values (μ)\n")
        report.append("- Dotted lines (···) = Median values\n")
        report.append("- Shaded regions = ±1 Standard Deviation (σ)\n")
        report.append("- Win % shown in title = probability of winning that specific category\n")
        report.append("\n")

        report.append("---\n\n")

    # Methodology
    report.append("## Methodology\n\n")
    report.append("### Simulation Approach\n")
    report.append("1. **Data Source:** Actual games played from `box_scores_latest.csv` (Week 6, October 2025)\n")
    report.append("2. **Player Models:** Bayesian projection models fitted on historical data (2019-2024)\n")
    report.append("3. **Simulations:** 500 Monte Carlo simulations per matchup\n")
    report.append("4. **Categories:** 11 standard fantasy basketball categories\n\n")

    report.append("### Model Details\n")
    report.append("- **Shooting Stats:** Beta-Binomial conjugate models with position-specific priors\n")
    report.append("- **Counting Stats:** Poisson distribution sampling with recency weighting\n")
    report.append("- **Category Winners:** Direct comparison of aggregated team totals\n")
    report.append("- **Matchup Winner:** Team winning 6+ categories\n\n")

    report.append("### Validation\n")
    report.append("- **Week 6 Accuracy:** 7/7 (100%)\n")
    report.append("- **Confidence Calibration:** Very good across all confidence levels\n")
    report.append("- See `SIMULATION_FIX_REPORT.md` for detailed validation analysis\n\n")

    report.append("---\n\n")
    report.append("*Generated by Fantasy 2026 Simulation System*\n")
    report.append(f"*Output Directory: `{output_dir}/`*\n")

    # Write report
    with open(output_dir / 'CONSOLIDATED_REPORT.md', 'w') as f:
        f.writelines(report)

    print(f"  Saved: CONSOLIDATED_REPORT.md")


def save_metadata(output_dir, summary_df, run_timestamp, week: int):
    """Save run metadata to JSON file."""
    metadata = {
        'run_timestamp': run_timestamp,
        'week': week,
        'total_matchups': len(summary_df),
        'simulations_per_matchup': 500,
        'data_source': 'box_scores_latest.csv',
        'model_type': 'Bayesian (Beta-Binomial + Poisson)',
        'historical_data': '2019-2024 seasons',
        'evolution_rate': 0.5,
        'categories': ['FG%', 'FT%', '3P%', '3PM', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'DD'],
        'matchup_summary': {
            'competitive': int(((summary_df['team_a_win_pct'] > 0.4) & (summary_df['team_b_win_pct'] > 0.4)).sum()),
            'high_confidence': int(((summary_df['team_a_win_pct'] > 0.8) | (summary_df['team_b_win_pct'] > 0.8)).sum()),
            'mean_win_prob_spread': float((summary_df['team_a_win_pct'] - summary_df['team_b_win_pct']).abs().mean()),
            'avg_games_per_team': float(summary_df[['team_a_total_games', 'team_b_total_games']].values.mean()),
            'avg_players_per_team': float(summary_df[['team_a_players', 'team_b_players']].values.mean())
        }
    }

    with open(output_dir / 'RUN_METADATA.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: RUN_METADATA.json")


def main():
    """Main execution."""
    # Parse command line arguments
    args = parse_args()

    # Determine week (using fantasy week, not ESPN internal week)
    if args.week is None:
        espn_week, fantasy_week = get_latest_week_from_matchups()
        print(f"ℹ️  No week specified, using latest week from matchups:")
        print(f"   Fantasy Week: {fantasy_week}, ESPN Internal Week: {espn_week}")
        target_week = fantasy_week  # User-facing week
        target_espn_week = espn_week  # For filtering box scores
    else:
        target_week = args.week
        # Will determine ESPN week during data loading
        target_espn_week = None
        print(f"ℹ️  Simulating Fantasy Week {target_week} (user-specified)")

    # Determine games per player (either from args or date-based calculation)
    if args.use_date_calc:
        games_per_player = calculate_remaining_games_from_dates(target_week, args.games_per_player)
        calculation_method = "date-based (remaining_days / 2)"
    else:
        games_per_player = args.games_per_player
        calculation_method = "fixed"

    print("="*80)
    print("FANTASY BASKETBALL MATCHUP SIMULATION")
    print("="*80)
    print(f"Fantasy Week: {target_week}")
    print(f"Games per player: {games_per_player} ({calculation_method})")
    print(f"Data source: box_scores + matchups (auto-fetch if missing)")
    print("="*80)

    # Load data (with auto-fetch if historical data not found locally)
    box_scores, historical, mapping, espn_proj, loaded_espn_week = load_data(
        fantasy_week=target_week if args.week is not None else None,
        auto_fetch=not args.no_auto_fetch
    )

    # Update ESPN week if we loaded specific week
    if loaded_espn_week is not None:
        target_espn_week = loaded_espn_week

    # Load matchups for the target week
    matchups_file = '/Users/rhu/fantasybasketball3/fantasy_2026/data/matchups/matchups_latest.csv'
    matchups_df = pd.read_csv(matchups_file)

    # Filter matchups - try by matchup_period (fantasy week) first, then by week (ESPN internal)
    if 'matchup_period' in matchups_df.columns:
        week_matchups = matchups_df[matchups_df['matchup_period'] == target_week]
        if len(week_matchups) == 0:
            week_matchups = matchups_df[matchups_df['week'] == target_espn_week] if target_espn_week else pd.DataFrame()
    else:
        week_matchups = matchups_df[matchups_df['week'] == target_week]

    # Check if we have box score data for this week
    if len(box_scores) > 0 and target_espn_week is not None:
        week_data = box_scores[box_scores['week'] == target_espn_week]
    elif len(box_scores) > 0:
        # Try filtering by fantasy week if column exists
        week_data = box_scores[box_scores['week'] == target_week]
    else:
        week_data = pd.DataFrame()

    if len(week_matchups) == 0 and len(week_data) == 0:
        print(f"\n❌ ERROR: No matchups found for Fantasy Week {target_week}")
        print(f"   - Not in matchups_latest.csv (checked matchup_period={target_week})")
        print(f"   - Not in box_scores (checked week={target_espn_week})")
        print(f"\n   Try running: python scripts/3_get_matchups.py --week {target_week}")
        return

    # Load roster for player positions
    roster = pd.read_csv('/Users/rhu/fantasybasketball3/fantasy_2026/data/roster_snapshots/roster_latest.csv')
    player_positions = roster[['player_name', 'position']].drop_duplicates()

    # Fit player models
    player_models = fit_player_models(historical, espn_proj)

    # Get matchup names and data for this week
    # Prefer box_scores (actual data) over matchups_df (scheduled matchups)
    if len(week_data) > 0:
        matchups = week_data['matchup'].unique()
        print(f"\n✅ Using box_scores_latest.csv for Week {target_week} matchups")
        # Create a dummy matchups_df from box_scores if matchups_df is empty
        if len(week_matchups) == 0:
            matchups_df = pd.DataFrame()  # Empty df, will be handled in get_matchup_players_from_box_scores
    else:
        # No box score data yet - construct matchup names from matchups_df
        matchups = []
        for _, row in week_matchups.iterrows():
            matchup_name = f"{row['home_team_name']} vs {row['away_team_name']}"
            matchups.append(matchup_name)
        print(f"\n✅ Using matchups_latest.csv for Week {target_week} matchups")

    print(f"\nFound {len(matchups)} Week {target_week} matchups:")
    for m in matchups:
        print(f"  - {m}")

    # Create output directory (clear any previous simulations)
    output_dir = Path('/Users/rhu/fantasybasketball3/fantasy_2026/fixed_simulations')
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
        print(f"\n🧹 Cleared previous simulations from {output_dir}/")
    output_dir.mkdir(exist_ok=True)

    all_summaries = []

    # Simulate each matchup
    for matchup_name in matchups:
        print(f"\n{'-'*80}")
        print(f"{matchup_name}")
        print(f"{'-'*80}")

        # Get players and games to simulate (use ESPN week for filtering box_scores)
        home_players, away_players, home_name, away_name, mode = get_matchup_players_from_box_scores(
            box_scores, matchup_name, matchups_df, target_espn_week, games_per_player, roster
        )

        mode_emoji = "📊" if mode == "mid-week" else "📜"
        print(f"  {mode_emoji} Mode: {mode.upper()}")
        print(f"  {home_name}: {len(home_players)} players, {sum(home_players.values())} games to simulate")
        print(f"  {away_name}: {len(away_players)} players, {sum(away_players.values())} games to simulate")

        if len(home_players) == 0 or len(away_players) == 0:
            print(f"  ⚠️  Skipping - no players found (need roster data for full projection)")
            continue

        # Run simulations
        print(f"  Simulating 500 matchups...")
        results_df, _ = simulate_matchup(
            home_players, away_players, player_models, mapping, player_positions, n_simulations=500
        )

        # Calculate summary
        team_a_wins = int((results_df['winner'] == 'A').sum())
        team_b_wins = int((results_df['winner'] == 'B').sum())
        ties = int((results_df['winner'] == 'TIE').sum())

        summary = {
            'matchup': matchup_name,
            'team_a_name': home_name,
            'team_b_name': away_name,
            'n_simulations': len(results_df),
            'team_a_wins': team_a_wins,
            'team_b_wins': team_b_wins,
            'ties': ties,
            'team_a_win_pct': float(team_a_wins / len(results_df)),
            'team_b_win_pct': float(team_b_wins / len(results_df)),
            'team_a_avg_cats_won': float(results_df['team_a_cats'].mean()),
            'team_b_avg_cats_won': float(results_df['team_b_cats'].mean()),
            'team_a_players': len(home_players),
            'team_b_players': len(away_players),
            'team_a_total_games': sum(home_players.values()),
            'team_b_total_games': sum(away_players.values())
        }

        all_summaries.append(summary)

        # Save results
        safe_name = matchup_name.replace(' ', '_').replace('vs', 'vs')
        matchup_dir = output_dir / safe_name
        matchup_dir.mkdir(exist_ok=True)

        results_df.to_csv(matchup_dir / 'all_simulations.csv', index=False)
        with open(matchup_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✅ Results: {home_name} {summary['team_a_win_pct']:.1%} | {away_name} {summary['team_b_win_pct']:.1%}")

    # Save overall summary
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(output_dir / 'all_matchups_summary.csv', index=False)

    print(f"\n{'='*80}")
    print(f"SIMULATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Simulated {len(all_summaries)} matchups for Week {target_week}")
    print(f"Output: {output_dir}/")

    # Only run comparison with actual results if we have actual data for this week
    week_data = box_scores[box_scores['week'] == target_espn_week] if len(box_scores) > 0 else pd.DataFrame()
    if len(week_data) > 0:
        print(f"\n{' COMPARING WITH ACTUAL RESULTS ':-^80}")
        print(f"Note: Running analysis on Week {target_week} actual data")

        # Load analyze script and run comparison
        import subprocess
        subprocess.run([
            'python', 'analyze_actual_vs_simulated.py'
        ], cwd='/Users/rhu/fantasybasketball3/fantasy_2026')
    else:
        print(f"\n{' SKIPPING ACTUAL VS SIMULATED COMPARISON ':-^80}")
        print(f"No actual game data available for Week {target_week} yet.")
        print(f"This is a forward-looking projection. Comparison will be available after games are played.")

    # Generate consolidated report
    print(f"\n{'='*80}")
    print("GENERATING CONSOLIDATED REPORT")
    print(f"{'='*80}")

    # Create timestamped output directory
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_report_dir = Path('/Users/rhu/fantasybasketball3/fantasy_2026/simulation_reports')
    report_output_dir = base_report_dir / f'week{target_week}_report_{run_timestamp}'
    report_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReport Directory: {report_output_dir}/")

    # Load simulation data from fixed_simulations
    print("\nLoading simulation data...")
    summary_df, matchup_data = load_all_simulation_data()
    print(f"  Loaded {len(matchup_data)} matchups")

    print("\nLoading actual results...")
    actual_results = load_actual_results(target_espn_week)

    # Save metadata
    print("\nSaving run metadata...")
    save_metadata(report_output_dir, summary_df, run_timestamp, target_week)

    # Create overview visualizations
    print("\nCreating overview visualizations...")
    create_overview_visualizations(summary_df, actual_results, report_output_dir, target_week)

    # Create individual matchup visualizations
    print("\nCreating individual matchup visualizations...")
    for matchup_name, data in matchup_data.items():
        print(f"  Processing: {matchup_name}")
        create_matchup_detail_viz(matchup_name, data, report_output_dir)

    # Generate markdown report
    print("\nGenerating consolidated markdown report...")
    generate_markdown_report(summary_df, matchup_data, actual_results, report_output_dir, run_timestamp, target_week)

    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE!")
    print("="*80)
    print(f"\n📁 Output Directory: {report_output_dir}/")
    print(f"\n📊 Generated Files:")
    print(f"   - CONSOLIDATED_REPORT.md      (Main report with all visualizations)")
    print(f"   - RUN_METADATA.json           (Run configuration and summary)")
    print(f"   - overview_visualizations.png (5-panel dashboard)")
    print(f"   - *_distributions.png         ({len(matchup_data)} individual matchup plots)")
    print(f"\n🎯 Total Files: {len(list(report_output_dir.glob('*.png'))) + 2}")
    print(f"📈 Open the report: open {report_output_dir}/CONSOLIDATED_REPORT.md")


if __name__ == '__main__':
    main()
