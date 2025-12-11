#!/usr/bin/env python3
"""
Get matchup data from ESPN Fantasy API.

Extracts:
- Matchups for specified week (or current week)
- Team lineups (who's playing vs who)
- Projected and actual scores
- Box scores if games have started

Usage:
    # Get current week
    python 3_get_matchups.py

    # Get specific historical week
    python 3_get_matchups.py --week 6

    # Get range of weeks (for backtesting)
    python 3_get_matchups.py --week 1 --end-week 10
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

# Add espn-api to path
parent_dir = Path(__file__).parent.parent.parent
espn_api_path = parent_dir / "espn-api"
sys.path.insert(0, str(espn_api_path))

from espn_api.basketball import League


def load_config():
    """Load league configuration."""
    config_path = Path(__file__).parent.parent / "config" / "league_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_week_matchups(league, week: int = None):
    """
    Get matchups for a specific week (or current week if not specified).

    Args:
        league: ESPN League object
        week: Week number to fetch (None = current week)

    Returns:
        List of matchup dictionaries
    """
    target_week = week if week is not None else league.current_week

    print(f"\n{'='*60}")
    print(f"WEEK {target_week} MATCHUPS")
    print(f"{'='*60}\n")

    # Fetch scoreboard for specific week
    matchups = league.scoreboard(matchupPeriod=target_week)
    matchup_data = []

    for i, matchup in enumerate(matchups, 1):
        print(f"Matchup {i}:")
        print(f"  {matchup.home_team.team_name} vs {matchup.away_team.team_name}")

        matchup_info = {
            'week': target_week,
            'matchup_period': target_week,
            'matchup_num': i,
            'home_team_id': matchup.home_team.team_id,
            'home_team_name': matchup.home_team.team_name,
            'home_team_abbrev': matchup.home_team.team_abbrev,
            'away_team_id': matchup.away_team.team_id,
            'away_team_name': matchup.away_team.team_name,
            'away_team_abbrev': matchup.away_team.team_abbrev,
        }

        # Check if it's a category or points league
        if hasattr(matchup, 'home_score'):
            # Points league
            matchup_info['scoring_type'] = 'points'
            matchup_info['home_score'] = matchup.home_score
            matchup_info['away_score'] = matchup.away_score
            matchup_info['home_projected'] = getattr(matchup, 'home_projected', None)
            matchup_info['away_projected'] = getattr(matchup, 'away_projected', None)

            print(f"  Score: {matchup.home_score} - {matchup.away_score}")
            if matchup_info['home_projected']:
                print(f"  Projected: {matchup_info['home_projected']} - {matchup_info['away_projected']}")

        elif hasattr(matchup, 'home_wins'):
            # Category league
            matchup_info['scoring_type'] = 'category'
            matchup_info['home_wins'] = matchup.home_wins
            matchup_info['home_losses'] = matchup.home_losses
            matchup_info['home_ties'] = matchup.home_ties
            matchup_info['away_wins'] = getattr(matchup, 'away_wins', None)
            matchup_info['away_losses'] = getattr(matchup, 'away_losses', None)
            matchup_info['away_ties'] = getattr(matchup, 'away_ties', None)

            print(f"  Category Score: {matchup_info['home_wins']}W-{matchup_info['home_ties']}T-{matchup_info['home_losses']}L")

        matchup_data.append(matchup_info)
        print()

    return matchup_data


def get_box_scores(league, week: int = None):
    """
    Get detailed box scores for a specific week (or current week).

    Args:
        league: ESPN League object
        week: Week number to fetch (None = current week)

    Returns:
        List of player performance data
    """
    target_week = week if week is not None else league.current_week

    print(f"\n{'='*60}")
    print(f"WEEK {target_week} BOX SCORES")
    print(f"{'='*60}\n")

    # Fetch box scores for specific week
    box_scores = league.box_scores(matchup_period=target_week)

    if not box_scores:
        print("No box scores available yet (week hasn't started)")
        return []

    all_player_stats = []

    for box in box_scores:
        matchup_info = f"{box.home_team.team_name} vs {box.away_team.team_name}"
        print(f"\n{matchup_info}")
        print("-" * len(matchup_info))

        # Process home team lineup
        for player in box.home_lineup:
            player_data = {
                'week': target_week,
                'matchup': matchup_info,
                'team_id': box.home_team.team_id,
                'team_name': box.home_team.team_name,
                'team_side': 'home',
                'player_name': player.name,
                'player_id': player.playerId,
                'position': player.position,
                'slot_position': player.slot_position,
                'pro_team': player.proTeam,
            }

            # Add stats if available
            if hasattr(player, 'points'):
                player_data['points'] = player.points
            if hasattr(player, 'projected_points'):
                player_data['projected_points'] = player.projected_points

            # Add nine-cat stats if available
            if hasattr(player, 'stats') and player.stats:
                for stat_name, stat_value in player.stats.items():
                    player_data[f'stat_{stat_name}'] = stat_value

            all_player_stats.append(player_data)

        # Process away team lineup
        for player in box.away_lineup:
            player_data = {
                'week': target_week,
                'matchup': matchup_info,
                'team_id': box.away_team.team_id,
                'team_name': box.away_team.team_name,
                'team_side': 'away',
                'player_name': player.name,
                'player_id': player.playerId,
                'position': player.position,
                'slot_position': player.slot_position,
                'pro_team': player.proTeam,
            }

            if hasattr(player, 'points'):
                player_data['points'] = player.points
            if hasattr(player, 'projected_points'):
                player_data['projected_points'] = player.projected_points

            if hasattr(player, 'stats') and player.stats:
                for stat_name, stat_value in player.stats.items():
                    player_data[f'stat_{stat_name}'] = stat_value

            all_player_stats.append(player_data)

        # Show summary
        if hasattr(box, 'home_score'):
            print(f"  Score: {box.home_score} - {box.away_score}")
            print(f"  Projected: {box.home_projected} - {box.away_projected}")

    return all_player_stats


def save_matchup_data(matchups, box_scores, output_dir, week: int = None, save_as_latest: bool = True):
    """
    Save matchup data to CSV files.

    Args:
        matchups: List of matchup dictionaries
        box_scores: List of box score dictionaries
        output_dir: Output directory path
        week: Week number (for filename)
        save_as_latest: Whether to also save as *_latest.csv
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    week_str = f"_week{week}" if week else ""

    # Save matchups
    if matchups:
        matchups_df = pd.DataFrame(matchups)
        matchups_file = output_path / f"matchups{week_str}_{timestamp}.csv"
        matchups_df.to_csv(matchups_file, index=False)
        print(f"\n✓ Saved matchups to: {matchups_file}")

        # Also save as latest (only if requested)
        if save_as_latest:
            latest_matchups = output_path / "matchups_latest.csv"
            matchups_df.to_csv(latest_matchups, index=False)
            print(f"✓ Saved as latest: {latest_matchups}")

    # Save box scores
    if box_scores:
        box_scores_df = pd.DataFrame(box_scores)
        box_scores_file = output_path / f"box_scores{week_str}_{timestamp}.csv"
        box_scores_df.to_csv(box_scores_file, index=False)
        print(f"\n✓ Saved box scores to: {box_scores_file}")

        # Also save as latest (only if requested)
        if save_as_latest:
            latest_box_scores = output_path / "box_scores_latest.csv"
            box_scores_df.to_csv(latest_box_scores, index=False)
            print(f"✓ Saved as latest: {latest_box_scores}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Fetch matchup and box score data from ESPN Fantasy API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Get current week data
    python 3_get_matchups.py

    # Get specific historical week
    python 3_get_matchups.py --week 6

    # Get range of weeks (for backtesting)
    python 3_get_matchups.py --week 1 --end-week 10

    # Get historical week without overwriting latest
    python 3_get_matchups.py --week 6 --no-latest
        """
    )
    parser.add_argument('--week', type=int, default=None,
                        help='Week number to fetch (default: current week)')
    parser.add_argument('--end-week', type=int, default=None,
                        help='End week for fetching a range (requires --week)')
    parser.add_argument('--no-latest', action='store_true',
                        help='Do not overwrite *_latest.csv files')
    args = parser.parse_args()

    print("="*60)
    print("ESPN FANTASY MATCHUP DATA EXTRACTION")
    print("="*60)

    # Load configuration
    config = load_config()
    league_id = config['league']['id']
    season = config['league']['season']
    output_dir = Path(__file__).parent.parent / "data" / "matchups"

    # Get ESPN credentials from environment
    espn_s2 = os.getenv('ESPN_S2')
    swid = os.getenv('ESPN_SWID')

    if not espn_s2 or not swid:
        print("\n✗ ERROR: ESPN credentials not found in environment variables")
        print("  Please set ESPN_S2 and ESPN_SWID")
        return 1

    # Connect to league
    print(f"\nConnecting to ESPN league {league_id}, season {season}...")

    try:
        league = League(
            league_id=league_id,
            year=season,
            espn_s2=espn_s2,
            swid=swid
        )
        print(f"✓ Connected successfully")
        print(f"  League: {config['league']['name']}")
        print(f"  Current Week: {league.current_week}")
        print(f"  Matchup Period: {league.currentMatchupPeriod}")

    except Exception as e:
        print(f"✗ Failed to connect to league: {e}")
        return 1

    # Determine weeks to fetch
    if args.week is not None:
        if args.end_week is not None:
            weeks_to_fetch = list(range(args.week, args.end_week + 1))
        else:
            weeks_to_fetch = [args.week]
        save_as_latest = not args.no_latest and len(weeks_to_fetch) == 1
    else:
        weeks_to_fetch = [None]  # None means current week
        save_as_latest = True

    # Fetch data for each week
    for week in weeks_to_fetch:
        week_label = week if week else league.current_week
        print(f"\n{'='*60}")
        print(f"FETCHING WEEK {week_label}")
        print(f"{'='*60}")

        # Get matchup data
        matchups = get_week_matchups(league, week)

        # Get box scores
        box_scores = get_box_scores(league, week)

        # Save data
        # Only save as latest for the last week in a range, or if fetching single week
        is_last_week = (week == weeks_to_fetch[-1])
        save_matchup_data(
            matchups, box_scores, output_dir,
            week=week_label,
            save_as_latest=save_as_latest and is_last_week
        )

    print("\n" + "="*60)
    print("✓ MATCHUP DATA EXTRACTION COMPLETE")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
