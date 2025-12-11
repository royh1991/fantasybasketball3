# CLAUDE.md - Fantasy Basketball Simulation System

## Project Purpose

This system simulates fantasy basketball matchups using Bayesian statistical models. It predicts matchup outcomes by:
1. Fetching real data from ESPN Fantasy Basketball API
2. Fitting player-level statistical models on historical NBA data
3. Running Monte Carlo simulations (500 iterations) to estimate win probabilities
4. Comparing predictions against actual results for validation

**Validated accuracy: 73% matchup prediction, 77% category-level prediction across 8 weeks.**

---

## Directory Structure

```
fantasybasketball3/
├── CLAUDE.md                    # This file
├── espn-api/                    # ESPN Fantasy API library (dependency)
├── data/
│   └── fantasy_basketball_clean2.csv  # ESPN player projections
└── fantasy_2026/
    ├── simulate_with_correct_data.py  # MAIN SCRIPT - run simulations
    ├── simulation_validation.py       # Validate predictions vs actuals
    ├── analyze_actual_vs_simulated.py # Compare simulation to real outcomes
    ├── run_all.py                     # Run all data collection scripts
    ├── scripts/                       # Data collection scripts (1-5)
    ├── projection_diagnostics/
    │   └── bayesian_projection_model.py  # Core Bayesian model
    ├── data/                          # All data files
    ├── fixed_simulations/             # Simulation output
    ├── simulation_reports/            # Generated reports
    └── config/                        # Configuration (ESPN credentials)
```

---

## Key Scripts

### 1. `fantasy_2026/simulate_with_correct_data.py` (MAIN)

**Purpose:** Simulate fantasy matchups and predict winners.

**Usage:**
```bash
cd /Users/rhu/fantasybasketball3/fantasy_2026

# Simulate current/latest week
python simulate_with_correct_data.py

# Simulate specific week
python simulate_with_correct_data.py --week 8

# Specify games per player (for projections)
python simulate_with_correct_data.py --week 9 --games-per-player 4
```

**Three Modes (auto-detected):**
- **HISTORICAL**: Week completed, uses actual games played from box scores
- **MID-WEEK**: Week in progress, combines actual + simulated remaining games
- **PROJECTION**: Week not started, simulates all games based on roster

**Output:**
- `fixed_simulations/{matchup_name}/summary.json` - Win probabilities
- `fixed_simulations/{matchup_name}/all_simulations.csv` - Raw simulation data
- `fixed_simulations/all_matchups_summary.csv` - Summary of all matchups
- `simulation_reports/week{N}_report_{timestamp}/` - Visual report

**Key Functions:**
- `load_data()` - Loads box scores, historical stats, player mappings
- `fit_player_models()` - Fits Bayesian models for each player
- `simulate_matchup()` - Runs 500 Monte Carlo simulations
- `create_consolidated_report()` - Generates visual report

---

### 2. `fantasy_2026/simulation_validation.py`

**Purpose:** Validate simulation accuracy across multiple weeks.

**Usage:**
```bash
# Validate weeks 1-8
python simulation_validation.py --weeks 1-8

# Skip re-fetching data (use cached)
python simulation_validation.py --weeks 1-8 --skip-fetch

# Skip re-running simulations
python simulation_validation.py --weeks 1-8 --skip-fetch --skip-simulations
```

**Output:**
- `simulation_reports/validation_report_{timestamp}/VALIDATION_REPORT.md`
- `validation_dashboard.png` - Overall accuracy visualization
- `category_calibration.png` - Per-category accuracy heatmap
- `weekly_details.png` - Week-by-week breakdown

**Analyzes:**
- Matchup-level accuracy (did we predict the winner?)
- Category-level accuracy (11 categories x weeks x matchups)
- Calibration (do 70% predictions win 70% of the time?)
- Z-score distribution (are predictions well-calibrated?)

---

### 3. `fantasy_2026/scripts/` (Data Collection)

Run in order, or use `run_all.py`:

| Script | Purpose | Output |
|--------|---------|--------|
| `1_extract_rosters.py` | Get current team rosters | `data/roster_snapshots/roster_latest.csv` |
| `2_collect_historical_data.py` | Fetch NBA historical stats | `data/historical_gamelogs/historical_gamelogs_latest.csv` |
| `3_get_matchups.py` | Get matchups and box scores | `data/matchups/box_scores_latest.csv` |
| `4_create_player_mapping.py` | Map ESPN names to NBA API | `data/mappings/player_mapping_latest.csv` |
| `5_daily_roster_history.py` | Track roster changes | `data/ownership_history/daily_rosters_latest.csv` |

**Fetch specific week:**
```bash
python scripts/3_get_matchups.py --week 6
```

---

### 4. `fantasy_2026/projection_diagnostics/bayesian_projection_model.py`

**Purpose:** Core statistical model for player projections.

**Model Architecture:**
- **Shooting percentages (FG%, FT%, 3P%)**: Beta-Binomial conjugate model
  - Prior: Position-specific league averages
  - Likelihood: Recent games (exponential decay weighting)
  - Posterior: Beta(α + makes, β + misses)

- **Counting stats (PTS, REB, AST, etc.)**: Normal-Normal conjugate model
  - Prior: Historical player mean with uncertainty
  - Posterior updates with current season data

**Key Class:** `BayesianProjectionModel`
```python
model = BayesianProjectionModel()
model.fit(historical_data, player_name)
simulated_game = model.simulate_game()  # Returns dict of stats
```

---

## Data Files

### Input Data (in `fantasy_2026/data/`)

| File | Description | Key Columns |
|------|-------------|-------------|
| `matchups/box_scores_latest.csv` | Player box scores by week | week, matchup, team_name, player_name, stat_0 |
| `matchups/matchups_latest.csv` | Weekly matchup definitions | week, home_team_name, away_team_name |
| `historical_gamelogs/historical_gamelogs_latest.csv` | NBA game logs 2019-2024 | PLAYER_NAME, GAME_DATE, FGM, FGA, PTS, REB, AST... |
| `mappings/player_mapping_latest.csv` | ESPN to NBA name mapping | espn_name, nba_api_name |
| `roster_snapshots/roster_latest.csv` | Current team rosters | team_name, player_name, injury_status |

### Output Data

| Location | Description |
|----------|-------------|
| `fixed_simulations/` | Raw simulation results (CSV, JSON) |
| `simulation_reports/` | Visual reports with PNG charts |

---

## Fantasy Categories (11-CAT)

The system simulates all 11 standard fantasy basketball categories:

| Category | Type | Winner |
|----------|------|--------|
| FG% | Percentage | Higher wins |
| FT% | Percentage | Higher wins |
| 3P% | Percentage | Higher wins |
| 3PM | Counting | Higher wins |
| PTS | Counting | Higher wins |
| REB | Counting | Higher wins |
| AST | Counting | Higher wins |
| STL | Counting | Higher wins |
| BLK | Counting | Higher wins |
| TO | Counting | **Lower wins** |
| DD | Counting | Higher wins |

**Matchup winner:** Team winning 6+ categories wins the matchup.

---

## ESPN API Authentication

ESPN API requires authentication cookies stored in environment variables:

```bash
export ESPN_S2="your_espn_s2_cookie"
export ESPN_SWID="your_swid_cookie"
```

**To get cookies:**
1. Log into ESPN Fantasy Basketball in browser
2. Open Developer Tools > Application > Cookies
3. Copy `espn_s2` and `SWID` values

**League Configuration:**
- League ID: 40204
- Season: 2026 (represents 2025-26 season)

---

## Common Workflows

### Weekly Simulation (Start of Week)
```bash
cd /Users/rhu/fantasybasketball3/fantasy_2026

# 1. Fetch latest matchup data
python scripts/3_get_matchups.py

# 2. Run simulation
python simulate_with_correct_data.py

# 3. View report
open simulation_reports/week*_report_*/CONSOLIDATED_REPORT.md
```

### Mid-Week Update
```bash
# Fetch updated box scores
python scripts/3_get_matchups.py

# Re-run simulation (auto-detects mid-week mode)
python simulate_with_correct_data.py
```

### Historical Validation
```bash
# Validate predictions for completed weeks
python simulation_validation.py --weeks 1-8 --skip-fetch
```

### Full Data Refresh
```bash
# Run all data collection scripts
python run_all.py
```

---

## Simulation Algorithm

```
For each matchup:
    1. Load home and away team rosters
    2. For each player, get games_to_simulate from box_scores or --games-per-player
    3. For 500 iterations:
        a. For each player on each team:
            - Simulate N games using Bayesian posterior
            - Aggregate stats (FGM, FGA, PTS, REB, AST, etc.)
        b. Calculate team totals
        c. Compare 11 categories, determine winner
    4. Output: win_probability = wins / 500
```

---

## Key Dependencies

**Python packages:**
- pandas, numpy - Data manipulation
- matplotlib - Visualization
- scipy - Statistical distributions
- nba_api - NBA statistics (for historical data)
- pyyaml - Configuration

**Local:**
- `espn-api/` - ESPN Fantasy API (included in repo)

---

## Troubleshooting

### "No box scores found for Week X"
```bash
# Fetch the week's data
python scripts/3_get_matchups.py --week X
```

### "Player not found in models"
- Check `data/mappings/player_mapping_latest.csv` for name mapping
- Rookies without historical data use replacement-level projections

### ESPN API Authentication Error
- Verify ESPN_S2 and ESPN_SWID environment variables are set
- Cookies may expire - refresh from browser

### Import Errors
- Ensure running from `fantasy_2026/` directory
- Check sys.path includes `/Users/rhu/fantasybasketball3`

---

## Performance Notes

- **Model fitting:** ~30 seconds for 500+ players
- **Single matchup simulation:** ~5-10 seconds (500 iterations)
- **Full week (7 matchups):** ~1-2 minutes
- **Full validation (8 weeks):** ~10 minutes

---

## File Naming Conventions

- `*_latest.csv` - Most recent version, updated by scripts
- `box_scores_week{N}_{timestamp}.csv` - Week-specific snapshots
- `week{N}_report_{timestamp}/` - Timestamped reports (never overwritten)
