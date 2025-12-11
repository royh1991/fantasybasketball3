#!/usr/bin/env python3
"""
Proper Bayesian Projection Model

Key principles:
1. Priors encode what we know BEFORE seeing current season data
2. Prior strength reflects uncertainty (not arbitrary decay)
3. Evolution variance determines how much talent can change over time
4. Posterior naturally balances prior and likelihood based on relative precision
5. FG% is DERIVED from 2P% and 3P%, not sampled directly

The Bayesian update for Normal-Normal conjugate:
    Prior: θ ~ N(μ_0, σ²_0)
    Likelihood: x̄ | θ ~ N(θ, σ²/n)
    Posterior: θ | x̄ ~ N(μ_n, σ²_n)

    where:
        precision_0 = 1/σ²_0
        precision_data = n/σ²

        μ_n = (precision_0 * μ_0 + precision_data * x̄) / (precision_0 + precision_data)
        σ²_n = 1 / (precision_0 + precision_data)

For Beta-Binomial (shooting percentages):
    Prior: θ ~ Beta(α_0, β_0)
    Likelihood: k successes in n trials
    Posterior: θ ~ Beta(α_0 + k, β_0 + n - k)
"""

import sys
sys.path.insert(0, '/Users/rhu/fantasybasketball3')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BetaPrior:
    """Beta distribution prior for shooting percentages."""
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b)**2 * (a + b + 1))

    @property
    def effective_sample_size(self) -> float:
        """How many observations is this prior worth?"""
        return self.alpha + self.beta

    def update(self, successes: int, trials: int) -> 'BetaPrior':
        """Bayesian update with new observations."""
        return BetaPrior(
            alpha=self.alpha + successes,
            beta=self.beta + (trials - successes)
        )

    def sample(self) -> float:
        """Sample from the distribution."""
        return np.random.beta(self.alpha, self.beta)

    def inflate_variance(self, factor: float) -> 'BetaPrior':
        """
        Inflate variance while keeping mean constant.
        This is how we account for evolution/drift over time.

        For Beta(α, β), to inflate variance by factor k while keeping mean:
        - New effective sample size = old / k
        - α_new = α / k, β_new = β / k
        """
        return BetaPrior(
            alpha=self.alpha / factor,
            beta=self.beta / factor
        )


@dataclass
class NormalPrior:
    """Normal distribution prior for counting stats."""
    mean: float
    variance: float

    @property
    def precision(self) -> float:
        return 1.0 / self.variance if self.variance > 0 else 1e6

    def update(self, sample_mean: float, sample_var: float, n: int) -> 'NormalPrior':
        """
        Bayesian update with new observations.

        Assumes known variance (use sample variance as plug-in estimate).
        """
        if n == 0 or sample_var <= 0:
            return self

        precision_0 = self.precision
        precision_data = n / sample_var

        precision_n = precision_0 + precision_data
        mean_n = (precision_0 * self.mean + precision_data * sample_mean) / precision_n

        return NormalPrior(
            mean=mean_n,
            variance=1.0 / precision_n
        )

    def sample(self) -> float:
        """Sample from the distribution."""
        return np.random.normal(self.mean, np.sqrt(self.variance))

    def inflate_variance(self, factor: float) -> 'NormalPrior':
        """Inflate variance while keeping mean constant (for evolution)."""
        return NormalPrior(
            mean=self.mean,
            variance=self.variance * factor
        )


# Position-based prior parameters (empirical Bayes from league data)
# These represent "what we'd expect from an average player at this position"
POSITION_PRIORS = {
    'C': {
        # Centers: High 2P%, low 3P volume, lower FT%
        '2P_PCT': BetaPrior(alpha=29, beta=21),      # ~58% mean, moderate confidence
        '3P_PCT': BetaPrior(alpha=5, beta=10),       # ~33% mean, LOW confidence (many don't shoot)
        'FT_PCT': BetaPrior(alpha=14, beta=6),       # ~70% mean
        '3PA_RATE': 0.12,  # Only 12% of shots are 3s
        'is_shooter': False,  # Most centers aren't 3P shooters
    },
    'PF': {
        '2P_PCT': BetaPrior(alpha=25, beta=22),      # ~53% mean
        '3P_PCT': BetaPrior(alpha=7, beta=13),       # ~35% mean
        'FT_PCT': BetaPrior(alpha=15, beta=5),       # ~75% mean
        '3PA_RATE': 0.30,
        'is_shooter': True,
    },
    'SF': {
        '2P_PCT': BetaPrior(alpha=24, beta=24),      # ~50% mean
        '3P_PCT': BetaPrior(alpha=7, beta=13),       # ~35% mean
        'FT_PCT': BetaPrior(alpha=16, beta=4),       # ~80% mean
        '3PA_RATE': 0.40,
        'is_shooter': True,
    },
    'SG': {
        '2P_PCT': BetaPrior(alpha=22, beta=24),      # ~48% mean
        '3P_PCT': BetaPrior(alpha=7, beta=12),       # ~37% mean
        'FT_PCT': BetaPrior(alpha=17, beta=3),       # ~85% mean
        '3PA_RATE': 0.48,
        'is_shooter': True,
    },
    'PG': {
        '2P_PCT': BetaPrior(alpha=21, beta=24),      # ~47% mean
        '3P_PCT': BetaPrior(alpha=7, beta=12),       # ~37% mean
        'FT_PCT': BetaPrior(alpha=17, beta=3),       # ~85% mean
        '3PA_RATE': 0.50,
        'is_shooter': True,
    },
}


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse dates in format 'OCT 25, 2023' to datetime."""
    try:
        return datetime.strptime(date_str.strip(), "%b %d, %Y")
    except:
        try:
            return datetime.strptime(date_str.strip(), "%B %d, %Y")
        except:
            return None


class BayesianProjectionModel:
    """
    Proper Bayesian projection model for fantasy basketball.

    Key features:
    1. Position-based priors (centers shoot differently than guards)
    2. Separate 2P% and 3P% modeling (FG% is derived)
    3. Evolution variance accounts for talent drift over time
    4. Proper conjugate Bayesian updates
    """

    def __init__(self, evolution_years: float = 1.0, position: str = 'SF'):
        """
        Args:
            evolution_years: How much variance inflation per year of separation
                           between historical data and now. Higher = more adaptive.
            position: Player position for prior selection
        """
        self.evolution_years = evolution_years
        self.position = position
        self.priors = POSITION_PRIORS.get(position, POSITION_PRIORS['SF'])

        # Posteriors (will be set by fit_player)
        self.post_2p_pct: Optional[BetaPrior] = None
        self.post_3p_pct: Optional[BetaPrior] = None
        self.post_ft_pct: Optional[BetaPrior] = None

        # Counting stat posteriors
        self.post_fga: Optional[NormalPrior] = None
        self.post_3pa_rate: Optional[float] = None  # Fraction of FGA that are 3PA
        self.post_fta: Optional[NormalPrior] = None
        self.post_reb: Optional[NormalPrior] = None
        self.post_ast: Optional[NormalPrior] = None
        self.post_stl: Optional[NormalPrior] = None
        self.post_blk: Optional[NormalPrior] = None
        self.post_tov: Optional[NormalPrior] = None

        # Game-to-game variance (observation noise)
        self.obs_var: Dict[str, float] = {}

        # For DD calculation
        self.dd_rate: float = 0.0

    def fit_player(self, historical_data: pd.DataFrame, player_name: str,
                   position: Optional[str] = None) -> bool:
        """
        Fit the model for a specific player.

        The Bayesian approach:
        1. Start with position-based priors
        2. Update with historical data (inflated variance based on age)
        3. Update with current season data (full precision)
        """
        if position:
            self.position = position
            self.priors = POSITION_PRIORS.get(position, POSITION_PRIORS['SF'])

        # Get player data
        player_data = historical_data[
            historical_data['PLAYER_NAME'].str.lower() == player_name.lower()
        ].copy()

        if len(player_data) == 0:
            return False

        # Parse dates
        player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
        player_data = player_data.dropna(subset=['parsed_date'])
        player_data = player_data.sort_values('parsed_date')

        if len(player_data) < 5:
            return False

        # Split into historical (before this season) and current
        cutoff_date = datetime(2025, 10, 1)
        historical = player_data[player_data['parsed_date'] < cutoff_date]
        current = player_data[player_data['parsed_date'] >= cutoff_date]

        # For rookies, use current season as both prior builder and update
        if len(historical) < 10 and len(current) >= 5:
            # Rookie: split current season
            # Use first half to build prior, second half to update
            n_current = len(current)
            historical = current.head(n_current // 2)
            current = current.tail(n_current - n_current // 2)

        # Step 1: Build prior from historical data
        self._build_prior_from_historical(historical)

        # Step 2: Update with current season data
        if len(current) > 0:
            self._update_with_current_season(current)

        # Calculate DD rate from all data
        self._calculate_dd_rate(player_data)

        return True

    def _build_prior_from_historical(self, historical: pd.DataFrame):
        """
        Build informative prior from historical data.

        KEY INSIGHT: Only use LAST SEASON's data as the prior.

        Why? Because:
        1. LeBron at 39 is different from LeBron at 36
        2. A player's role changes year to year
        3. We want the prior to reflect "what we expected BEFORE this season"

        The prior should answer: "Based on last season, what do we expect?"
        Then current season data updates that prior.

        For shooting percentages: Use position prior + last season data
        For counting stats: Use last season averages directly
        """
        # Only use games from LAST season (2024-25) for the prior
        # Current season (2025-26) will be used in the update step
        last_season_start = datetime(2024, 10, 1)
        last_season_end = datetime(2025, 6, 30)

        last_season = historical[
            (historical['parsed_date'] >= last_season_start) &
            (historical['parsed_date'] <= last_season_end)
        ]

        # For players without last season data, use older data but with more uncertainty
        if len(last_season) < 10:
            # Use all historical but with high uncertainty
            last_season = historical.tail(82)  # At most one season worth
            uncertainty_multiplier = 2.0  # Double the variance
        else:
            uncertainty_multiplier = 1.0

        if len(last_season) == 0:
            # No historical data - use position priors directly
            self.post_2p_pct = self.priors['2P_PCT']
            self.post_3p_pct = self.priors['3P_PCT']
            self.post_ft_pct = self.priors['FT_PCT']
            self.post_3pa_rate = self.priors['3PA_RATE']

            # Use weak priors for counting stats
            self.post_fga = NormalPrior(mean=10.0, variance=25.0)
            self.post_fta = NormalPrior(mean=3.0, variance=9.0)
            self.post_reb = NormalPrior(mean=5.0, variance=16.0)
            self.post_ast = NormalPrior(mean=3.0, variance=9.0)
            self.post_stl = NormalPrior(mean=1.0, variance=1.0)
            self.post_blk = NormalPrior(mean=0.5, variance=0.5)
            self.post_tov = NormalPrior(mean=1.5, variance=2.0)
            return

        n_games = len(last_season)

        # KEY TUNING PARAMETER:
        # effective_n controls prior strength
        # - With effective_n=60 and 20 current games: prior gets 75% weight (too much!)
        # - With effective_n=30 and 20 current games: prior gets 60% weight
        # - With effective_n=20 and 20 current games: prior gets 50% weight (equal!)
        #
        # For veterans like LeBron declining fast, we want current season to dominate.
        # For breakout players, we want current season to be believed.
        #
        # Solution: Use effective_n = 25-30, which gives:
        # - Prior: ~55-60% weight
        # - Current season (20 games): ~40-45% weight
        #
        # This is still Bayesian (prior matters) but responsive to changes.
        effective_n = min(n_games, 30)

        # --- Shooting percentages (Beta-Binomial) ---
        # Use last season data to build prior, then current season updates

        # Scale factor: we want last_season to count as effective_n games
        sample_scale = effective_n / max(n_games, 1)

        # 2P%: Calculate actual 2-point makes and attempts from LAST SEASON
        fg2m_raw = (last_season['FGM'] - last_season['FG3M']).sum()
        fg2a_raw = (last_season['FGA'] - last_season['FG3A']).sum()

        if fg2a_raw > 0:
            # Scale to effective sample size
            fg2a = int(fg2a_raw * sample_scale)
            fg2a = max(fg2a, 1)
            fg2m = int(fg2a * (fg2m_raw / fg2a_raw))

            self.post_2p_pct = self.priors['2P_PCT'].update(fg2m, fg2a)
        else:
            self.post_2p_pct = self.priors['2P_PCT']

        # 3P%: Only for players who actually shoot 3s
        fg3m_raw = last_season['FG3M'].sum()
        fg3a_raw = last_season['FG3A'].sum()

        if fg3a_raw >= 30:  # Meaningful 3P sample for one season
            fg3a = int(fg3a_raw * sample_scale)
            fg3a = max(fg3a, 1)
            fg3m = int(fg3a * (fg3m_raw / fg3a_raw))

            self.post_3p_pct = self.priors['3P_PCT'].update(fg3m, fg3a)
        else:
            # Not a 3P shooter - use very weak prior
            self.post_3p_pct = BetaPrior(alpha=2, beta=4)  # ~33% but very uncertain

        # FT%
        ftm_raw = last_season['FTM'].sum()
        fta_raw = last_season['FTA'].sum()

        if fta_raw > 0:
            fta = int(fta_raw * sample_scale)
            fta = max(fta, 1)
            ftm = int(fta * (ftm_raw / fta_raw))

            self.post_ft_pct = self.priors['FT_PCT'].update(ftm, fta)
        else:
            self.post_ft_pct = self.priors['FT_PCT']

        # 3PA rate (what fraction of shots are 3s)
        total_fga = last_season['FGA'].sum()
        if total_fga > 0:
            self.post_3pa_rate = fg3a_raw / total_fga
        else:
            self.post_3pa_rate = self.priors['3PA_RATE']

        # --- Counting stats (Normal-Normal) ---
        # Use last season means with appropriate uncertainty
        #
        # KEY INSIGHT: Counting stats are ROLE-DEPENDENT, not skill-dependent
        # - FGA depends on minutes, offensive load, team context
        # - REB depends on position, pace, team rebounding scheme
        # - AST depends on ball-handling responsibility
        #
        # These can change dramatically with:
        # - Load management (LeBron)
        # - Role expansion (breakout players)
        # - Team changes (trades)
        # - Injury to teammates
        #
        # Therefore, use a WEAKER prior for counting stats than shooting percentages
        #
        # With n_counting=10 and 20 current games: prior gets ~33%, current gets ~67%
        # With n_counting=10 and 7 current games: prior gets ~59%, current gets ~41%
        effective_n_counting = min(n_games, 10)  # Only 10 effective games for counting stats

        # FGA - use last season mean directly
        fga_mean = last_season['FGA'].mean()
        fga_var = last_season['FGA'].var()
        self.obs_var['FGA'] = fga_var if fga_var > 0 else 1.0

        # Prior from last season (not position average)
        # Variance scaled by uncertainty_multiplier and effective_n_counting
        prior_var = (self.obs_var['FGA'] / effective_n_counting) * uncertainty_multiplier
        self.post_fga = NormalPrior(mean=fga_mean, variance=prior_var)

        # FTA
        fta_mean = last_season['FTA'].mean()
        fta_var = last_season['FTA'].var()
        self.obs_var['FTA'] = fta_var if fta_var > 0 else 1.0
        prior_var = (self.obs_var['FTA'] / effective_n_counting) * uncertainty_multiplier
        self.post_fta = NormalPrior(mean=fta_mean, variance=prior_var)

        # REB
        reb_mean = last_season['REB'].mean()
        reb_var = last_season['REB'].var()
        self.obs_var['REB'] = reb_var if reb_var > 0 else 1.0
        prior_var = (self.obs_var['REB'] / effective_n_counting) * uncertainty_multiplier
        self.post_reb = NormalPrior(mean=reb_mean, variance=prior_var)

        # AST
        ast_mean = last_season['AST'].mean()
        ast_var = last_season['AST'].var()
        self.obs_var['AST'] = ast_var if ast_var > 0 else 1.0
        prior_var = (self.obs_var['AST'] / effective_n_counting) * uncertainty_multiplier
        self.post_ast = NormalPrior(mean=ast_mean, variance=prior_var)

        # STL
        stl_mean = last_season['STL'].mean()
        stl_var = last_season['STL'].var()
        self.obs_var['STL'] = stl_var if stl_var > 0 else 0.5
        prior_var = (self.obs_var['STL'] / effective_n_counting) * uncertainty_multiplier
        self.post_stl = NormalPrior(mean=stl_mean, variance=prior_var)

        # BLK
        blk_mean = last_season['BLK'].mean()
        blk_var = last_season['BLK'].var()
        self.obs_var['BLK'] = blk_var if blk_var > 0 else 0.5
        prior_var = (self.obs_var['BLK'] / effective_n_counting) * uncertainty_multiplier
        self.post_blk = NormalPrior(mean=blk_mean, variance=prior_var)

        # TOV
        tov_mean = last_season['TOV'].mean()
        tov_var = last_season['TOV'].var()
        self.obs_var['TOV'] = tov_var if tov_var > 0 else 1.0
        prior_var = (self.obs_var['TOV'] / effective_n_counting) * uncertainty_multiplier
        self.post_tov = NormalPrior(mean=tov_mean, variance=prior_var)

    def _update_with_current_season(self, current: pd.DataFrame):
        """
        Update posteriors with current season data.

        This is fresh data, so no variance inflation needed.
        The relative precision of prior vs current determines how much we shift.
        """
        if len(current) == 0:
            return

        n_games = len(current)

        # --- Shooting percentages ---

        # 2P%
        fg2m = (current['FGM'] - current['FG3M']).sum()
        fg2a = (current['FGA'] - current['FG3A']).sum()

        if fg2a > 0:
            self.post_2p_pct = self.post_2p_pct.update(int(fg2m), int(fg2a))

        # 3P%
        fg3m = current['FG3M'].sum()
        fg3a = current['FG3A'].sum()

        if fg3a > 10:  # Need some volume
            self.post_3p_pct = self.post_3p_pct.update(int(fg3m), int(fg3a))

        # FT%
        ftm = current['FTM'].sum()
        fta = current['FTA'].sum()

        if fta > 0:
            self.post_ft_pct = self.post_ft_pct.update(int(ftm), int(fta))

        # Update 3PA rate (blend historical and current)
        total_fga = current['FGA'].sum()
        if total_fga > 0:
            current_3pa_rate = fg3a / total_fga
            # Weighted average, giving current season 2x weight per game
            historical_weight = self.post_3pa_rate * 50  # Assume ~50 games historical equivalent
            current_weight = current_3pa_rate * n_games * 2
            self.post_3pa_rate = (historical_weight + current_weight) / (50 + n_games * 2)

        # --- Counting stats ---

        # FGA
        fga_mean = current['FGA'].mean()
        fga_var = current['FGA'].var()
        if fga_var > 0:
            self.post_fga = self.post_fga.update(fga_mean, fga_var, n_games)

        # FTA
        fta_mean = current['FTA'].mean()
        fta_var = current['FTA'].var()
        if fta_var > 0:
            self.post_fta = self.post_fta.update(fta_mean, fta_var, n_games)

        # REB
        reb_mean = current['REB'].mean()
        reb_var = current['REB'].var()
        if reb_var > 0:
            self.post_reb = self.post_reb.update(reb_mean, reb_var, n_games)

        # AST
        ast_mean = current['AST'].mean()
        ast_var = current['AST'].var()
        if ast_var > 0:
            self.post_ast = self.post_ast.update(ast_mean, ast_var, n_games)

        # STL
        stl_mean = current['STL'].mean()
        stl_var = current['STL'].var()
        if stl_var > 0:
            self.post_stl = self.post_stl.update(stl_mean, stl_var, n_games)

        # BLK
        blk_mean = current['BLK'].mean()
        blk_var = current['BLK'].var()
        if blk_var > 0:
            self.post_blk = self.post_blk.update(blk_mean, blk_var, n_games)

        # TOV
        tov_mean = current['TOV'].mean()
        tov_var = current['TOV'].var()
        if tov_var > 0:
            self.post_tov = self.post_tov.update(tov_mean, tov_var, n_games)

    def _calculate_dd_rate(self, all_data: pd.DataFrame):
        """Calculate historical double-double rate."""
        if len(all_data) == 0:
            self.dd_rate = 0.05
            return

        dd_count = 0
        for _, row in all_data.iterrows():
            dd_cats = [row['PTS'], row['REB'], row['AST'], row['STL'], row['BLK']]
            if sum(s >= 10 for s in dd_cats) >= 2:
                dd_count += 1

        # Bayesian estimate with weak prior (Beta(1,19) = 5% prior mean)
        self.dd_rate = (dd_count + 1) / (len(all_data) + 20)

    def simulate_game(self) -> Dict:
        """
        Simulate a single game.

        Key: FG% is DERIVED from 2P% and 3P%, not sampled directly.
        """
        stats = {}

        # Step 1: Sample attempt volumes
        fga = max(1, int(round(max(0, self.post_fga.sample()))))
        fta = max(0, int(round(max(0, self.post_fta.sample()))))

        # Determine 3PA based on rate
        fg3a = int(round(fga * self.post_3pa_rate))
        fg3a = max(0, min(fg3a, fga))  # Can't exceed FGA
        fg2a = fga - fg3a

        stats['FGA'] = fga
        stats['FG3A'] = fg3a
        stats['FTA'] = fta

        # Step 2: Sample shooting percentages and determine makes

        # 3P shooting
        fg3_pct = self.post_3p_pct.sample()
        fg3_pct = max(0.0, min(1.0, fg3_pct))
        fg3m = np.random.binomial(fg3a, fg3_pct)
        stats['FG3M'] = fg3m

        # 2P shooting
        fg2_pct = self.post_2p_pct.sample()
        fg2_pct = max(0.0, min(1.0, fg2_pct))
        fg2m = np.random.binomial(fg2a, fg2_pct)

        # Total FG (derived, not sampled!)
        stats['FGM'] = fg2m + fg3m

        # FT shooting
        ft_pct = self.post_ft_pct.sample()
        ft_pct = max(0.0, min(1.0, ft_pct))
        stats['FTM'] = np.random.binomial(fta, ft_pct)

        # Step 3: Calculate points (derived!)
        stats['PTS'] = 3 * fg3m + 2 * fg2m + stats['FTM']

        # Step 4: Sample counting stats
        stats['REB'] = max(0, int(round(self.post_reb.sample())))
        stats['AST'] = max(0, int(round(self.post_ast.sample())))
        stats['STL'] = max(0, int(round(self.post_stl.sample())))
        stats['BLK'] = max(0, int(round(self.post_blk.sample())))
        stats['TOV'] = max(0, int(round(self.post_tov.sample())))

        # Step 5: Double-double (based on simulated stats + historical rate)
        dd_cats = [stats['PTS'], stats['REB'], stats['AST'], stats['STL'], stats['BLK']]
        cats_at_10 = sum(s >= 10 for s in dd_cats)

        if cats_at_10 >= 2:
            stats['DD'] = 1
        elif cats_at_10 == 1 and any(8 <= s < 10 for s in dd_cats):
            # Close to DD - use historical rate
            stats['DD'] = 1 if np.random.random() < self.dd_rate else 0
        else:
            stats['DD'] = 0

        return stats

    def get_posterior_summary(self) -> Dict:
        """Return summary of posterior distributions."""
        return {
            '2P_PCT': {'mean': self.post_2p_pct.mean, 'ess': self.post_2p_pct.effective_sample_size},
            '3P_PCT': {'mean': self.post_3p_pct.mean, 'ess': self.post_3p_pct.effective_sample_size},
            'FT_PCT': {'mean': self.post_ft_pct.mean, 'ess': self.post_ft_pct.effective_sample_size},
            'FGA': {'mean': self.post_fga.mean, 'var': self.post_fga.variance},
            'FTA': {'mean': self.post_fta.mean, 'var': self.post_fta.variance},
            '3PA_RATE': self.post_3pa_rate,
            'REB': {'mean': self.post_reb.mean, 'var': self.post_reb.variance},
            'AST': {'mean': self.post_ast.mean, 'var': self.post_ast.variance},
            'STL': {'mean': self.post_stl.mean, 'var': self.post_stl.variance},
            'BLK': {'mean': self.post_blk.mean, 'var': self.post_blk.variance},
            'TOV': {'mean': self.post_tov.mean, 'var': self.post_tov.variance},
            'DD_RATE': self.dd_rate,
        }

    def fit_replacement_level(self, position: str = 'SF'):
        """
        Create a replacement-level player model based on position.

        Uses conservative estimates for deep-roster players at each position.
        This is used for rookies with no historical data.
        """
        # Replacement level stats by position (per-game averages)
        replacement_stats = {
            'PG': {
                'FGA': 7.0, 'FTA': 2.0, 'FG3A': 2.5,
                'REB': 3.0, 'AST': 3.5, 'STL': 0.8, 'BLK': 0.2, 'TOV': 1.5,
                '2P_PCT': 0.43, '3P_PCT': 0.34, 'FT_PCT': 0.75,
                '3PA_RATE': 0.36,
            },
            'SG': {
                'FGA': 7.5, 'FTA': 2.0, 'FG3A': 2.5,
                'REB': 2.5, 'AST': 2.0, 'STL': 0.7, 'BLK': 0.2, 'TOV': 1.2,
                '2P_PCT': 0.44, '3P_PCT': 0.35, 'FT_PCT': 0.78,
                '3PA_RATE': 0.33,
            },
            'SF': {
                'FGA': 7.5, 'FTA': 2.0, 'FG3A': 2.0,
                'REB': 4.0, 'AST': 1.5, 'STL': 0.7, 'BLK': 0.4, 'TOV': 1.0,
                '2P_PCT': 0.45, '3P_PCT': 0.35, 'FT_PCT': 0.76,
                '3PA_RATE': 0.27,
            },
            'PF': {
                'FGA': 7.0, 'FTA': 2.5, 'FG3A': 1.5,
                'REB': 5.5, 'AST': 1.5, 'STL': 0.6, 'BLK': 0.6, 'TOV': 1.0,
                '2P_PCT': 0.50, '3P_PCT': 0.33, 'FT_PCT': 0.74,
                '3PA_RATE': 0.21,
            },
            'C': {
                'FGA': 6.0, 'FTA': 2.5, 'FG3A': 0.5,
                'REB': 6.0, 'AST': 1.0, 'STL': 0.5, 'BLK': 0.8, 'TOV': 1.2,
                '2P_PCT': 0.55, '3P_PCT': 0.30, 'FT_PCT': 0.68,
                '3PA_RATE': 0.08,
            }
        }

        stats = replacement_stats.get(position, replacement_stats['SF'])

        # Set shooting percentages with high variance (weak prior)
        self.post_2p_pct = BetaPrior(alpha=stats['2P_PCT'] * 10, beta=(1 - stats['2P_PCT']) * 10)
        self.post_3p_pct = BetaPrior(alpha=stats['3P_PCT'] * 10, beta=(1 - stats['3P_PCT']) * 10)
        self.post_ft_pct = BetaPrior(alpha=stats['FT_PCT'] * 10, beta=(1 - stats['FT_PCT']) * 10)

        # Set 3PA rate
        self.post_3pa_rate = stats['3PA_RATE']

        # Set counting stats with high variance
        high_var_multiplier = 2.0  # Higher uncertainty for replacement level

        self.post_fga = NormalPrior(mean=stats['FGA'], variance=(stats['FGA'] * 0.4) ** 2 * high_var_multiplier)
        self.post_fta = NormalPrior(mean=stats['FTA'], variance=(stats['FTA'] * 0.5) ** 2 * high_var_multiplier)
        self.post_reb = NormalPrior(mean=stats['REB'], variance=(stats['REB'] * 0.4) ** 2 * high_var_multiplier)
        self.post_ast = NormalPrior(mean=stats['AST'], variance=(stats['AST'] * 0.5) ** 2 * high_var_multiplier)
        self.post_stl = NormalPrior(mean=stats['STL'], variance=(stats['STL'] * 0.6) ** 2 * high_var_multiplier)
        self.post_blk = NormalPrior(mean=stats['BLK'], variance=(stats['BLK'] * 0.6) ** 2 * high_var_multiplier)
        self.post_tov = NormalPrior(mean=stats['TOV'], variance=(stats['TOV'] * 0.5) ** 2 * high_var_multiplier)

        # Set observation variances
        self.obs_var = {
            'FGA': (stats['FGA'] * 0.4) ** 2,
            'FTA': (stats['FTA'] * 0.5) ** 2,
            'REB': (stats['REB'] * 0.4) ** 2,
            'AST': (stats['AST'] * 0.5) ** 2,
            'STL': (stats['STL'] * 0.6) ** 2,
            'BLK': (stats['BLK'] * 0.6) ** 2,
            'TOV': (stats['TOV'] * 0.5) ** 2,
        }

        # Replacement level players rarely get double-doubles
        self.dd_rate = 0.02

        return True


def test_model_on_players(test_all=True):
    """Test the new model on players."""

    # Load data
    base_path = Path('/Users/rhu/fantasybasketball3/fantasy_2026/data')
    historical = pd.read_csv(base_path / 'historical_gamelogs' / 'historical_gamelogs_latest.csv')
    mapping = pd.read_csv(base_path / 'mappings' / 'player_mapping_latest.csv')
    roster = pd.read_csv(base_path / 'roster_snapshots' / 'roster_latest.csv')

    espn_to_nba = dict(zip(mapping['espn_name'], mapping['nba_api_name']))

    # Position mapping - try to infer from roster or use default
    # For now, use a simple heuristic based on common positions
    default_positions = {
        'Norman Powell': 'SG', 'Immanuel Quickley': 'PG', 'Toumani Camara': 'SF',
        'Keyonte George': 'PG', 'Keegan Murray': 'PF', 'Mark Williams': 'C',
        'Brandon Ingram': 'SF', 'LeBron James': 'SF', 'Anthony Davis': 'PF',
        'Stephen Curry': 'PG', 'Kevin Durant': 'SF', 'Nikola Jokic': 'C',
        'Luka Doncic': 'PG', 'Giannis Antetokounmpo': 'PF', 'Joel Embiid': 'C',
        'Jayson Tatum': 'SF', 'Damian Lillard': 'PG', 'Trae Young': 'PG',
        'Donovan Mitchell': 'SG', 'Devin Booker': 'SG', 'Ja Morant': 'PG',
        'Tyrese Haliburton': 'PG', 'Shai Gilgeous-Alexander': 'PG',
        'Anthony Edwards': 'SG', 'Paolo Banchero': 'PF', 'Victor Wembanyama': 'C',
        'Chet Holmgren': 'C', 'Scottie Barnes': 'SF', 'Evan Mobley': 'PF',
    }

    if test_all:
        # Get all rostered players
        active_players = roster[roster['currently_rostered'] == True]['player_name'].unique()
        target_players = [p for p in active_players if p in espn_to_nba]
    else:
        target_players = ['Norman Powell', 'Immanuel Quickley', 'Toumani Camara',
                         'Keyonte George', 'Keegan Murray', 'Mark Williams', 'Brandon Ingram']

    print("=" * 80)
    print("BAYESIAN PROJECTION MODEL - TEST RESULTS")
    print(f"Testing {len(target_players)} players...")
    print("=" * 80)

    # Collect results for summary
    all_results = []

    for i, espn_name in enumerate(target_players):
        nba_name = espn_to_nba.get(espn_name, espn_name)
        position = default_positions.get(espn_name, 'SF')  # Default to SF

        if not test_all or (i < 10):  # Only print details for first 10 or if not testing all
            print(f"\n{'='*60}")
            print(f"PLAYER: {espn_name} ({position})")
            print(f"{'='*60}")

        # Fit model
        model = BayesianProjectionModel(evolution_years=1.0, position=position)
        success = model.fit_player(historical, nba_name, position=position)

        if not success:
            if not test_all or (i < 10):
                print("  FAILED TO FIT MODEL")
            continue

        # Get current season actuals
        player_data = historical[historical['PLAYER_NAME'].str.lower() == nba_name.lower()].copy()
        player_data['parsed_date'] = player_data['GAME_DATE'].apply(parse_date)
        player_data = player_data.dropna(subset=['parsed_date'])

        cutoff = datetime(2025, 10, 1)
        current = player_data[player_data['parsed_date'] >= cutoff]
        historical_before_cutoff = player_data[player_data['parsed_date'] < cutoff]

        if len(current) == 0:
            if not test_all or (i < 10):
                print("  No current season data")
            continue

        # Actual current season stats
        actual_pts = current['PTS'].mean()
        actual_reb = current['REB'].mean()
        actual_ast = current['AST'].mean()

        # Calculate actual shooting percentages
        actual_fgm = current['FGM'].sum()
        actual_fga = current['FGA'].sum()
        actual_fg_pct = actual_fgm / actual_fga if actual_fga > 0 else 0

        actual_fg2m = (current['FGM'] - current['FG3M']).sum()
        actual_fg2a = (current['FGA'] - current['FG3A']).sum()
        actual_2p_pct = actual_fg2m / actual_fg2a if actual_fg2a > 0 else 0

        actual_fg3m = current['FG3M'].sum()
        actual_fg3a = current['FG3A'].sum()
        actual_3p_pct = actual_fg3m / actual_fg3a if actual_fg3a > 0 else 0

        actual_ftm = current['FTM'].sum()
        actual_fta = current['FTA'].sum()
        actual_ft_pct = actual_ftm / actual_fta if actual_fta > 0 else 0

        # Simulate 100 games
        sim_stats = {k: [] for k in ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']}

        for _ in range(100):
            game = model.simulate_game()
            for k in sim_stats:
                sim_stats[k].append(game[k])

        # Calculate simulated averages
        sim_pts = np.mean(sim_stats['PTS'])
        sim_reb = np.mean(sim_stats['REB'])
        sim_ast = np.mean(sim_stats['AST'])

        sim_fgm = np.sum(sim_stats['FGM'])
        sim_fga = np.sum(sim_stats['FGA'])
        sim_fg_pct = sim_fgm / sim_fga if sim_fga > 0 else 0

        sim_fg3m = np.sum(sim_stats['FG3M'])
        sim_fg3a = np.sum(sim_stats['FG3A'])
        sim_3p_pct = sim_fg3m / sim_fg3a if sim_fg3a > 0 else 0

        sim_ftm = np.sum(sim_stats['FTM'])
        sim_fta = np.sum(sim_stats['FTA'])
        sim_ft_pct = sim_ftm / sim_fta if sim_fta > 0 else 0

        # Get posterior summary
        posterior = model.get_posterior_summary()

        # Print comparison
        # Calculate errors
        pts_err = (sim_pts - actual_pts) / actual_pts * 100 if actual_pts > 0 else 0
        reb_err = (sim_reb - actual_reb) / actual_reb * 100 if actual_reb > 0 else 0
        ast_err = (sim_ast - actual_ast) / actual_ast * 100 if actual_ast > 0 else 0
        fg_err = (sim_fg_pct - actual_fg_pct) / actual_fg_pct * 100 if actual_fg_pct > 0 else 0
        ft_err = (sim_ft_pct - actual_ft_pct) / actual_ft_pct * 100 if actual_ft_pct > 0 else 0

        # Show FGA comparison
        actual_fga_pg = current['FGA'].mean()
        sim_fga_pg = np.mean(sim_stats['FGA'])
        hist_fga_pg = historical_before_cutoff['FGA'].mean() if len(historical_before_cutoff) > 0 else 0

        # Store result
        all_results.append({
            'player': espn_name,
            'position': position,
            'n_hist': len(historical_before_cutoff),
            'n_curr': len(current),
            'hist_fga': hist_fga_pg,
            'curr_fga': actual_fga_pg,
            'sim_fga': sim_fga_pg,
            'actual_pts': actual_pts,
            'sim_pts': sim_pts,
            'pts_err': pts_err,
            'actual_reb': actual_reb,
            'sim_reb': sim_reb,
            'reb_err': reb_err,
            'actual_ast': actual_ast,
            'sim_ast': sim_ast,
            'ast_err': ast_err,
            'actual_fg_pct': actual_fg_pct,
            'sim_fg_pct': sim_fg_pct,
            'fg_err': fg_err,
            'actual_ft_pct': actual_ft_pct,
            'sim_ft_pct': sim_ft_pct,
            'ft_err': ft_err,
        })

        # Print details for first 10 players or if not test_all
        if not test_all or (i < 10):
            print(f"\n  Current Season Games: {len(current)}")
            print(f"\n  POSTERIOR PARAMETERS:")
            print(f"    2P%: {posterior['2P_PCT']['mean']:.3f} (ESS: {posterior['2P_PCT']['ess']:.0f})")
            print(f"    3P%: {posterior['3P_PCT']['mean']:.3f} (ESS: {posterior['3P_PCT']['ess']:.0f})")
            print(f"    FT%: {posterior['FT_PCT']['mean']:.3f} (ESS: {posterior['FT_PCT']['ess']:.0f})")
            print(f"    3PA Rate: {posterior['3PA_RATE']:.2%}")

            print(f"\n  FGA: Historical={hist_fga_pg:.1f}, Current={actual_fga_pg:.1f}, Simulated={sim_fga_pg:.1f}")

            print(f"\n  COMPARISON (Actual vs Simulated):")
            print(f"  {'Stat':<10} {'Actual':>10} {'Simulated':>10} {'Error':>10}")
            print(f"  {'-'*42}")

            print(f"  {'PTS':<10} {actual_pts:>10.1f} {sim_pts:>10.1f} {pts_err:>+10.1f}%")
            print(f"  {'REB':<10} {actual_reb:>10.1f} {sim_reb:>10.1f} {reb_err:>+10.1f}%")
            print(f"  {'AST':<10} {actual_ast:>10.1f} {sim_ast:>10.1f} {ast_err:>+10.1f}%")
            print(f"  {'FG%':<10} {actual_fg_pct:>10.3f} {sim_fg_pct:>10.3f} {fg_err:>+10.1f}%")
            print(f"  {'2P%':<10} {actual_2p_pct:>10.3f} {posterior['2P_PCT']['mean']:>10.3f} {(posterior['2P_PCT']['mean']-actual_2p_pct)/actual_2p_pct*100 if actual_2p_pct > 0 else 0:>+10.1f}%")
            print(f"  {'3P%':<10} {actual_3p_pct:>10.3f} {sim_3p_pct:>10.3f} {(sim_3p_pct-actual_3p_pct)/actual_3p_pct*100 if actual_3p_pct > 0 else 0:>+10.1f}%")
            print(f"  {'FT%':<10} {actual_ft_pct:>10.3f} {sim_ft_pct:>10.3f} {ft_err:>+10.1f}%")
        else:
            # Progress indicator for remaining players
            if i % 20 == 0:
                print(f"  Processing player {i+1}/{len(target_players)}...")

    # Print summary
    if len(all_results) > 0:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        results_df = pd.DataFrame(all_results)

        print(f"\nTotal players analyzed: {len(results_df)}")

        # Mean absolute errors
        print(f"\nMean Absolute Errors:")
        print(f"  PTS: {results_df['pts_err'].abs().mean():.1f}%")
        print(f"  REB: {results_df['reb_err'].abs().mean():.1f}%")
        print(f"  AST: {results_df['ast_err'].abs().mean():.1f}%")
        print(f"  FG%: {results_df['fg_err'].abs().mean():.1f}%")
        print(f"  FT%: {results_df['ft_err'].abs().mean():.1f}%")

        # Bias (mean signed error)
        print(f"\nMean Signed Errors (Bias):")
        print(f"  PTS: {results_df['pts_err'].mean():+.1f}%")
        print(f"  REB: {results_df['reb_err'].mean():+.1f}%")
        print(f"  AST: {results_df['ast_err'].mean():+.1f}%")
        print(f"  FG%: {results_df['fg_err'].mean():+.1f}%")
        print(f"  FT%: {results_df['ft_err'].mean():+.1f}%")

        # Worst performers
        print(f"\nWorst PTS Errors (>20%):")
        worst_pts = results_df[results_df['pts_err'].abs() > 20].sort_values('pts_err', key=abs, ascending=False)
        for _, row in worst_pts.head(10).iterrows():
            print(f"  {row['player']}: {row['pts_err']:+.1f}% (Actual: {row['actual_pts']:.1f}, Sim: {row['sim_pts']:.1f})")

        print(f"\nWorst FG% Errors (>10%):")
        worst_fg = results_df[results_df['fg_err'].abs() > 10].sort_values('fg_err', key=abs, ascending=False)
        for _, row in worst_fg.head(10).iterrows():
            print(f"  {row['player']}: {row['fg_err']:+.1f}% (Actual: {row['actual_fg_pct']:.3f}, Sim: {row['sim_fg_pct']:.3f})")

        # Best performers
        print(f"\nBest Predictions (PTS error <5%):")
        best = results_df[results_df['pts_err'].abs() < 5].sort_values('pts_err', key=abs)
        for _, row in best.head(10).iterrows():
            print(f"  {row['player']}: {row['pts_err']:+.1f}% (Actual: {row['actual_pts']:.1f}, Sim: {row['sim_pts']:.1f})")

        # Save to CSV
        output_path = Path('/Users/rhu/fantasybasketball3/fantasy_2026/projection_diagnostics')
        results_df.to_csv(output_path / 'bayesian_model_all_players.csv', index=False)
        print(f"\nResults saved to: {output_path / 'bayesian_model_all_players.csv'}")


if __name__ == '__main__':
    test_model_on_players(test_all=True)
