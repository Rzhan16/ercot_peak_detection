"""Trading strategies for ERCOT peak detection."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from price data.
        
        CRITICAL: Can only use data UP TO current timestamp (no lookahead).
        
        Args:
            df: DataFrame with at least 'timestamp' and 'price' columns
            
        Returns:
            DataFrame with added 'signal' column (1=trigger, 0=no action)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """Return dictionary of strategy parameters."""
        pass
    
    def __repr__(self):
        return f"{self.name}(params={self.get_params()})"


class NaiveTimeStrategy(BaseStrategy):
    """
    Naive baseline: Trigger at fixed time each day.
    
    Logic:
    - Historical peaks typically occur afternoon/evening
    - Simply trigger at a fixed time daily (e.g., 4:30 PM or 7:00 PM)
    - This establishes minimum performance threshold
    
    Expected performance: ~35-40% success rate
    """
    
    def __init__(self, trigger_hour: int = 19, trigger_minute: int = 0):
        """
        Initialize naive time-based strategy.
        
        Args:
            trigger_hour: Hour to trigger (0-23), default 19 (7 PM)
            trigger_minute: Minute to trigger (0-59), default 0
        """
        super().__init__("NaiveTime")
        
        if not (0 <= trigger_hour <= 23):
            raise ValueError("trigger_hour must be between 0 and 23")
        if not (0 <= trigger_minute <= 59):
            raise ValueError("trigger_minute must be between 0 and 59")
            
        self.trigger_hour = trigger_hour
        self.trigger_minute = trigger_minute
        
        logger.info(f"Initialized {self.name} strategy: trigger at {trigger_hour}:{trigger_minute:02d}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals by triggering at fixed time daily.
        
        Args:
            df: DataFrame with 'timestamp' and 'price' columns
            
        Returns:
            DataFrame with added 'signal' column
        """
        df = df.copy()
        
        # Extract hour and minute
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Trigger at specified time
        df['signal'] = 0
        df.loc[(df['hour'] == self.trigger_hour) & (df['minute'] == self.trigger_minute), 'signal'] = 1
        
        # Clean up temporary columns
        df = df.drop(['hour', 'minute'], axis=1)
        
        num_signals = df['signal'].sum()
        logger.debug(f"{self.name}: Generated {num_signals} signals")
        
        return df
    
    def get_params(self) -> Dict:
        """Return strategy parameters."""
        return {
            'trigger_hour': self.trigger_hour,
            'trigger_minute': self.trigger_minute
        }


class PriceDropStrategy(BaseStrategy):
    """
    Detect price peaks by identifying drop from recent high.
    
    Logic:
    1. Track rolling maximum price over lookback window
    2. Track rolling average price over long-term window
    3. Trigger when:
       - Current price drops from recent high by drop_threshold %
       - AND recent high is significantly above long-term average
    
    This catches the "price just peaked and is falling" moment.
    """
    
    def __init__(self, 
                 lookback_minutes: int = 15,
                 drop_threshold: float = 0.02,
                 min_price_multiplier: float = 1.15,
                 longterm_minutes: int = 120):
        """
        Initialize price drop strategy.
        
        Args:
            lookback_minutes: Window to find recent high (default: 15 min)
            drop_threshold: % drop to trigger (default: 2%)
            min_price_multiplier: Recent high must be X times long-term avg (default: 1.15)
            longterm_minutes: Long-term average window (default: 120 min)
        """
        super().__init__("PriceDrop")
        
        if lookback_minutes <= 0:
            raise ValueError("lookback_minutes must be positive")
        if not (0 < drop_threshold < 1):
            raise ValueError("drop_threshold must be between 0 and 1")
        if min_price_multiplier < 1:
            raise ValueError("min_price_multiplier must be >= 1")
        if longterm_minutes <= lookback_minutes:
            raise ValueError("longterm_minutes must be > lookback_minutes")
        
        self.lookback_minutes = lookback_minutes
        self.drop_threshold = drop_threshold
        self.min_price_multiplier = min_price_multiplier
        self.longterm_minutes = longterm_minutes
        
        logger.info(f"Initialized {self.name} strategy with lookback={lookback_minutes}min, drop={drop_threshold}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using price drop detection.
        
        Args:
            df: DataFrame with 'timestamp' and 'price' columns
            
        Returns:
            DataFrame with added 'signal' column
        """
        df = df.copy()
        
        # Set timestamp as index for time-based rolling
        df = df.set_index('timestamp')
        
        # Calculate rolling features using time-based windows
        lookback_window = f'{self.lookback_minutes}min'
        longterm_window = f'{self.longterm_minutes}min'
        
        # Rolling max over lookback window
        df['rolling_max'] = df['price'].rolling(lookback_window, min_periods=1).max()
        
        # Rolling mean over long-term window
        df['rolling_mean'] = df['price'].rolling(longterm_window, min_periods=1).mean()
        
        # Calculate drop from recent high
        df['drop_from_high'] = (df['rolling_max'] - df['price']) / df['rolling_max']
        
        # Check if recent high is elevated
        df['is_elevated'] = df['rolling_max'] > (self.min_price_multiplier * df['rolling_mean'])
        
        # Trigger condition
        df['signal'] = 0
        trigger_mask = (
            (df['drop_from_high'] >= self.drop_threshold) & 
            (df['is_elevated'])
        )
        df.loc[trigger_mask, 'signal'] = 1
        
        # Clean up and reset index
        df = df.reset_index()
        df = df[['timestamp', 'price', 'signal']]
        
        num_signals = df['signal'].sum()
        logger.debug(f"{self.name}: Generated {num_signals} signals")
        
        return df
    
    def get_params(self) -> Dict:
        """Return strategy parameters."""
        return {
            'lookback_minutes': self.lookback_minutes,
            'drop_threshold': self.drop_threshold,
            'min_price_multiplier': self.min_price_multiplier,
            'longterm_minutes': self.longterm_minutes
        }


class VelocityReversalStrategy(BaseStrategy):
    """
    Detect peaks by identifying when price acceleration reverses.
    
    Logic:
    - Calculate price velocity (rate of change)
    - Calculate price acceleration (2nd derivative)
    - Trigger when acceleration turns negative AND price is in top percentile
    
    This catches the "price stopped rising" moment.
    """
    
    def __init__(self, 
                 velocity_window_minutes: int = 10,
                 acceleration_threshold: float = -0.5,
                 price_percentile: int = 75,
                 lookback_minutes: int = 60):
        """
        Initialize velocity reversal strategy.
        
        Args:
            velocity_window_minutes: Window for velocity calculation (default: 10 min)
            acceleration_threshold: Negative acceleration to trigger (default: -0.5)
            price_percentile: Current price must be in top X% (default: 75)
            lookback_minutes: Window for percentile calculation (default: 60 min)
        """
        super().__init__("VelocityReversal")
        
        if velocity_window_minutes <= 0:
            raise ValueError("velocity_window_minutes must be positive")
        if acceleration_threshold >= 0:
            raise ValueError("acceleration_threshold must be negative")
        if not (0 < price_percentile < 100):
            raise ValueError("price_percentile must be between 0 and 100")
        
        self.velocity_window_minutes = velocity_window_minutes
        self.acceleration_threshold = acceleration_threshold
        self.price_percentile = price_percentile
        self.lookback_minutes = lookback_minutes
        
        logger.info(f"Initialized {self.name} strategy")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on velocity reversal.
        
        Steps:
        1. Calculate rolling velocity (price change / time)
        2. Calculate rolling acceleration (velocity change / time)
        3. Calculate rolling price percentile
        4. Trigger when: acceleration < threshold AND price > percentile
        
        Args:
            df: DataFrame with 'timestamp' and 'price' columns
            
        Returns:
            DataFrame with added 'signal' column
        """
        df = df.copy()
        df = df.set_index('timestamp')
        
        # Calculate velocity (first derivative)
        velocity_window = f'{self.velocity_window_minutes}min'
        df['price_change'] = df['price'].diff()
        df['velocity'] = df['price_change'].rolling(velocity_window, min_periods=1).mean()
        
        # Calculate acceleration (second derivative)
        df['acceleration'] = df['velocity'].diff()
        
        # Calculate rolling percentile threshold
        lookback_window = f'{self.lookback_minutes}min'
        df['percentile_threshold'] = df['price'].rolling(
            lookback_window, 
            min_periods=1
        ).quantile(self.price_percentile / 100)
        
        # Trigger conditions
        df['signal'] = 0
        trigger_mask = (
            (df['acceleration'] < self.acceleration_threshold) &
            (df['price'] > df['percentile_threshold'])
        )
        df.loc[trigger_mask, 'signal'] = 1
        
        # Clean up
        df = df.reset_index()
        df = df[['timestamp', 'price', 'signal']]
        
        num_signals = df['signal'].sum()
        logger.debug(f"{self.name}: Generated {num_signals} signals")
        
        return df
    
    def get_params(self) -> Dict:
        """Return strategy parameters."""
        return {
            'velocity_window_minutes': self.velocity_window_minutes,
            'acceleration_threshold': self.acceleration_threshold,
            'price_percentile': self.price_percentile,
            'lookback_minutes': self.lookback_minutes
        }


class DayAheadDeviationStrategy(BaseStrategy):
    """
    Detect peaks when real-time price deviates significantly from forecast.
    
    Logic:
    - Compare RTM price to DAM forecast
    - Trigger when RTM >> DAM (high demand surprise) AND price starts falling
    
    This catches unexpected demand spikes.
    
    Note: Requires merged DataFrame with both RTM and DAM prices.
    """
    
    def __init__(self, 
                 deviation_threshold: float = 1.25,
                 drop_threshold: float = 0.015,
                 lookback_minutes: int = 10):
        """
        Initialize day-ahead deviation strategy.
        
        Args:
            deviation_threshold: RTM/DAM ratio to trigger (default: 1.25 = 25% above)
            drop_threshold: Price drop % after deviation (default: 1.5%)
            lookback_minutes: Window to detect drop (default: 10 min)
        """
        super().__init__("DayAheadDeviation")
        
        if deviation_threshold <= 1:
            raise ValueError("deviation_threshold must be > 1")
        if not (0 < drop_threshold < 1):
            raise ValueError("drop_threshold must be between 0 and 1")
        
        self.deviation_threshold = deviation_threshold
        self.drop_threshold = drop_threshold
        self.lookback_minutes = lookback_minutes
        
        logger.info(f"Initialized {self.name} strategy")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on DAM deviation.
        
        Input df must have columns: timestamp, price, dam_price
        
        Trigger when:
        1. price / dam_price > deviation_threshold
        2. Price drops by drop_threshold from recent high
        
        Args:
            df: DataFrame with 'timestamp', 'price', and 'dam_price' columns
            
        Returns:
            DataFrame with added 'signal' column
        """
        if 'dam_price' not in df.columns:
            raise ValueError("DataFrame must include 'dam_price' column. Use merged RTM+DAM data.")
        
        df = df.copy()
        df = df.set_index('timestamp')
        
        # Calculate deviation from forecast
        df['deviation_ratio'] = df['price'] / df['dam_price'].replace(0, np.nan)
        
        # Calculate drop from recent high
        lookback_window = f'{self.lookback_minutes}min'
        df['rolling_max'] = df['price'].rolling(lookback_window, min_periods=1).max()
        df['drop_from_high'] = (df['rolling_max'] - df['price']) / df['rolling_max']
        
        # Trigger conditions
        df['signal'] = 0
        trigger_mask = (
            (df['deviation_ratio'] > self.deviation_threshold) &
            (df['drop_from_high'] > self.drop_threshold)
        )
        df.loc[trigger_mask, 'signal'] = 1
        
        # Clean up
        df = df.reset_index()
        df = df[['timestamp', 'price', 'signal']]
        
        num_signals = df['signal'].sum()
        logger.debug(f"{self.name}: Generated {num_signals} signals")
        
        return df
    
    def get_params(self) -> Dict:
        """Return strategy parameters."""
        return {
            'deviation_threshold': self.deviation_threshold,
            'drop_threshold': self.drop_threshold,
            'lookback_minutes': self.lookback_minutes
        }


class EnsembleStrategy(BaseStrategy):
    """
    Combine multiple strategies with voting.
    
    Logic:
    - Run all sub-strategies
    - Trigger only when at least min_votes strategies agree
    - This reduces false positives while maintaining coverage
    """
    
    def __init__(self, strategies: List[BaseStrategy], min_votes: int = 2):
        """
        Initialize ensemble strategy.
        
        Args:
            strategies: List of strategy objects to combine
            min_votes: Minimum number of strategies that must trigger (default: 2)
        """
        super().__init__("Ensemble")
        
        if not strategies:
            raise ValueError("Must provide at least one strategy")
        if min_votes < 1 or min_votes > len(strategies):
            raise ValueError(f"min_votes must be between 1 and {len(strategies)}")
        
        self.strategies = strategies
        self.min_votes = min_votes
        
        strategy_names = [s.name for s in strategies]
        logger.info(f"Initialized {self.name} with {len(strategies)} strategies: {strategy_names}")
        logger.info(f"Min votes required: {min_votes}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all strategies and combine votes.
        
        Steps:
        1. For each strategy, generate signals
        2. Sum signals across strategies at each timestamp
        3. Trigger when sum >= min_votes
        
        Args:
            df: DataFrame with required columns for all strategies
            
        Returns:
            DataFrame with 'signal' column
        """
        df_result = df.copy()
        
        # Collect signals from all strategies
        all_signals = []
        
        for strategy in self.strategies:
            try:
                strategy_signals = strategy.generate_signals(df.copy())
                # Ensure same index
                if 'signal' in strategy_signals.columns:
                    all_signals.append(strategy_signals.set_index('timestamp')['signal'])
                else:
                    logger.warning(f"Strategy {strategy.name} didn't return 'signal' column")
                    continue
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")
                continue
        
        if not all_signals:
            raise ValueError("No strategies successfully generated signals")
        
        # Align all signals to same timestamps
        df_result = df_result.set_index('timestamp')
        
        # Create DataFrame of all votes
        votes_df = pd.DataFrame({f'vote_{i}': sig for i, sig in enumerate(all_signals)})
        
        # Ensure same index
        votes_df = votes_df.reindex(df_result.index, fill_value=0)
        
        # Sum votes
        vote_sum = votes_df.sum(axis=1)
        
        # Trigger when votes >= threshold
        df_result['signal'] = (vote_sum >= self.min_votes).astype(int)
        
        df_result = df_result.reset_index()
        df_result = df_result[['timestamp', 'price', 'signal']]
        
        num_signals = df_result['signal'].sum()
        logger.debug(f"{self.name}: Generated {num_signals} signals (min_votes={self.min_votes})")
        
        return df_result
    
    def get_params(self) -> Dict:
        """Return ensemble parameters."""
        return {
            'strategies': [s.name for s in self.strategies],
            'strategy_params': [s.get_params() for s in self.strategies],
            'min_votes': self.min_votes
        }


def create_best_ensemble(use_merged_data: bool = False) -> EnsembleStrategy:
    """
    Create ensemble of best-performing strategies.
    
    Args:
        use_merged_data: Whether to include strategies requiring DAM data
        
    Returns:
        Configured EnsembleStrategy
    """
    strategies = [
        PriceDropStrategy(lookback_minutes=15, drop_threshold=0.02, min_price_multiplier=1.15),
        VelocityReversalStrategy(velocity_window_minutes=10, acceleration_threshold=-0.5),
    ]
    
    if use_merged_data:
        strategies.append(
            DayAheadDeviationStrategy(deviation_threshold=1.25, drop_threshold=0.015)
        )
    
    return EnsembleStrategy(strategies, min_votes=2)
