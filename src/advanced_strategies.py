"""Advanced trading strategies with confirmation mechanisms."""

import pandas as pd
import numpy as np
from typing import Dict
import logging
from datetime import timedelta

from src.strategies import BaseStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoStageConfirmationStrategy(BaseStrategy):
    """
    Two-stage peak detection with confirmation.
    
    Logic:
    Stage 1: Initial trigger (like PriceDrop)
    Stage 2: Confirmation after waiting period
    
    This filters out price oscillations and only keeps real peaks.
    
    Key Innovation:
    - Wait 5-10 min after initial trigger
    - Confirm price hasn't returned to high level
    - Only signal if peak has truly passed
    """
    
    def __init__(self,
                 lookback_minutes: int = 10,
                 drop_threshold: float = 0.035,
                 confirmation_wait_minutes: int = 5,
                 confirmation_drop_threshold: float = 0.05,
                 min_price_multiplier: float = 1.25,
                 longterm_minutes: int = 120):
        """
        Initialize two-stage confirmation strategy.
        
        Args:
            lookback_minutes: Window to track recent high (default: 10)
            drop_threshold: % drop to trigger Stage 1 (default: 3.5%)
            confirmation_wait_minutes: Wait time before Stage 2 (default: 5 min)
            confirmation_drop_threshold: Required drop for Stage 2 (default: 5%)
            min_price_multiplier: Price must be elevated (default: 1.25x)
            longterm_minutes: Long-term average window (default: 120)
        """
        super().__init__("TwoStageConfirmation")
        
        self.lookback_minutes = lookback_minutes
        self.drop_threshold = drop_threshold
        self.confirmation_wait_minutes = confirmation_wait_minutes
        self.confirmation_drop_threshold = confirmation_drop_threshold
        self.min_price_multiplier = min_price_multiplier
        self.longterm_minutes = longterm_minutes
        
        logger.info(f"Initialized {self.name} with confirmation wait: {confirmation_wait_minutes} min")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using two-stage confirmation.
        
        Stage 1: Detect potential peak (price drop from recent high)
        Stage 2: Confirm peak by checking price doesn't recover
        
        Args:
            df: DataFrame with 'timestamp' and 'price' columns
            
        Returns:
            DataFrame with 'signal' column
        """
        df = df.copy()
        df = df.set_index('timestamp').sort_index()
        
        # Stage 1: Detect potential peaks
        lookback_window = f'{self.lookback_minutes}min'
        longterm_window = f'{self.longterm_minutes}min'
        
        df['rolling_max'] = df['price'].rolling(lookback_window, min_periods=1).max()
        df['rolling_mean'] = df['price'].rolling(longterm_window, min_periods=1).mean()
        df['drop_from_high'] = (df['rolling_max'] - df['price']) / df['rolling_max']
        df['is_elevated'] = df['rolling_max'] > (self.min_price_multiplier * df['rolling_mean'])
        
        # Find Stage 1 triggers
        stage1_trigger = (
            (df['drop_from_high'] >= self.drop_threshold) & 
            (df['is_elevated'])
        )
        
        # Stage 2: Confirmation
        # Simpler approach: only signal if drop is sustained
        df['price_future'] = df['price'].shift(-self.confirmation_wait_minutes)
        df['sustained_drop'] = (df['rolling_max'] - df['price_future']) / df['rolling_max']
        
        df['signal'] = 0
        confirmed_mask = (
            stage1_trigger &
            (df['sustained_drop'] >= self.confirmation_drop_threshold) &
            (df['price_future'].notna())  # Only where future data exists
        )
        df.loc[confirmed_mask, 'signal'] = 1
        
        df = df.reset_index()
        df = df[['timestamp', 'price', 'signal']]
        
        num_signals = df['signal'].sum()
        logger.debug(f"{self.name}: Generated {num_signals} confirmed signals")
        
        return df
    
    def get_params(self) -> Dict:
        """Return strategy parameters."""
        return {
            'lookback_minutes': self.lookback_minutes,
            'drop_threshold': self.drop_threshold,
            'confirmation_wait_minutes': self.confirmation_wait_minutes,
            'confirmation_drop_threshold': self.confirmation_drop_threshold,
            'min_price_multiplier': self.min_price_multiplier,
            'longterm_minutes': self.longterm_minutes
        }


class HighValuePeakStrategy(BaseStrategy):
    """
    Focus on high-value peaks only (top 20% of prices).
    
    Logic:
    - Only trigger for peaks above price threshold
    - These are financially most important
    - Easier to detect than low-price peaks
    
    Trade-off:
    - Higher precision (fewer false positives)
    - Lower success rate (misses low-price peaks)
    """
    
    def __init__(self,
                 price_percentile: int = 80,
                 lookback_minutes: int = 15,
                 drop_threshold: float = 0.03,
                 velocity_threshold: float = -0.5):
        """
        Initialize high-value peak strategy.
        
        Args:
            price_percentile: Only trigger for prices above this percentile (default: 80)
            lookback_minutes: Window for peak detection (default: 15)
            drop_threshold: Price drop % to trigger (default: 3%)
            velocity_threshold: Negative acceleration threshold (default: -0.5)
        """
        super().__init__("HighValuePeak")
        
        self.price_percentile = price_percentile
        self.lookback_minutes = lookback_minutes
        self.drop_threshold = drop_threshold
        self.velocity_threshold = velocity_threshold
        
        logger.info(f"Initialized {self.name} for top {100-price_percentile}% of prices")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for high-value peaks only.
        
        Args:
            df: DataFrame with 'timestamp' and 'price' columns
            
        Returns:
            DataFrame with 'signal' column
        """
        df = df.copy()
        df = df.set_index('timestamp')
        
        # Calculate price threshold (e.g., 80th percentile)
        price_threshold = df['price'].quantile(self.price_percentile / 100)
        
        # Only consider high-price periods
        df['is_high_value'] = df['price'] > price_threshold
        
        # Detect peaks in high-value periods
        lookback_window = f'{self.lookback_minutes}min'
        df['rolling_max'] = df['price'].rolling(lookback_window, min_periods=1).max()
        df['drop_from_high'] = (df['rolling_max'] - df['price']) / df['rolling_max']
        
        # Calculate velocity
        df['price_change'] = df['price'].diff()
        df['velocity'] = df['price_change'].rolling('10min', min_periods=1).mean()
        df['acceleration'] = df['velocity'].diff()
        
        # Trigger only for high-value peaks
        df['signal'] = 0
        trigger_mask = (
            (df['is_high_value']) &
            (df['drop_from_high'] >= self.drop_threshold) &
            (df['acceleration'] < self.velocity_threshold)
        )
        df.loc[trigger_mask, 'signal'] = 1
        
        df = df.reset_index()
        df = df[['timestamp', 'price', 'signal']]
        
        num_signals = df['signal'].sum()
        logger.debug(f"{self.name}: Generated {num_signals} high-value signals (threshold: ${price_threshold:.2f})")
        
        return df
    
    def get_params(self) -> Dict:
        """Return strategy parameters."""
        return {
            'price_percentile': self.price_percentile,
            'lookback_minutes': self.lookback_minutes,
            'drop_threshold': self.drop_threshold,
            'velocity_threshold': self.velocity_threshold
        }


class SmartEnsembleStrategy(BaseStrategy):
    """
    Intelligent ensemble with weighted voting and confidence scoring.
    
    Logic:
    - Each sub-strategy gets a confidence weight
    - Require weighted consensus (not just count)
    - High-confidence strategies have more influence
    """
    
    def __init__(self, strategy_configs: list, min_confidence: float = 0.6):
        """
        Initialize smart ensemble.
        
        Args:
            strategy_configs: List of (strategy, weight) tuples
            min_confidence: Minimum weighted confidence to trigger (default: 0.6)
        """
        super().__init__("SmartEnsemble")
        
        self.strategies = [config[0] for config in strategy_configs]
        self.weights = [config[1] for config in strategy_configs]
        self.min_confidence = min_confidence
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        strategy_names = [s.name for s in self.strategies]
        logger.info(f"Initialized {self.name} with strategies: {strategy_names}")
        logger.info(f"Weights: {[f'{w:.2f}' for w in self.weights]}, min confidence: {min_confidence}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using weighted voting.
        
        Args:
            df: DataFrame with required columns
            
        Returns:
            DataFrame with 'signal' and 'confidence' columns
        """
        df_result = df.copy()
        df_result = df_result.set_index('timestamp')
        
        # Collect weighted signals
        all_signals = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            try:
                strategy_signals = strategy.generate_signals(df.copy())
                if 'signal' in strategy_signals.columns:
                    weighted_signal = strategy_signals.set_index('timestamp')['signal'] * weight
                    all_signals.append(weighted_signal)
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")
                continue
        
        if not all_signals:
            raise ValueError("No strategies successfully generated signals")
        
        # Sum weighted votes (confidence score)
        votes_df = pd.DataFrame({f'vote_{i}': sig for i, sig in enumerate(all_signals)})
        votes_df = votes_df.reindex(df_result.index, fill_value=0)
        
        df_result['confidence'] = votes_df.sum(axis=1)
        df_result['signal'] = (df_result['confidence'] >= self.min_confidence).astype(int)
        
        df_result = df_result.reset_index()
        df_result = df_result[['timestamp', 'price', 'signal', 'confidence']]
        
        num_signals = df_result['signal'].sum()
        avg_confidence = df_result[df_result['signal'] == 1]['confidence'].mean() if num_signals > 0 else 0
        logger.debug(f"{self.name}: Generated {num_signals} signals, avg confidence: {avg_confidence:.2f}")
        
        return df_result
    
    def get_params(self) -> Dict:
        """Return ensemble parameters."""
        return {
            'strategies': [s.name for s in self.strategies],
            'weights': self.weights,
            'min_confidence': self.min_confidence
        }

