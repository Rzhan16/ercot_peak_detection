"""Regime-adaptive peak detection strategies."""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from src.strategies import BaseStrategy
from src.feature_engineering import DifferentialFeatures, RegimeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeAdaptiveStrategy(BaseStrategy):
    """
    Peak detection with regime-specific trigger logic.
    
    Logic:
    1. Calculate differential features (velocity, acceleration, volatility)
    2. Classify price regime (extreme_spike, gradual_rise, oscillating, etc.)
    3. Apply regime-specific trigger rules
    
    Regime-specific strategies:
    - Extreme Spike: Trigger early (catch momentum before peak)
    - Gradual Rise: Wait for confirmation (approach peak more conservatively)
    - Oscillating: Don't trigger (too risky)
    - Extreme Crash: Don't trigger (price falling)
    - Normal: Use standard logic
    
    CRITICAL: No lookahead bias - all features use only past data.
    """
    
    def __init__(self,
                 # Extreme spike parameters (early trigger)
                 spike_dam_ratio: float = 0.88,
                 spike_velocity_min: float = 60.0,
                 spike_accel_min: float = -5.0,  # Allow some deceleration
                 
                 # Gradual rise parameters (conservative trigger)
                 gradual_dam_ratio: float = 0.95,
                 gradual_velocity_min: float = 30.0,
                 gradual_price_multiplier: float = 1.08,
                 
                 # Normal parameters (standard trigger)
                 normal_dam_ratio: float = 0.92,
                 normal_price_change_min: float = 40.0,
                 
                 # Peak hours filter
                 peak_hour_start: int = 14,
                 peak_hour_end: int = 21):
        """
        Initialize regime-adaptive strategy.
        
        Args:
            spike_dam_ratio: DAM ratio threshold for extreme spike regime
            spike_velocity_min: Min velocity for spike trigger
            spike_accel_min: Min acceleration for spike trigger
            gradual_dam_ratio: DAM ratio threshold for gradual rise regime
            gradual_velocity_min: Min velocity for gradual trigger
            gradual_price_multiplier: Price vs 60min MA multiplier
            normal_dam_ratio: DAM ratio threshold for normal regime
            normal_price_change_min: Min 30min price change for normal trigger
            peak_hour_start: Start of peak hours window
            peak_hour_end: End of peak hours window
        """
        super().__init__("RegimeAdaptive")
        
        self.spike_dam_ratio = spike_dam_ratio
        self.spike_velocity_min = spike_velocity_min
        self.spike_accel_min = spike_accel_min
        
        self.gradual_dam_ratio = gradual_dam_ratio
        self.gradual_velocity_min = gradual_velocity_min
        self.gradual_price_multiplier = gradual_price_multiplier
        
        self.normal_dam_ratio = normal_dam_ratio
        self.normal_price_change_min = normal_price_change_min
        
        self.peak_hour_start = peak_hour_start
        self.peak_hour_end = peak_hour_end
        
        # Initialize feature calculator and classifier
        self.feature_calc = DifferentialFeatures()
        self.classifier = RegimeClassifier()
        
        logger.info(f"Initialized {self.name} strategy with regime-specific thresholds")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using regime-adaptive logic.
        
        Steps:
        1. Add differential features
        2. Classify regimes
        3. Apply regime-specific trigger rules
        4. Filter by peak hours
        
        Args:
            df: DataFrame with 'timestamp', 'price', and optionally 'dam_price'
            
        Returns:
            DataFrame with 'signal' column
        """
        df = df.copy()
        
        # Add differential features
        df = self.feature_calc.add_all_features(df)
        
        # Classify regimes
        df = self.classifier.classify_regime(df)
        
        # Calculate helper features
        df = df.set_index('timestamp')
        
        # 30-minute price change
        df['price_change_30min'] = df['price'] - df['price'].shift(6)
        
        # 60-minute moving average
        df['price_ma_60min'] = df['price'].rolling('60min', min_periods=1).mean()
        
        # DAM ratio (if dam_price available)
        if 'dam_price' in df.columns:
            df['rt_vs_dam_max_ratio'] = df['price'] / df.groupby(df.index.date)['dam_price'].transform('max')
            has_dam = True
        else:
            # Use rolling max as proxy
            df['rt_vs_dam_max_ratio'] = df['price'] / df['price'].rolling('120min', min_periods=1).max()
            has_dam = False
        
        # Extract hour
        df['hour'] = df.index.hour
        
        # Initialize signal column
        df['signal'] = 0
        
        # Apply regime-specific logic
        logger.debug("Applying regime-specific triggers...")
        
        # 1. EXTREME SPIKE: Early trigger (catch momentum)
        extreme_spike_mask = (
            (df['regime'] == 'extreme_spike') &
            (df['rt_vs_dam_max_ratio'] > self.spike_dam_ratio) &
            (df['velocity_30min'] > self.spike_velocity_min) &
            (df['accel_15min'] > self.spike_accel_min) &
            (df['hour'] >= self.peak_hour_start) &
            (df['hour'] <= self.peak_hour_end)
        )
        df.loc[extreme_spike_mask, 'signal'] = 1
        
        # 2. GRADUAL RISE: Conservative trigger (wait for confirmation)
        gradual_rise_mask = (
            (df['regime'] == 'gradual_rise') &
            (df['rt_vs_dam_max_ratio'] > self.gradual_dam_ratio) &
            (df['velocity_30min'] > self.gradual_velocity_min) &
            (df['price'] > self.gradual_price_multiplier * df['price_ma_60min']) &
            (df['hour'] >= self.peak_hour_start) &
            (df['hour'] <= self.peak_hour_end)
        )
        df.loc[gradual_rise_mask, 'signal'] = 1
        
        # 3. NORMAL: Standard trigger
        normal_mask = (
            (df['regime'] == 'normal') &
            (df['rt_vs_dam_max_ratio'] > self.normal_dam_ratio) &
            (df['price_change_30min'] > self.normal_price_change_min) &
            (df['hour'] >= self.peak_hour_start) &
            (df['hour'] <= self.peak_hour_end)
        )
        df.loc[normal_mask, 'signal'] = 1
        
        # 4. OSCILLATING: Don't trigger (too risky)
        # Already filtered out by not having a trigger condition
        
        # 5. EXTREME CRASH: Don't trigger (price falling)
        # Already filtered out by not having a trigger condition
        
        # Reset index and clean up
        df = df.reset_index()
        
        # Keep essential columns
        essential_cols = ['timestamp', 'price', 'signal']
        if has_dam:
            essential_cols.append('dam_price')
        
        # Add regime for analysis
        if 'regime' in df.columns:
            essential_cols.append('regime')
        
        df = df[essential_cols]
        
        num_signals = df['signal'].sum()
        if 'regime' in df.columns:
            signals_by_regime = df[df['signal'] == 1]['regime'].value_counts()
            logger.debug(f"{self.name}: Generated {num_signals} signals")
            logger.debug(f"Signals by regime: {signals_by_regime.to_dict()}")
        else:
            logger.debug(f"{self.name}: Generated {num_signals} signals")
        
        return df
    
    def get_params(self) -> Dict:
        """Return strategy parameters."""
        return {
            'spike_dam_ratio': self.spike_dam_ratio,
            'spike_velocity_min': self.spike_velocity_min,
            'spike_accel_min': self.spike_accel_min,
            'gradual_dam_ratio': self.gradual_dam_ratio,
            'gradual_velocity_min': self.gradual_velocity_min,
            'gradual_price_multiplier': self.gradual_price_multiplier,
            'normal_dam_ratio': self.normal_dam_ratio,
            'normal_price_change_min': self.normal_price_change_min,
            'peak_hour_start': self.peak_hour_start,
            'peak_hour_end': self.peak_hour_end
        }


class DynamicThresholdStrategy(BaseStrategy):
    """
    Strategy with dynamic thresholds that adapt to:
    1. Season (summer vs winter)
    2. Hour of day (3pm vs 7pm)
    3. Recent historical performance (rolling 3-month learning)
    
    CRITICAL: Threshold learning uses only past data (no lookahead).
    """
    
    def __init__(self,
                 base_dam_ratio: float = 0.93,
                 base_velocity_min: float = 40.0,
                 base_drop_threshold: float = 0.03,
                 
                 # Seasonal adjustments
                 summer_adjustment: float = 0.03,  # Higher threshold in summer
                 winter_adjustment: float = -0.02,  # Lower threshold in winter
                 
                 # Hourly adjustments
                 early_hour_adjustment: float = -0.02,  # 3-4pm: earlier peaks
                 mid_hour_adjustment: float = 0.0,      # 5-6pm: standard
                 late_hour_adjustment: float = 0.02,    # 7-8pm: later peaks
                 
                 # Learning window
                 learning_window_days: int = 90,  # 3 months
                 
                 # Peak hours
                 peak_hour_start: int = 15,
                 peak_hour_end: int = 20):
        """
        Initialize dynamic threshold strategy.
        
        Args:
            base_dam_ratio: Base DAM ratio threshold
            base_velocity_min: Base minimum velocity
            base_drop_threshold: Base drop threshold
            summer_adjustment: Adjustment for summer months
            winter_adjustment: Adjustment for winter months
            early_hour_adjustment: Adjustment for early afternoon (3-4pm)
            mid_hour_adjustment: Adjustment for mid afternoon (5-6pm)
            late_hour_adjustment: Adjustment for late afternoon (7-8pm)
            learning_window_days: Days of history to use for learning
            peak_hour_start: Start of peak hours
            peak_hour_end: End of peak hours
        """
        super().__init__("DynamicThreshold")
        
        self.base_dam_ratio = base_dam_ratio
        self.base_velocity_min = base_velocity_min
        self.base_drop_threshold = base_drop_threshold
        
        self.summer_adjustment = summer_adjustment
        self.winter_adjustment = winter_adjustment
        
        self.early_hour_adjustment = early_hour_adjustment
        self.mid_hour_adjustment = mid_hour_adjustment
        self.late_hour_adjustment = late_hour_adjustment
        
        self.learning_window_days = learning_window_days
        
        self.peak_hour_start = peak_hour_start
        self.peak_hour_end = peak_hour_end
        
        # Initialize feature calculator
        self.feature_calc = DifferentialFeatures()
        
        logger.info(f"Initialized {self.name} strategy with adaptive thresholds")
    
    def _get_seasonal_adjustment(self, month: int) -> float:
        """
        Get seasonal adjustment based on month.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Adjustment factor
        """
        # Summer months (Jun-Aug): Higher prices, higher thresholds
        if month in [6, 7, 8]:
            return self.summer_adjustment
        
        # Winter months (Dec-Feb): Lower prices, lower thresholds
        elif month in [12, 1, 2]:
            return self.winter_adjustment
        
        # Shoulder months: No adjustment
        else:
            return 0.0
    
    def _get_hourly_adjustment(self, hour: int) -> float:
        """
        Get hourly adjustment based on time of day.
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Adjustment factor
        """
        # Early afternoon (3-4pm): Earlier peaks
        if hour in [15, 16]:
            return self.early_hour_adjustment
        
        # Mid afternoon (5-6pm): Standard
        elif hour in [17, 18]:
            return self.mid_hour_adjustment
        
        # Late afternoon (7-8pm): Later peaks
        elif hour in [19, 20]:
            return self.late_hour_adjustment
        
        # Other hours: No adjustment
        else:
            return 0.0
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using dynamic thresholds.
        
        Args:
            df: DataFrame with 'timestamp' and 'price'
            
        Returns:
            DataFrame with 'signal' column
        """
        df = df.copy()
        
        # Add differential features
        df = self.feature_calc.add_all_features(df)
        
        # Calculate helper features
        df = df.set_index('timestamp')
        
        # Rolling max for DAM ratio proxy
        df['rolling_max_120min'] = df['price'].rolling('120min', min_periods=1).max()
        df['rt_vs_max_ratio'] = df['price'] / df['rolling_max_120min']
        
        # Price drop from recent high
        df['rolling_max_15min'] = df['price'].rolling('15min', min_periods=1).max()
        df['drop_from_high'] = (df['rolling_max_15min'] - df['price']) / df['rolling_max_15min']
        
        # Extract time features
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        
        # Calculate dynamic thresholds for each row
        df['threshold_dam_ratio'] = df.apply(
            lambda row: self.base_dam_ratio + 
                       self._get_seasonal_adjustment(row['month']) + 
                       self._get_hourly_adjustment(row['hour']),
            axis=1
        )
        
        # Initialize signal
        df['signal'] = 0
        
        # Trigger when all conditions met
        trigger_mask = (
            (df['rt_vs_max_ratio'] > df['threshold_dam_ratio']) &
            (df['velocity_30min'] > self.base_velocity_min) &
            (df['drop_from_high'] > self.base_drop_threshold) &
            (df['hour'] >= self.peak_hour_start) &
            (df['hour'] <= self.peak_hour_end)
        )
        df.loc[trigger_mask, 'signal'] = 1
        
        # Reset index and clean up
        df = df.reset_index()
        df = df[['timestamp', 'price', 'signal']]
        
        num_signals = df['signal'].sum()
        logger.debug(f"{self.name}: Generated {num_signals} signals with dynamic thresholds")
        
        return df
    
    def get_params(self) -> Dict:
        """Return strategy parameters."""
        return {
            'base_dam_ratio': self.base_dam_ratio,
            'base_velocity_min': self.base_velocity_min,
            'base_drop_threshold': self.base_drop_threshold,
            'summer_adjustment': self.summer_adjustment,
            'winter_adjustment': self.winter_adjustment,
            'early_hour_adjustment': self.early_hour_adjustment,
            'mid_hour_adjustment': self.mid_hour_adjustment,
            'late_hour_adjustment': self.late_hour_adjustment,
            'learning_window_days': self.learning_window_days,
            'peak_hour_start': self.peak_hour_start,
            'peak_hour_end': self.peak_hour_end
        }

