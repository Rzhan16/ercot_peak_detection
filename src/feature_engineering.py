"""Feature engineering for peak detection - differential geometry features."""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifferentialFeatures:
    """
    Calculate differential geometry features: velocity, acceleration, volatility.
    
    These features capture price dynamics at multiple time scales.
    
    CRITICAL: All calculations use ONLY backward-looking rolling windows (no lookahead).
    """
    
    def __init__(self):
        """Initialize differential features calculator."""
        self.feature_names = []
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all differential features to DataFrame.
        
        Features added:
        - Velocity (1st derivative) at 5min, 15min, 30min scales
        - Acceleration (2nd derivative) at 5min, 15min scales
        - Local volatility at 30min, 60min scales
        
        Args:
            df: DataFrame with 'timestamp' and 'price' columns
            
        Returns:
            DataFrame with added feature columns
            
        CRITICAL: No lookahead bias - only uses past data
        """
        df = df.copy()
        
        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Set timestamp as index for rolling operations
        df = df.set_index('timestamp')
        
        logger.info("Adding multi-scale differential features...")
        
        # 1. First Derivative: Velocity (Price Rate of Change)
        df = self._add_velocity_features(df)
        
        # 2. Second Derivative: Acceleration (Velocity Rate of Change)
        df = self._add_acceleration_features(df)
        
        # 3. Local Volatility (Standard Deviation)
        df = self._add_volatility_features(df)
        
        # Reset index
        df = df.reset_index()
        
        logger.info(f"Added {len(self.feature_names)} features: {self.feature_names}")
        
        return df
    
    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add velocity features (first derivative).
        
        Velocity = (price_t - price_t-n) / time_window
        
        Positive velocity = price rising
        Negative velocity = price falling
        
        Args:
            df: DataFrame with price (timestamp as index)
            
        Returns:
            DataFrame with velocity columns added
        """
        # 5-minute velocity (very short term)
        # This captures immediate price movement
        df['velocity_5min'] = df['price'].diff(periods=1)  # 1 period = 5 min
        self.feature_names.append('velocity_5min')
        
        # 15-minute velocity (short term)
        # Smooths out noise, shows clear direction
        df['velocity_15min'] = df['price'].diff(periods=3) / 3  # 3 periods = 15 min
        self.feature_names.append('velocity_15min')
        
        # 30-minute velocity (medium term)
        # Shows sustained trends
        df['velocity_30min'] = df['price'].diff(periods=6) / 6  # 6 periods = 30 min
        self.feature_names.append('velocity_30min')
        
        # 60-minute velocity (long term)
        # Shows major price movements
        df['velocity_60min'] = df['price'].diff(periods=12) / 12  # 12 periods = 60 min
        self.feature_names.append('velocity_60min')
        
        logger.debug(f"Added velocity features: 5min, 15min, 30min, 60min")
        
        return df
    
    def _add_acceleration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add acceleration features (second derivative).
        
        Acceleration = (velocity_t - velocity_t-1) / time_window
        
        Positive acceleration = price rising faster (peak approaching)
        Negative acceleration = price rising slower OR falling faster (peak passed)
        
        Args:
            df: DataFrame with velocity columns
            
        Returns:
            DataFrame with acceleration columns added
        """
        # Must have velocity features first
        if 'velocity_5min' not in df.columns:
            raise ValueError("Must add velocity features before acceleration")
        
        # 5-minute acceleration (immediate change in momentum)
        df['accel_5min'] = df['velocity_5min'].diff()
        self.feature_names.append('accel_5min')
        
        # 15-minute acceleration (short-term momentum change)
        # Most useful for peak detection
        df['accel_15min'] = df['velocity_15min'].diff()
        self.feature_names.append('accel_15min')
        
        # 30-minute acceleration (medium-term momentum change)
        df['accel_30min'] = df['velocity_30min'].diff()
        self.feature_names.append('accel_30min')
        
        logger.debug(f"Added acceleration features: 5min, 15min, 30min")
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add local volatility features (standard deviation).
        
        Volatility = std(price) over rolling window
        
        High volatility = unstable, oscillating prices (risky to trigger)
        Low volatility = stable trends (safer to trigger)
        
        Args:
            df: DataFrame with price (timestamp as index)
            
        Returns:
            DataFrame with volatility columns added
        """
        # 30-minute volatility (short-term price stability)
        df['volatility_30min'] = df['price'].rolling('30min', min_periods=1).std()
        self.feature_names.append('volatility_30min')
        
        # 60-minute volatility (medium-term price stability)
        df['volatility_60min'] = df['price'].rolling('60min', min_periods=1).std()
        self.feature_names.append('volatility_60min')
        
        # 120-minute volatility (long-term price stability)
        df['volatility_120min'] = df['price'].rolling('120min', min_periods=1).std()
        self.feature_names.append('volatility_120min')
        
        logger.debug(f"Added volatility features: 30min, 60min, 120min")
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for all features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with feature statistics
        """
        summary = {}
        
        for feature in self.feature_names:
            if feature in df.columns:
                summary[feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'nulls': df[feature].isnull().sum()
                }
        
        return summary


class RegimeClassifier:
    """
    Classify price regimes based on differential features.
    
    Regime types:
    - extreme_spike: Very fast price increase (trigger early)
    - gradual_rise: Steady price increase (wait for confirmation)
    - extreme_crash: Very fast price decrease (don't trigger)
    - oscillating: High volatility, no clear trend (don't trigger)
    - normal: Regular price behavior (use standard logic)
    """
    
    def __init__(self,
                 spike_velocity_threshold: float = 80.0,
                 spike_accel_threshold: float = 0.0,
                 spike_volatility_threshold: float = 40.0,
                 gradual_velocity_min: float = 20.0,
                 gradual_velocity_max: float = 80.0,
                 gradual_volatility_max: float = 30.0,
                 crash_velocity_threshold: float = -80.0,
                 oscillating_velocity_max: float = 20.0,
                 oscillating_volatility_min: float = 35.0):
        """
        Initialize regime classifier.
        
        Args:
            spike_velocity_threshold: Min velocity for extreme spike ($/MWh per 30min)
            spike_accel_threshold: Min acceleration for extreme spike
            spike_volatility_threshold: Min volatility for extreme spike
            gradual_velocity_min: Min velocity for gradual rise
            gradual_velocity_max: Max velocity for gradual rise
            gradual_volatility_max: Max volatility for gradual rise
            crash_velocity_threshold: Max velocity for extreme crash
            oscillating_velocity_max: Max abs velocity for oscillating
            oscillating_volatility_min: Min volatility for oscillating
        """
        self.spike_velocity_threshold = spike_velocity_threshold
        self.spike_accel_threshold = spike_accel_threshold
        self.spike_volatility_threshold = spike_volatility_threshold
        self.gradual_velocity_min = gradual_velocity_min
        self.gradual_velocity_max = gradual_velocity_max
        self.gradual_volatility_max = gradual_volatility_max
        self.crash_velocity_threshold = crash_velocity_threshold
        self.oscillating_velocity_max = oscillating_velocity_max
        self.oscillating_volatility_min = oscillating_volatility_min
        
        logger.info("Initialized RegimeClassifier")
    
    def classify_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify price regime for each timestamp.
        
        Args:
            df: DataFrame with differential features
            
        Returns:
            DataFrame with 'regime' column added
            
        CRITICAL: No lookahead - uses only current and past features
        """
        df = df.copy()
        
        # Check required features exist
        required_features = ['velocity_30min', 'accel_15min', 'volatility_30min']
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Initialize regime column
        df['regime'] = 'normal'
        
        # 1. Extreme Spike: Fast rise + accelerating + high volatility
        extreme_spike_mask = (
            (df['velocity_30min'] > self.spike_velocity_threshold) &
            (df['accel_15min'] > self.spike_accel_threshold) &
            (df['volatility_30min'] > self.spike_volatility_threshold)
        )
        df.loc[extreme_spike_mask, 'regime'] = 'extreme_spike'
        
        # 2. Gradual Rise: Steady rise + low volatility
        gradual_rise_mask = (
            (df['velocity_30min'] > self.gradual_velocity_min) &
            (df['velocity_30min'] < self.gradual_velocity_max) &
            (df['volatility_30min'] < self.gradual_volatility_max) &
            (df['regime'] == 'normal')  # Don't override extreme_spike
        )
        df.loc[gradual_rise_mask, 'regime'] = 'gradual_rise'
        
        # 3. Extreme Crash: Fast fall
        extreme_crash_mask = (
            (df['velocity_30min'] < self.crash_velocity_threshold) &
            (df['accel_15min'] < 0) &
            (df['regime'] == 'normal')
        )
        df.loc[extreme_crash_mask, 'regime'] = 'extreme_crash'
        
        # 4. Oscillating: Low velocity + high volatility
        oscillating_mask = (
            (df['velocity_30min'].abs() < self.oscillating_velocity_max) &
            (df['volatility_30min'] > self.oscillating_volatility_min) &
            (df['regime'] == 'normal')
        )
        df.loc[oscillating_mask, 'regime'] = 'oscillating'
        
        # Log distribution
        regime_counts = df['regime'].value_counts()
        logger.info(f"Regime distribution:\n{regime_counts}")
        
        return df
    
    def get_regime_stats(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics for each regime type.
        
        Args:
            df: DataFrame with 'regime' column
            
        Returns:
            Dictionary with regime statistics
        """
        if 'regime' not in df.columns:
            raise ValueError("DataFrame must have 'regime' column")
        
        stats = {}
        
        for regime in df['regime'].unique():
            regime_df = df[df['regime'] == regime]
            stats[regime] = {
                'count': len(regime_df),
                'percentage': len(regime_df) / len(df) * 100,
                'avg_price': regime_df['price'].mean(),
                'avg_velocity_30min': regime_df['velocity_30min'].mean() if 'velocity_30min' in regime_df.columns else None,
                'avg_volatility_30min': regime_df['volatility_30min'].mean() if 'volatility_30min' in regime_df.columns else None
            }
        
        return stats

