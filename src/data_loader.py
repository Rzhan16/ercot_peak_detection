"""Data loading and preprocessing for ERCOT electricity market data."""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERCOTDataLoader:
    """Load and preprocess ERCOT market data."""
    
    def load_rtm_data(self, filepath: str) -> pd.DataFrame:
        """
        Load real-time market data (5-minute intervals).
        
        CSV Format:
        - interval_start_local: e.g., "1/1/24 0:00" or "2024-01-01 00:05:00-06:00"
        - interval_end_local: datetime with timezone
        - sced_timestamp_local: datetime with timezone
        - market: "REAL_TIME_SCED"
        - location: "CSC_CSECG1_2"
        - location_type: "Resource Node"
        - lmp: float (price in $/MWh)
        
        Args:
            filepath: Path to RTM CSV file
            
        Returns:
            DataFrame with columns [timestamp, price], sorted by timestamp
        """
        logger.info(f"Loading RTM data from {filepath}")
        
        try:
            # Load CSV
            df = pd.read_csv(filepath)
            
            # Parse timestamp (auto-detect format)
            df['timestamp'] = pd.to_datetime(df['interval_start_local'])
            
            # Extract price
            df = df[['timestamp', 'lmp']].copy()
            df.columns = ['timestamp', 'price']
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Validate
            if df.empty:
                raise ValueError("No data loaded from RTM file")
            if df['price'].isna().all():
                raise ValueError("All prices are NaN")
            
            logger.info(f"Loaded {len(df)} RTM records from {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except KeyError as e:
            logger.error(f"Missing expected column in RTM data: {e}")
            raise ValueError(f"RTM CSV missing required column: {e}")
        except Exception as e:
            logger.error(f"Error loading RTM data: {e}")
            raise
    
    def load_dam_data(self, filepath: str) -> pd.DataFrame:
        """
        Load day-ahead market data (hourly intervals).
        
        CSV Format:
        - interval_start_local: datetime
        - interval_end_local: datetime with timezone
        - location: "CSC_CSECG1_2"
        - location_type: "Resource Node"
        - market: "DAY_AHEAD_HOURLY"
        - lmp: float (price in $/MWh)
        
        Args:
            filepath: Path to DAM CSV file
            
        Returns:
            DataFrame with columns [timestamp, dam_price]
        """
        logger.info(f"Loading DAM data from {filepath}")
        
        try:
            # Load CSV
            df = pd.read_csv(filepath)
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['interval_start_local'])
            
            # Extract price
            df = df[['timestamp', 'lmp']].copy()
            df.columns = ['timestamp', 'dam_price']
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} DAM records")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except KeyError as e:
            logger.error(f"Missing expected column in DAM data: {e}")
            raise ValueError(f"DAM CSV missing required column: {e}")
        except Exception as e:
            logger.error(f"Error loading DAM data: {e}")
            raise
    
    def get_daily_peaks(self, rtm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find actual daily peak prices (ground truth).
        
        Args:
            rtm_df: Real-time price DataFrame with [timestamp, price]
            
        Returns:
            DataFrame with columns:
            - date: date only
            - peak_price: maximum price of the day
            - peak_time: timestamp of peak
            - peak_hour: hour of day (0-23)
        """
        logger.info("Calculating daily peaks")
        
        df = rtm_df.copy()
        df['date'] = df['timestamp'].dt.date
        
        # Find peak for each day
        peaks = df.groupby('date', group_keys=False).apply(
            lambda day: pd.Series({
                'peak_price': day['price'].max(),
                'peak_time': day.loc[day['price'].idxmax(), 'timestamp'],
                'peak_hour': day.loc[day['price'].idxmax(), 'timestamp'].hour
            }),
            include_groups=False
        ).reset_index()
        
        logger.info(f"Identified peaks for {len(peaks)} days")
        
        return peaks
    
    def merge_rtm_dam(self, rtm_df: pd.DataFrame, dam_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge real-time and day-ahead data.
        
        Match RTM 5-min data with corresponding DAM hourly forecast.
        Uses forward fill to propagate hourly DAM price to 5-min intervals.
        
        Args:
            rtm_df: Real-time DataFrame [timestamp, price]
            dam_df: Day-ahead DataFrame [timestamp, dam_price]
            
        Returns:
            DataFrame with columns [timestamp, price, dam_price]
        """
        logger.info("Merging RTM and DAM data")
        
        # Merge on timestamp (left join, then forward fill)
        merged = rtm_df.copy()
        merged = pd.merge_asof(
            merged.sort_values('timestamp'),
            dam_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        logger.info(f"Merged {len(merged)} records")
        
        return merged
