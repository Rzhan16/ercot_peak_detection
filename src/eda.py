"""Exploratory data analysis for ERCOT price data."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import logging
from pathlib import Path

sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERCOTAnalyzer:
    """Exploratory analysis of ERCOT price data."""
    
    def __init__(self, output_dir: str = 'results/plots'):
        """
        Initialize analyzer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ERCOTAnalyzer, saving plots to {self.output_dir}")
    
    def analyze_peak_timing(self, peaks_df: pd.DataFrame) -> Dict:
        """
        Analyze when peaks occur during the day.
        
        Args:
            peaks_df: DataFrame with [date, peak_time, peak_price, peak_hour]
            
        Returns:
            Dictionary with timing statistics
        """
        logger.info("Analyzing peak timing patterns")
        
        # Count peaks by hour
        hour_counts = peaks_df['peak_hour'].value_counts().sort_index()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart of peaks by hour
        hour_counts.plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black', alpha=0.8)
        ax1.set_title('Peak Frequency by Hour of Day', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Days')
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart of top 5 hours
        top_hours = hour_counts.nlargest(5)
        colors = sns.color_palette('Set2', len(top_hours))
        ax2.pie(top_hours.values, labels=[f'{h}:00' for h in top_hours.index], 
                autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Top 5 Peak Hours', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'peak_timing_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        stats = {
            'most_common_hour': int(hour_counts.idxmax()),
            'most_common_hour_count': int(hour_counts.max()),
            'peak_concentration_14_18': int(hour_counts.loc[14:18].sum()) if len(hour_counts.loc[14:18]) > 0 else 0,
            'total_days': len(peaks_df)
        }
        
        stats['peak_concentration_14_18_pct'] = 100 * stats['peak_concentration_14_18'] / stats['total_days'] if stats['total_days'] > 0 else 0
        
        logger.info(f"Most common peak hour: {stats['most_common_hour']}:00 ({stats['most_common_hour_count']} days)")
        logger.info(f"Peaks 2-6 PM: {stats['peak_concentration_14_18_pct']:.1f}%")
        
        return stats
    
    def seasonal_patterns(self, rtm_df: pd.DataFrame, peaks_df: pd.DataFrame) -> Dict:
        """
        Analyze seasonal price patterns.
        
        Args:
            rtm_df: Real-time price DataFrame
            peaks_df: Daily peaks DataFrame
            
        Returns:
            Dictionary with seasonal statistics
        """
        logger.info("Analyzing seasonal patterns")
        
        # Add month column
        rtm_df = rtm_df.copy()
        rtm_df['month'] = rtm_df['timestamp'].dt.month
        
        peaks_df = peaks_df.copy()
        peaks_df['month'] = pd.to_datetime(peaks_df['date']).dt.month
        
        # Monthly statistics
        monthly_avg_price = rtm_df.groupby('month')['price'].mean()
        monthly_peak_price = peaks_df.groupby('month')['peak_price'].mean()
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Line chart: Monthly average prices
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        x = range(1, 13)
        ax1.plot(x, monthly_avg_price.values, marker='o', linewidth=2, 
                label='Avg Price', color='steelblue', markersize=8)
        ax1.plot(x, monthly_peak_price.values, marker='s', linewidth=2, 
                label='Avg Peak Price', color='coral', markersize=8)
        ax1.set_title('Monthly Price Patterns (2024)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Price ($/MWh)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(months)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Box plot: Price distribution by month
        # Sample data for performance (max 1000 points per month)
        rtm_sample = rtm_df.groupby('month', group_keys=False).apply(
            lambda x: x.sample(n=min(1000, len(x)), random_state=42),
            include_groups=False
        )
        
        box_data = [rtm_sample[rtm_sample['month'] == m]['price'].values for m in range(1, 13)]
        bp = ax2.boxplot(box_data, labels=months, patch_artist=True)
        
        # Color boxes
        colors = sns.color_palette('Set2', 12)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('Price Distribution by Month', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Price ($/MWh)')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        stats = {
            'highest_month': months[monthly_peak_price.idxmax() - 1],
            'highest_month_price': float(monthly_peak_price.max()),
            'lowest_month': months[monthly_peak_price.idxmin() - 1],
            'lowest_month_price': float(monthly_peak_price.min())
        }
        
        logger.info(f"Highest prices: {stats['highest_month']} (${stats['highest_month_price']:.2f}/MWh)")
        logger.info(f"Lowest prices: {stats['lowest_month']} (${stats['lowest_month_price']:.2f}/MWh)")
        
        return stats
    
    def price_volatility(self, rtm_df: pd.DataFrame) -> Dict:
        """
        Analyze daily price volatility.
        
        Args:
            rtm_df: Real-time price DataFrame
            
        Returns:
            Dictionary with volatility statistics
        """
        logger.info("Analyzing price volatility")
        
        df = rtm_df.copy()
        df['date'] = df['timestamp'].dt.date
        
        # Calculate daily volatility metrics
        daily_stats = df.groupby('date')['price'].agg([
            ('daily_range', lambda x: x.max() - x.min()),
            ('daily_std', 'std'),
            ('daily_mean', 'mean'),
            ('daily_max', 'max')
        ]).reset_index()
        
        # Identify spike days (price > 2 std above mean)
        overall_mean = df['price'].mean()
        overall_std = df['price'].std()
        spike_threshold = overall_mean + 2 * overall_std
        
        daily_stats['has_spike'] = daily_stats['daily_max'] > spike_threshold
        
        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Daily price range
        ax1.plot(range(len(daily_stats)), daily_stats['daily_range'], 
                linewidth=1, alpha=0.7, color='steelblue')
        ax1.set_title('Daily Price Range', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Day of Year')
        ax1.set_ylabel('Range ($/MWh)')
        ax1.grid(alpha=0.3)
        
        # Daily std deviation
        ax2.plot(range(len(daily_stats)), daily_stats['daily_std'], 
                linewidth=1, color='coral', alpha=0.7)
        ax2.set_title('Daily Price Std Deviation', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Day of Year')
        ax2.set_ylabel('Std Dev ($/MWh)')
        ax2.grid(alpha=0.3)
        
        # Histogram of daily range
        ax3.hist(daily_stats['daily_range'], bins=30, edgecolor='black', 
                alpha=0.7, color='steelblue')
        ax3.set_title('Distribution of Daily Price Ranges', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Daily Range ($/MWh)')
        ax3.set_ylabel('Frequency')
        ax3.grid(alpha=0.3)
        
        # Spike days by month
        daily_stats['month'] = pd.to_datetime(daily_stats['date']).dt.month
        spike_counts = daily_stats.groupby('month')['has_spike'].sum()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax4.bar(range(1, 13), [spike_counts.get(i, 0) for i in range(1, 13)], 
               color='crimson', alpha=0.7, edgecolor='black')
        ax4.set_title('Price Spike Days by Month', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Number of Spike Days')
        ax4.set_xticks(range(1, 13))
        ax4.set_xticklabels(months)
        ax4.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'volatility_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        stats = {
            'avg_daily_range': float(daily_stats['daily_range'].mean()),
            'avg_daily_std': float(daily_stats['daily_std'].mean()),
            'spike_days': int(daily_stats['has_spike'].sum()),
            'total_days': len(daily_stats),
            'spike_threshold': float(spike_threshold)
        }
        
        logger.info(f"Avg daily range: ${stats['avg_daily_range']:.2f}")
        logger.info(f"Spike days: {stats['spike_days']}/{stats['total_days']} ({100*stats['spike_days']/stats['total_days']:.1f}%)")
        
        return stats
    
    def weekday_vs_weekend(self, rtm_df: pd.DataFrame, peaks_df: pd.DataFrame) -> Dict:
        """
        Compare weekday vs weekend patterns.
        
        Args:
            rtm_df: Real-time price DataFrame
            peaks_df: Daily peaks DataFrame
            
        Returns:
            Dictionary with comparison statistics
        """
        logger.info("Analyzing weekday vs weekend patterns")
        
        # Add day of week
        rtm_df = rtm_df.copy()
        rtm_df['is_weekend'] = rtm_df['timestamp'].dt.dayofweek >= 5
        
        peaks_df = peaks_df.copy()
        peaks_df['is_weekend'] = pd.to_datetime(peaks_df['date']).dt.dayofweek >= 5
        
        # Calculate statistics
        weekday_avg_price = rtm_df[~rtm_df['is_weekend']]['price'].mean()
        weekend_avg_price = rtm_df[rtm_df['is_weekend']]['price'].mean()
        
        weekday_avg_peak = peaks_df[~peaks_df['is_weekend']]['peak_price'].mean()
        weekend_avg_peak = peaks_df[peaks_df['is_weekend']]['peak_price'].mean()
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart: Average prices
        categories = ['Weekday\nAvg Price', 'Weekend\nAvg Price', 
                     'Weekday\nPeak Price', 'Weekend\nPeak Price']
        values = [weekday_avg_price, weekend_avg_price, weekday_avg_peak, weekend_avg_peak]
        colors = ['steelblue', 'lightblue', 'coral', 'lightsalmon']
        
        bars = ax1.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_title('Weekday vs Weekend Price Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($/MWh)')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.1f}',
                    ha='center', va='bottom', fontweight='bold')
        
        # Box plot: Peak hour distribution
        weekday_hours = peaks_df[~peaks_df['is_weekend']]['peak_hour']
        weekend_hours = peaks_df[peaks_df['is_weekend']]['peak_hour']
        
        bp = ax2.boxplot([weekday_hours, weekend_hours], labels=['Weekday', 'Weekend'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('lightblue')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        ax2.set_title('Peak Hour Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Hour of Day')
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weekday_weekend_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        stats = {
            'weekday_avg_price': float(weekday_avg_price),
            'weekend_avg_price': float(weekend_avg_price),
            'weekday_avg_peak': float(weekday_avg_peak),
            'weekend_avg_peak': float(weekend_avg_peak),
            'peak_price_diff_pct': 100 * (weekday_avg_peak - weekend_avg_peak) / weekend_avg_peak if weekend_avg_peak > 0 else 0
        }
        
        logger.info(f"Weekday avg peak: ${stats['weekday_avg_peak']:.2f}")
        logger.info(f"Weekend avg peak: ${stats['weekend_avg_peak']:.2f}")
        logger.info(f"Difference: {stats['peak_price_diff_pct']:.1f}% higher on weekdays")
        
        return stats
    
    def run_full_analysis(self, rtm_df: pd.DataFrame, peaks_df: pd.DataFrame) -> Dict:
        """
        Run all analyses and generate summary report.
        
        Args:
            rtm_df: Real-time price DataFrame
            peaks_df: Daily peaks DataFrame
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("="*60)
        logger.info("RUNNING FULL EDA")
        logger.info("="*60)
        
        results = {}
        
        results['timing'] = self.analyze_peak_timing(peaks_df)
        results['seasonal'] = self.seasonal_patterns(rtm_df, peaks_df)
        results['volatility'] = self.price_volatility(rtm_df)
        results['weekday_weekend'] = self.weekday_vs_weekend(rtm_df, peaks_df)
        
        logger.info("="*60)
        logger.info("EDA COMPLETE")
        logger.info(f"Generated plots in: {self.output_dir}")
        logger.info("="*60)
        
        return results
