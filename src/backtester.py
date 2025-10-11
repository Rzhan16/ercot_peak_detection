"""Backtesting framework for peak detection strategies."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import timedelta
import logging
from tqdm import tqdm
from pathlib import Path

from src.strategies import BaseStrategy

sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtest trading strategies on historical data.
    
    Evaluates if strategy signals occur within ±5 minutes of actual daily peaks.
    """
    
    def __init__(self, strategy: BaseStrategy, peaks_df: pd.DataFrame, output_dir: str = 'results/plots'):
        """
        Initialize backtester.
        
        Args:
            strategy: Strategy object with generate_signals() method
            peaks_df: Ground truth peaks [date, peak_time, peak_price]
            output_dir: Directory for saving plots
        """
        self.strategy = strategy
        self.peaks_df = peaks_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = None
        
        logger.info(f"Initialized backtester for {strategy.name}")
    
    def run_backtest(self, rtm_df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Run backtest day by day.
        
        For each day:
        1. Extract that day's data
        2. Generate signals (simulating real-time)
        3. Check if any signal within ±5 min of peak
        4. Record results
        
        Args:
            rtm_df: Real-time price DataFrame
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Running backtest for {self.strategy.name}")
        logger.info(f"Total days: {len(self.peaks_df)}")
        
        daily_results = []
        
        # Group by date
        rtm_df = rtm_df.copy()
        rtm_df['date'] = rtm_df['timestamp'].dt.date
        
        # Process day by day
        peaks_list = list(self.peaks_df.iterrows())
        if verbose:
            peaks_list = tqdm(peaks_list, desc=f"Backtesting {self.strategy.name}")
        
        for idx, peak_row in peaks_list:
            date = peak_row['date']
            peak_time = peak_row['peak_time']
            
            # Get that day's data
            day_data = rtm_df[rtm_df['date'] == date].copy()
            
            if day_data.empty:
                logger.warning(f"No data for {date}, skipping")
                continue
            
            # Generate signals for this day
            try:
                day_signals = self.strategy.generate_signals(day_data)
            except Exception as e:
                logger.error(f"Error generating signals for {date}: {e}")
                continue
            
            # Evaluate this day
            day_result = self._evaluate_day(day_signals, peak_time)
            day_result['date'] = date
            daily_results.append(day_result)
        
        # Calculate overall metrics
        self.results = self._calculate_metrics(daily_results)
        self.results['daily_results'] = daily_results
        
        if verbose:
            self._print_results()
        
        return self.results
    
    def _evaluate_day(self, day_data: pd.DataFrame, peak_time: pd.Timestamp) -> Dict:
        """
        Evaluate strategy performance for one day.
        
        Args:
            day_data: Day's data with 'signal' column
            peak_time: Actual peak timestamp
            
        Returns:
            Dictionary with day's evaluation
        """
        # Find all signal times
        signals = day_data[day_data['signal'] == 1]
        signal_times = signals['timestamp'].tolist()
        
        if len(signal_times) == 0:
            # No signals generated
            return {
                'peak_time': peak_time,
                'signals': [],
                'success': False,
                'delay_minutes': None,
                'false_positives': 0
            }
        
        # Check if any signal within ±5 minutes of peak
        success_window_start = peak_time - timedelta(minutes=5)
        success_window_end = peak_time + timedelta(minutes=5)
        
        successful_signals = [
            t for t in signal_times 
            if success_window_start <= t <= success_window_end
        ]
        
        success = len(successful_signals) > 0
        
        # Calculate delay (for first successful signal)
        if success:
            first_successful = successful_signals[0]
            delay_minutes = (first_successful - peak_time).total_seconds() / 60
        else:
            delay_minutes = None
        
        # Count false positives (signals outside success window)
        false_positives = len([t for t in signal_times if t not in successful_signals])
        
        return {
            'peak_time': peak_time,
            'signals': signal_times,
            'success': success,
            'delay_minutes': delay_minutes,
            'false_positives': false_positives
        }
    
    def _calculate_metrics(self, daily_results: List[Dict]) -> Dict:
        """
        Calculate overall performance metrics.
        
        Args:
            daily_results: List of per-day evaluation results
            
        Returns:
            Dictionary with aggregate metrics
        """
        total_days = len(daily_results)
        successful_days = sum(1 for r in daily_results if r['success'])
        total_signals = sum(len(r['signals']) for r in daily_results)
        total_false_positives = sum(r['false_positives'] for r in daily_results)
        
        # Calculate delays for successful days
        delays = [r['delay_minutes'] for r in daily_results if r['delay_minutes'] is not None]
        avg_delay = np.mean(np.abs(delays)) if delays else None
        
        # Calculate precision (true positives / all signals)
        true_positives = successful_days
        precision = true_positives / total_signals if total_signals > 0 else 0
        
        return {
            'strategy_name': self.strategy.name,
            'strategy_params': self.strategy.get_params(),
            'total_days': total_days,
            'successful_days': successful_days,
            'success_rate': successful_days / total_days if total_days > 0 else 0,
            'total_signals': total_signals,
            'false_positives': total_false_positives,
            'precision': precision,
            'avg_delay_minutes': avg_delay,
            'avg_signals_per_day': total_signals / total_days if total_days > 0 else 0
        }
    
    def _print_results(self):
        """Print results to console."""
        r = self.results
        
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS: {r['strategy_name']}")
        print("="*60)
        print(f"Success Rate:     {r['success_rate']:.1%}")
        print(f"Precision:        {r['precision']:.1%}")
        print(f"Total Days:       {r['total_days']}")
        print(f"Successful Days:  {r['successful_days']}")
        print(f"Total Signals:    {r['total_signals']}")
        print(f"False Positives:  {r['false_positives']}")
        if r['avg_delay_minutes'] is not None:
            print(f"Avg Delay:        {r['avg_delay_minutes']:.2f} minutes")
        else:
            print(f"Avg Delay:        N/A")
        print(f"Signals/Day:      {r['avg_signals_per_day']:.2f}")
        print("="*60 + "\n")
    
    def visualize_day(self, rtm_df: pd.DataFrame, date, save: bool = True) -> plt.Figure:
        """
        Visualize strategy performance for a specific day.
        
        Args:
            rtm_df: Real-time price DataFrame
            date: Date to visualize
            save: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get day's data
        if isinstance(date, str):
            date = pd.to_datetime(date).date()
        
        day_data = rtm_df[rtm_df['timestamp'].dt.date == date].copy()
        
        if day_data.empty:
            logger.warning(f"No data for {date}")
            return None
        
        # Generate signals
        day_signals = self.strategy.generate_signals(day_data)
        
        # Get peak for this day
        peak_row = self.peaks_df[self.peaks_df['date'] == date]
        if peak_row.empty:
            logger.warning(f"No peak data for {date}")
            return None
        
        peak_time = peak_row.iloc[0]['peak_time']
        peak_price = peak_row.iloc[0]['peak_price']
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot price curve
        ax.plot(day_signals['timestamp'], day_signals['price'], 
                linewidth=2, label='Price', color='steelblue', alpha=0.8)
        
        # Plot actual peak
        ax.scatter([peak_time], [peak_price], 
                   s=200, c='red', marker='*', label='Actual Peak', zorder=5)
        
        # Highlight success window (±5 min)
        success_start = peak_time - timedelta(minutes=5)
        success_end = peak_time + timedelta(minutes=5)
        ax.axvspan(success_start, success_end, alpha=0.2, color='green', 
                   label='Success Window (±5 min)')
        
        # Plot trigger points
        triggers = day_signals[day_signals['signal'] == 1]
        if not triggers.empty:
            ax.scatter(triggers['timestamp'], triggers['price'], 
                       s=100, c='blue', marker='o', label='Trigger Signals', zorder=4)
        
        # Labels and title
        ax.set_title(f'{self.strategy.name} Performance - {date}', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price ($/MWh)')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f'backtest_{self.strategy.name}_{date}.png'
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization: {filename}")
        
        return fig
    
    def plot_summary(self, save: bool = True) -> plt.Figure:
        """
        Plot backtest summary statistics.
        
        Args:
            save: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.results is None:
            raise ValueError("Must run backtest first")
        
        daily_results = self.results['daily_results']
        
        # Extract data for plotting
        dates = [r['date'] for r in daily_results]
        successes = [1 if r['success'] else 0 for r in daily_results]
        num_signals = [len(r['signals']) for r in daily_results]
        delays = [r['delay_minutes'] for r in daily_results if r['delay_minutes'] is not None]
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Success rate by month
        ax1 = fig.add_subplot(gs[0, :])
        df_temp = pd.DataFrame({'date': dates, 'success': successes})
        df_temp['month'] = pd.to_datetime(df_temp['date']).dt.month
        monthly_success = df_temp.groupby('month')['success'].mean() * 100
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax1.bar(range(1, 13), [monthly_success.get(i, 0) for i in range(1, 13)], 
                color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axhline(self.results['success_rate'] * 100, color='red', 
                    linestyle='--', linewidth=2, label=f'Overall: {self.results["success_rate"]:.1%}')
        ax1.set_title(f'Success Rate by Month - {self.strategy.name}', 
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(months)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Number of signals per day histogram
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(num_signals, bins=20, edgecolor='black', alpha=0.7, color='coral')
        ax2.axvline(np.mean(num_signals), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(num_signals):.1f}')
        ax2.set_title('Signals Per Day Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Signals')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Delay distribution (for successful triggers)
        ax3 = fig.add_subplot(gs[1, 1])
        if delays:
            ax3.hist(delays, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
            ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Exact Peak')
            ax3.set_title('Signal Delay Distribution', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Delay from Peak (minutes)')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No successful signals', 
                     ha='center', va='center', fontsize=14)
            ax3.set_title('Signal Delay Distribution', fontsize=12, fontweight='bold')
        
        # 4. Key metrics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Success Rate', f"{self.results['success_rate']:.1%}"],
            ['Precision', f"{self.results['precision']:.1%}"],
            ['Total Days', str(self.results['total_days'])],
            ['Successful Days', str(self.results['successful_days'])],
            ['Total Signals', str(self.results['total_signals'])],
            ['False Positives', str(self.results['false_positives'])],
            ['Avg Delay', f"{self.results['avg_delay_minutes']:.2f} min" if self.results['avg_delay_minutes'] else "N/A"],
            ['Signals/Day', f"{self.results['avg_signals_per_day']:.2f}"]
        ]
        
        table = ax4.table(cellText=metrics_data, cellLoc='left', loc='center',
                          colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle(f'Backtest Summary: {self.strategy.name}', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            filename = f'backtest_summary_{self.strategy.name}.png'
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved summary: {filename}")
        
        return fig
