"""Failure analysis for peak detection strategies."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from pathlib import Path
import logging

sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """Analyze missed peaks and false positives."""
    
    def __init__(self, rtm_df: pd.DataFrame, peaks_df: pd.DataFrame, 
                 backtest_results: Dict, output_dir: str = 'results/plots'):
        """
        Initialize failure analyzer.
        
        Args:
            rtm_df: Real-time price DataFrame
            peaks_df: Daily peaks DataFrame
            backtest_results: Results from backtester
            output_dir: Output directory
        """
        self.rtm_df = rtm_df
        self.peaks_df = peaks_df
        self.backtest_results = backtest_results
        self.daily_results = backtest_results.get('daily_results', [])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FailureAnalyzer for {backtest_results['strategy_name']}")
    
    def analyze_misses(self) -> Dict:
        """
        Analyze days when peak was missed.
        
        Returns:
            Dictionary with miss analysis
        """
        logger.info("Analyzing missed peaks...")
        
        missed_days = [r for r in self.daily_results if not r['success']]
        
        if not missed_days:
            logger.info("No missed days - perfect performance!")
            return {'total_misses': 0}
        
        # Categorize misses
        miss_categories = {
            'no_signals': 0,  # Strategy didn't trigger at all
            'signals_too_early': 0,  # Triggered >5 min before peak
            'signals_too_late': 0,  # Triggered >5 min after peak
            'no_signals_near_peak': 0  # Triggered but not near peak
        }
        
        for day_result in missed_days:
            if len(day_result['signals']) == 0:
                miss_categories['no_signals'] += 1
            else:
                # Check timing of signals relative to peak
                peak_time = day_result['peak_time']
                signals = day_result['signals']
                
                # Find closest signal
                time_diffs = [(s - peak_time).total_seconds() / 60 for s in signals]
                min_diff = min(time_diffs, key=abs)
                
                if min_diff < -5:  # More than 5 min before peak
                    miss_categories['signals_too_early'] += 1
                elif min_diff > 5:  # More than 5 min after peak
                    miss_categories['signals_too_late'] += 1
                else:
                    miss_categories['no_signals_near_peak'] += 1
        
        analysis = {
            'total_misses': len(missed_days),
            'miss_rate': len(missed_days) / len(self.daily_results),
            'categories': miss_categories,
            'missed_dates': [r['date'] for r in missed_days[:10]]  # First 10
        }
        
        logger.info(f"Total misses: {len(missed_days)}/{len(self.daily_results)} ({analysis['miss_rate']:.1%})")
        logger.info(f"Categories: {miss_categories}")
        
        return analysis
    
    def analyze_false_positives(self) -> Dict:
        """
        Analyze false positive patterns.
        
        Returns:
            Dictionary with false positive analysis
        """
        logger.info("Analyzing false positives...")
        
        total_fp = self.backtest_results['false_positives']
        total_signals = self.backtest_results['total_signals']
        
        if total_fp == 0:
            logger.info("No false positives - perfect precision!")
            return {'total_false_positives': 0}
        
        # Calculate FP rate by time of day
        fp_by_hour = {}
        
        for day_result in self.daily_results:
            peak_time = day_result['peak_time']
            signals = day_result['signals']
            
            # Mark signals as FP or TP
            for signal_time in signals:
                signal_hour = signal_time.hour
                
                # Check if this signal is within ±5 min of peak
                time_diff = abs((signal_time - peak_time).total_seconds() / 60)
                
                if time_diff > 5:  # False positive
                    fp_by_hour[signal_hour] = fp_by_hour.get(signal_hour, 0) + 1
        
        analysis = {
            'total_false_positives': total_fp,
            'fp_rate': total_fp / total_signals if total_signals > 0 else 0,
            'fp_by_hour': fp_by_hour,
            'avg_fp_per_day': total_fp / len(self.daily_results)
        }
        
        logger.info(f"Total FP: {total_fp} ({analysis['fp_rate']:.1%} of signals)")
        logger.info(f"Avg FP per day: {analysis['avg_fp_per_day']:.1f}")
        
        return analysis
    
    def generate_report(self) -> str:
        """
        Generate comprehensive failure analysis report.
        
        Returns:
            Report text
        """
        miss_analysis = self.analyze_misses()
        fp_analysis = self.analyze_false_positives()
        
        report = []
        report.append("="*70)
        report.append(f"FAILURE ANALYSIS: {self.backtest_results['strategy_name']}")
        report.append("="*70)
        report.append("")
        
        # Overview
        report.append("📊 PERFORMANCE OVERVIEW")
        report.append("-"*70)
        report.append(f"Success Rate: {self.backtest_results['success_rate']:.1%}")
        report.append(f"Precision: {self.backtest_results['precision']:.1%}")
        report.append(f"Total Days: {self.backtest_results['total_days']}")
        report.append(f"Successful Days: {self.backtest_results['successful_days']}")
        report.append(f"Missed Days: {miss_analysis['total_misses']}")
        report.append("")
        
        # Missed peaks analysis
        report.append("❌ MISSED PEAKS ANALYSIS")
        report.append("-"*70)
        report.append(f"Total Misses: {miss_analysis['total_misses']} ({miss_analysis.get('miss_rate', 0):.1%})")
        
        if miss_analysis['total_misses'] > 0:
            categories = miss_analysis['categories']
            report.append("")
            report.append("Miss Categories:")
            report.append(f"  • No signals at all: {categories['no_signals']}")
            report.append(f"  • Signals too early (>5min before): {categories['signals_too_early']}")
            report.append(f"  • Signals too late (>5min after): {categories['signals_too_late']}")
            report.append(f"  • No signals near peak: {categories['no_signals_near_peak']}")
            
            if miss_analysis.get('missed_dates'):
                report.append("")
                report.append("Sample Missed Dates:")
                for date in miss_analysis['missed_dates'][:5]:
                    report.append(f"  • {date}")
        
        report.append("")
        
        # False positives analysis
        report.append("⚠️  FALSE POSITIVES ANALYSIS")
        report.append("-"*70)
        report.append(f"Total False Positives: {fp_analysis['total_false_positives']}")
        report.append(f"FP Rate: {fp_analysis.get('fp_rate', 0):.1%} of all signals")
        report.append(f"Avg FP per Day: {fp_analysis.get('avg_fp_per_day', 0):.1f}")
        
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-"*70)
        
        if miss_analysis.get('categories', {}).get('no_signals', 0) > 10:
            report.append("• Strategy is too conservative - consider loosening thresholds")
        
        if miss_analysis.get('categories', {}).get('signals_too_early', 0) > 10:
            report.append("• Many early signals - consider adding confirmation delay")
        
        if fp_analysis.get('fp_rate', 0) > 0.5:
            report.append("• High false positive rate - tighten trigger conditions")
            report.append("• Consider ensemble voting to reduce FPs")
        
        if fp_analysis.get('avg_fp_per_day', 0) > 20:
            report.append("• Too many daily signals - increase thresholds significantly")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)
    
    def plot_failure_patterns(self, save: bool = True) -> plt.Figure:
        """
        Visualize failure patterns.
        
        Args:
            save: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Success/Failure by month
        ax1 = axes[0, 0]
        success_by_month = {}
        for result in self.daily_results:
            month = pd.to_datetime(result['date']).month
            if month not in success_by_month:
                success_by_month[month] = {'success': 0, 'total': 0}
            success_by_month[month]['total'] += 1
            if result['success']:
                success_by_month[month]['success'] += 1
        
        months = sorted(success_by_month.keys())
        success_rates = [success_by_month[m]['success'] / success_by_month[m]['total'] * 100 for m in months]
        
        ax1.bar(months, success_rates, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_title('Success Rate by Month', fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_xticks(range(1, 13))
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Signals per day distribution
        ax2 = axes[0, 1]
        signals_per_day = [len(r['signals']) for r in self.daily_results]
        ax2.hist(signals_per_day, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(signals_per_day), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(signals_per_day):.1f}')
        ax2.set_title('Signals Per Day Distribution', fontweight='bold')
        ax2.set_xlabel('Number of Signals')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Miss categories pie chart
        ax3 = axes[1, 0]
        miss_analysis = self.analyze_misses()
        if miss_analysis['total_misses'] > 0:
            categories = miss_analysis['categories']
            labels = ['No Signals', 'Too Early', 'Too Late', 'Not Near Peak']
            sizes = [categories['no_signals'], categories['signals_too_early'],
                     categories['signals_too_late'], categories['no_signals_near_peak']]
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 10})
            ax3.set_title('Miss Categories', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Misses!', ha='center', va='center', fontsize=16)
            ax3.set_title('Miss Categories', fontweight='bold')
        
        # 4. Success vs failure summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Days', str(len(self.daily_results))],
            ['Successful Days', str(self.backtest_results['successful_days'])],
            ['Missed Days', str(miss_analysis['total_misses'])],
            ['Success Rate', f"{self.backtest_results['success_rate']:.1%}"],
            ['Total Signals', str(self.backtest_results['total_signals'])],
            ['False Positives', str(self.backtest_results['false_positives'])],
            ['Precision', f"{self.backtest_results['precision']:.1%}"]
        ]
        
        table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                          colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle(f'Failure Analysis: {self.backtest_results["strategy_name"]}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filename = f'failure_analysis_{self.backtest_results["strategy_name"]}.png'
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved failure analysis: {filename}")
        
        return fig

