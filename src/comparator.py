"""Strategy comparison and reporting tools."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from pathlib import Path
import logging

sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyComparator:
    """Compare multiple strategy backtest results."""
    
    def __init__(self, results_list: List[Dict], output_dir: str = 'results/plots'):
        """
        Initialize comparator.
        
        Args:
            results_list: List of backtest result dictionaries
            output_dir: Directory for saving plots
        """
        self.results_list = results_list
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized comparator with {len(results_list)} strategies")
    
    def plot_comparison(self, save: bool = True) -> plt.Figure:
        """
        Create comprehensive comparison visualization.
        
        Args:
            save: Whether to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Extract data
        names = [r['strategy_name'] for r in self.results_list]
        success_rates = [r['success_rate'] * 100 for r in self.results_list]
        precisions = [r['precision'] * 100 for r in self.results_list]
        signals_per_day = [r['avg_signals_per_day'] for r in self.results_list]
        false_positives = [r['false_positives'] for r in self.results_list]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        # 1. Success Rate Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(names, success_rates, color=colors, edgecolor='black', alpha=0.7)
        ax1.set_title('Success Rate by Strategy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Rotate labels
        ax1.set_xticklabels(names, rotation=45, ha='right')
        
        # 2. Precision Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(names, precisions, color=colors, edgecolor='black', alpha=0.7)
        ax2.set_title('Precision by Strategy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Precision (%)')
        ax2.set_ylim(0, max(precisions) * 1.2 if precisions else 10)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars2, precisions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xticklabels(names, rotation=45, ha='right')
        
        # 3. Signals Per Day
        ax3 = fig.add_subplot(gs[1, 0])
        bars3 = ax3.bar(names, signals_per_day, color=colors, edgecolor='black', alpha=0.7)
        ax3.set_title('Average Signals Per Day', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Signals/Day')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars3, signals_per_day):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax3.set_xticklabels(names, rotation=45, ha='right')
        
        # 4. False Positives
        ax4 = fig.add_subplot(gs[1, 1])
        bars4 = ax4.bar(names, false_positives, color=colors, edgecolor='black', alpha=0.7)
        ax4.set_title('Total False Positives', fontsize=14, fontweight='bold')
        ax4.set_ylabel('False Positives')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars4, false_positives):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax4.set_xticklabels(names, rotation=45, ha='right')
        
        # 5. Success Rate vs Precision Scatter
        ax5 = fig.add_subplot(gs[2, 0])
        for i, (name, success, precision) in enumerate(zip(names, success_rates, precisions)):
            ax5.scatter(success, precision, s=300, color=colors[i], 
                       edgecolor='black', linewidth=2, alpha=0.7, label=name)
        
        ax5.set_xlabel('Success Rate (%)', fontweight='bold')
        ax5.set_ylabel('Precision (%)', fontweight='bold')
        ax5.set_title('Success Rate vs Precision Trade-off', fontsize=14, fontweight='bold')
        ax5.legend(loc='best')
        ax5.grid(alpha=0.3)
        
        # Add diagonal reference line
        ax5.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect Balance')
        
        # 6. Summary Table
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        # Create summary data
        summary_data = [['Strategy', 'Success', 'Precision', 'Signals/Day']]
        for r in self.results_list:
            summary_data.append([
                r['strategy_name'],
                f"{r['success_rate']:.1%}",
                f"{r['precision']:.1%}",
                f"{r['avg_signals_per_day']:.1f}"
            ])
        
        table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                          colWidths=[0.35, 0.2, 0.2, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code rows by performance
        best_idx = success_rates.index(max(success_rates)) + 1
        table[(best_idx, 0)].set_facecolor('#90EE90')
        table[(best_idx, 1)].set_facecolor('#90EE90')
        table[(best_idx, 2)].set_facecolor('#90EE90')
        table[(best_idx, 3)].set_facecolor('#90EE90')
        
        plt.suptitle('Strategy Performance Comparison', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        if save:
            filename = 'strategy_comparison.png'
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison: {filename}")
        
        return fig
    
    def generate_report(self, save: bool = True) -> str:
        """
        Generate text report of comparison.
        
        Args:
            save: Whether to save report to file
            
        Returns:
            Report text
        """
        report = []
        report.append("="*80)
        report.append("ERCOT PEAK DETECTION - STRATEGY COMPARISON REPORT")
        report.append("="*80)
        report.append("")
        
        # Summary statistics
        report.append("📊 OVERALL SUMMARY")
        report.append("-"*80)
        report.append(f"Total Strategies Evaluated: {len(self.results_list)}")
        report.append(f"Test Period: 365 days (full year 2024)")
        report.append(f"Success Window: ±5 minutes from actual peak")
        report.append("")
        
        # Individual strategy results
        report.append("🎯 STRATEGY PERFORMANCE")
        report.append("-"*80)
        
        # Sort by success rate
        sorted_results = sorted(self.results_list, key=lambda x: x['success_rate'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            report.append(f"\n{i}. {result['strategy_name']}")
            report.append(f"   {'─'*70}")
            report.append(f"   Success Rate:       {result['success_rate']:>6.1%}  ({result['successful_days']}/{result['total_days']} days)")
            report.append(f"   Precision:          {result['precision']:>6.1%}  (true positives / all signals)")
            report.append(f"   Total Signals:      {result['total_signals']:>6,}")
            report.append(f"   False Positives:    {result['false_positives']:>6,}")
            report.append(f"   Signals Per Day:    {result['avg_signals_per_day']:>6.2f}")
            if result['avg_delay_minutes'] is not None:
                report.append(f"   Avg Delay:          {result['avg_delay_minutes']:>6.2f} minutes")
            else:
                report.append(f"   Avg Delay:          {'N/A':>6}")
            report.append(f"   Parameters:         {result['strategy_params']}")
        
        # Key insights
        report.append("")
        report.append("💡 KEY INSIGHTS")
        report.append("-"*80)
        
        best_success = max(sorted_results, key=lambda x: x['success_rate'])
        best_precision = max(sorted_results, key=lambda x: x['precision'])
        
        report.append(f"• Highest Success Rate: {best_success['strategy_name']} ({best_success['success_rate']:.1%})")
        report.append(f"• Highest Precision: {best_precision['strategy_name']} ({best_precision['precision']:.1%})")
        
        # Identify trade-offs
        report.append(f"\n• Trade-off Analysis:")
        for result in sorted_results:
            if result['success_rate'] > 0.7:
                if result['precision'] < 0.05:
                    report.append(f"  - {result['strategy_name']}: High success but LOW precision (too many signals)")
                else:
                    report.append(f"  - {result['strategy_name']}: Balanced performance")
        
        # Recommendations
        report.append("")
        report.append("🔧 RECOMMENDATIONS FOR IMPROVEMENT")
        report.append("-"*80)
        report.append("1. Parameter Optimization: Fine-tune thresholds to reduce false positives")
        report.append("2. Ensemble Method: Combine multiple strategies for better precision")
        report.append("3. Time-of-Day Filters: Apply different parameters for different times")
        report.append("4. Seasonal Adjustments: Adapt strategies to summer vs winter patterns")
        report.append("")
        
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        if save:
            report_path = self.output_dir.parent / 'strategy_comparison_report.txt'
            with open(report_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Saved report: {report_path}")
        
        return report_text

