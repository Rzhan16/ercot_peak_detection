"""
ERCOT Peak Detection - Main Execution Script

This is the master orchestration script for the entire project.
Use this to run different stages of the analysis.
"""

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main execution pipeline for ERCOT Peak Detection project.
    
    Usage:
        python main.py --mode full       # Run complete pipeline
        python main.py --mode eda        # Only exploratory analysis
        python main.py --mode backtest   # Only backtest strategies
        python main.py --mode optimize   # Only parameter optimization
        python main.py --mode report     # Only generate reports
    """
    
    parser = argparse.ArgumentParser(
        description='ERCOT Peak Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode full                    # Run everything
  python main.py --mode eda                     # Just EDA
  python main.py --mode backtest --quick        # Quick backtest (sample data)
  
For detailed implementation guide, see IMPLEMENTATION_GUIDE.md
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'eda', 'backtest', 'optimize', 'report', 'week1', 'enhanced'],
        default='full',
        help='Execution mode (default: full)'
    )
    
    parser.add_argument(
        '--rtm-data',
        default='data/rtm_prices.csv',
        help='Path to real-time market data CSV'
    )
    
    parser.add_argument(
        '--dam-data',
        default='data/dam_prices.csv',
        help='Path to day-ahead market data CSV'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: use subset of data for testing'
    )
    
    args = parser.parse_args()
    
    print_header("ERCOT REAL-TIME PEAK DETECTION SYSTEM")
    print(f"Mode: {args.mode.upper()}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Quick Mode: {'Yes' if args.quick else 'No'}")
    print("="*70 + "\n")
    
    try:
        # Check if data files exist
        rtm_path = Path(args.rtm_data)
        if not rtm_path.exists():
            logger.error(f"RTM data file not found: {args.rtm_data}")
            logger.error("Please place your rtm_prices.csv file in the data/ directory")
            sys.exit(1)
        
        # Execute based on mode
        if args.mode in ['full', 'week1', 'eda']:
            run_eda(args.rtm_data, args.dam_data, args.quick)
        
        if args.mode in ['full', 'week1', 'backtest']:
            run_backtest(args.rtm_data, args.dam_data, args.quick)
        
        if args.mode in ['full', 'optimize']:
            run_optimization(args.rtm_data, args.quick)
        
        if args.mode in ['full', 'report']:
            generate_reports()
        
        if args.mode == 'enhanced':
            run_enhanced_backtest(args.rtm_data, args.dam_data, args.quick)
        
        print("\n" + "="*70)
        print(" EXECUTION COMPLETED SUCCESSFULLY")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n  Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print("\n" + "="*70)
        print(" EXECUTION FAILED")
        print(f"Error: {str(e)}")
        print("="*70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)


def run_eda(rtm_path, dam_path, quick=False):
    """Run exploratory data analysis."""
    print_header("STAGE 1: EXPLORATORY DATA ANALYSIS")
    
    from src.data_loader import ERCOTDataLoader
    from src.eda import ERCOTAnalyzer
    
    logger.info("Loading data...")
    loader = ERCOTDataLoader()
    rtm_df = loader.load_rtm_data(rtm_path)
    peaks_df = loader.get_daily_peaks(rtm_df)
    
    if quick:
        logger.info("Quick mode: Using first 30 days")
        first_date = rtm_df['timestamp'].min().date()
        cutoff_date = first_date + pd.Timedelta(days=30)
        rtm_df = rtm_df[rtm_df['timestamp'].dt.date < cutoff_date]
        peaks_df = peaks_df[peaks_df['date'] < cutoff_date]
    
    logger.info(f"Loaded {len(rtm_df)} RTM records, {len(peaks_df)} days")
    
    logger.info("Running analysis...")
    analyzer = ERCOTAnalyzer()
    results = analyzer.run_full_analysis(rtm_df, peaks_df)
    
    logger.info("✅ EDA Complete - Check results/plots/")


def run_backtest(rtm_path, dam_path, quick=False):
    """Run strategy backtesting."""
    print_header("STAGE 2: STRATEGY BACKTESTING")
    
    from src.data_loader import ERCOTDataLoader
    from src.strategies import PriceDropStrategy, VelocityReversalStrategy, EnsembleStrategy
    from src.backtester import Backtester
    from src.comparator import StrategyComparator
    import pandas as pd
    
    logger.info("Loading data...")
    loader = ERCOTDataLoader()
    rtm_df = loader.load_rtm_data(rtm_path)
    dam_df = loader.load_dam_data(dam_path)
    merged_df = loader.merge_rtm_dam(rtm_df, dam_df)
    peaks_df = loader.get_daily_peaks(rtm_df)
    
    if quick:
        logger.info("Quick mode: Using first 30 days")
        first_date = rtm_df['timestamp'].min().date()
        cutoff_date = first_date + pd.Timedelta(days=30)
        rtm_df = rtm_df[rtm_df['timestamp'].dt.date < cutoff_date]
        merged_df = merged_df[merged_df['timestamp'].dt.date < cutoff_date]
        peaks_df = peaks_df[peaks_df['date'] < cutoff_date]
    
    logger.info(f"Testing on {len(peaks_df)} days...")
    
    # Run all strategies
    logger.info("Testing strategies...")
    strategies = [
        ('PriceDrop (Optimized)', 
         PriceDropStrategy(lookback_minutes=10, drop_threshold=0.035, 
                          min_price_multiplier=1.25, longterm_minutes=120), 
         rtm_df),
        ('VelocityReversal (Optimized)', 
         VelocityReversalStrategy(velocity_window_minutes=15, 
                                 acceleration_threshold=-1.0, 
                                 price_percentile=80, lookback_minutes=60), 
         rtm_df),
        ('Ensemble (2-vote)', 
         EnsembleStrategy([
             PriceDropStrategy(lookback_minutes=10, drop_threshold=0.035),
             VelocityReversalStrategy(velocity_window_minutes=15, acceleration_threshold=-1.0)
         ], min_votes=2), 
         rtm_df)
    ]
    
    results_all = []
    for name, strategy, data in strategies:
        logger.info(f"  • {name}...")
        backtester = Backtester(strategy, peaks_df)
        results = backtester.run_backtest(data, verbose=False)
        results_all.append(results)
        logger.info(f"    Success: {results['success_rate']:.1%}, Precision: {results['precision']:.1%}")
    
    # Generate comparison
    logger.info("Generating comparison report...")
    comparator = StrategyComparator(results_all)
    comparator.plot_comparison()
    report = comparator.generate_report()
    
    logger.info("✅ Backtesting Complete - Check results/plots/")


def run_optimization(rtm_path, quick=False):
    """Run parameter optimization."""
    print_header("STAGE 3: PARAMETER OPTIMIZATION")
    
    if quick:
        logger.warning("Skipping optimization in quick mode (takes too long)")
        return
    
    from src.parameter_tuning import optimize_all_strategies
    
    logger.info("Starting parameter optimization...")
    logger.warning("This may take 10-30 minutes...")
    
    results = optimize_all_strategies()
    
    logger.info(" Optimization Complete")


def generate_reports():
    """Generate final reports."""
    print_header("STAGE 4: REPORT GENERATION")
    
    logger.info("Reports already generated:")
    logger.info("  • FINAL_REPORT.md - Comprehensive project report")
    logger.info("  • PROJECT_COMPLETE.md - Summary & celebration")
    logger.info("  • results/strategy_comparison_report.txt")
    logger.info("  • results/plots/strategy_comparison.png")
    logger.info("  • results/plots/backtest_summary_*.png")
    
    logger.info("\n All reports are ready - Check project root and results/")


def run_enhanced_backtest(rtm_path, dam_path, quick=False):
    """Run enhanced strategies with regime classification and dynamic thresholds."""
    print_header("ENHANCED STRATEGIES: REGIME CLASSIFICATION & DYNAMIC THRESHOLDS")
    
    from src.data_loader import ERCOTDataLoader
    from src.strategies import PriceDropStrategy, VelocityReversalStrategy, EnsembleStrategy
    from src.regime_adaptive_strategy import RegimeAdaptiveStrategy, DynamicThresholdStrategy
    from src.backtester import Backtester
    import json
    
    logger.info("Loading data...")
    loader = ERCOTDataLoader()
    rtm_df = loader.load_rtm_data(rtm_path)
    peaks_df = loader.get_daily_peaks(rtm_df)
    
    if quick:
        logger.info("Quick mode: Using first 60 days")
        first_date = rtm_df['timestamp'].min().date()
        cutoff_date = first_date + pd.Timedelta(days=60)
        rtm_df = rtm_df[rtm_df['timestamp'].dt.date < cutoff_date]
        peaks_df = peaks_df[peaks_df['date'] < cutoff_date]
    
    logger.info(f"Testing on {len(peaks_df)} days...")
    logger.info("")
    
    # Define strategies
    strategies = {
        'Baseline: PriceDrop': PriceDropStrategy(
            lookback_minutes=10,
            drop_threshold=0.035,
            min_price_multiplier=1.25,
            longterm_minutes=120
        ),
        'Baseline: VelocityReversal': VelocityReversalStrategy(
            velocity_window_minutes=15,
            acceleration_threshold=-1.0,
            price_percentile=80,
            lookback_minutes=60
        ),
        'Baseline: Ensemble': EnsembleStrategy([
            PriceDropStrategy(lookback_minutes=10, drop_threshold=0.035, 
                            min_price_multiplier=1.25, longterm_minutes=120),
            VelocityReversalStrategy(velocity_window_minutes=15, 
                                   acceleration_threshold=-1.0,
                                   price_percentile=80, lookback_minutes=60)
        ], min_votes=2),
        'NEW: RegimeAdaptive': RegimeAdaptiveStrategy(),
        'NEW: DynamicThreshold': DynamicThresholdStrategy(),
        'NEW: Enhanced Ensemble': EnsembleStrategy([
            RegimeAdaptiveStrategy(),
            DynamicThresholdStrategy(),
            PriceDropStrategy(lookback_minutes=10, drop_threshold=0.035)
        ], min_votes=2)
    }
    
    # Run backtests
    results = {}
    for name, strategy in strategies.items():
        logger.info(f"Testing: {name}")
        try:
            backtester = Backtester(strategy, peaks_df)
            result = backtester.run_backtest(rtm_df, verbose=False)
            results[name] = result
            
            logger.info(f"  Success: {result['success_rate']:.1%}")
            logger.info(f"  Precision: {result['precision']:.1%}")
            logger.info(f"  Signals/Day: {result['signals_per_day']:.1f}")
            logger.info(f"  ✓ Complete")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            results[name] = None
        logger.info("")
    
    # Print comparison table
    logger.info("="*70)
    logger.info("RESULTS COMPARISON")
    logger.info("="*70)
    print()
    print(f"{'Strategy':<30} {'Success':>10} {'Precision':>10} {'Sig/Day':>10}")
    print("-"*70)
    
    for name, result in results.items():
        if result:
            print(f"{name:<30} {result['success_rate']:>9.1%} {result['precision']:>9.1%} {result['signals_per_day']:>9.1f}")
    
    print()
    
    # Calculate improvements
    if results.get('Baseline: Ensemble') and results.get('NEW: Enhanced Ensemble'):
        baseline = results['Baseline: Ensemble']
        enhanced = results['NEW: Enhanced Ensemble']
        
        success_imp = ((enhanced['success_rate'] - baseline['success_rate']) / baseline['success_rate'] * 100)
        precision_imp = ((enhanced['precision'] - baseline['precision']) / baseline['precision'] * 100)
        
        logger.info("="*70)
        logger.info("IMPROVEMENTS")
        logger.info("="*70)
        logger.info(f"Success Rate:  {success_imp:+.1f}% improvement")
        logger.info(f"Precision:     {precision_imp:+.1f}% improvement")
        logger.info("")
    
    # Save results
    logger.info("Saving results...")
    results_serializable = {}
    for name, result in results.items():
        if result:
            results_serializable[name] = {
                'success_rate': result['success_rate'],
                'precision': result['precision'],
                'signals_per_day': result['signals_per_day'],
                'avg_delay_minutes': result['avg_delay_minutes']
            }
    
    with open('results/enhanced_strategies_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(" Enhanced backtest complete")
    logger.info("   Results saved: results/enhanced_strategies_results.json")


if __name__ == "__main__":
    main()
