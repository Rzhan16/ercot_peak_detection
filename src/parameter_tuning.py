"""Parameter optimization for trading strategies."""

import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm
import json
from pathlib import Path

from src.strategies import BaseStrategy
from src.backtester import Backtester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """Optimize strategy parameters using grid search."""
    
    def __init__(self, rtm_df: pd.DataFrame, peaks_df: pd.DataFrame, 
                 strategy_class, output_dir: str = 'results'):
        """
        Initialize parameter optimizer.
        
        Args:
            rtm_df: Real-time price DataFrame
            peaks_df: Daily peaks DataFrame
            strategy_class: Strategy class to optimize
            output_dir: Output directory
        """
        self.rtm_df = rtm_df
        self.peaks_df = peaks_df
        self.strategy_class = strategy_class
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ParameterOptimizer for {strategy_class.__name__}")
    
    def grid_search(self, param_grid: Dict[str, List], 
                    metric: str = 'success_rate',
                    top_n: int = 5) -> Tuple[Dict, List[Dict]]:
        """
        Grid search over parameter space.
        
        Args:
            param_grid: Dictionary of {param_name: [values to test]}
            metric: Metric to optimize ('success_rate' or 'precision')
            top_n: Number of top results to return
            
        Returns:
            Tuple of (best_params, all_results)
        """
        logger.info("="*60)
        logger.info(f"GRID SEARCH: {self.strategy_class.__name__}")
        logger.info("="*60)
        logger.info(f"Parameter grid: {param_grid}")
        logger.info(f"Optimizing for: {metric}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Total combinations: {len(combinations)}")
        
        results = []
        
        # Test each combination
        for combo in tqdm(combinations, desc="Testing combinations"):
            params = dict(zip(param_names, combo))
            
            try:
                # Create strategy with these parameters
                strategy = self.strategy_class(**params)
                
                # Run backtest
                backtester = Backtester(strategy, self.peaks_df)
                backtest_results = backtester.run_backtest(self.rtm_df, verbose=False)
                
                # Store results
                result = {
                    'params': params,
                    'success_rate': backtest_results['success_rate'],
                    'precision': backtest_results['precision'],
                    'total_signals': backtest_results['total_signals'],
                    'avg_delay': backtest_results['avg_delay_minutes']
                }
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                continue
        
        if not results:
            raise ValueError("No successful parameter combinations found")
        
        # Sort by metric
        results_sorted = sorted(results, key=lambda x: x[metric], reverse=True)
        
        # Get best parameters
        best_result = results_sorted[0]
        best_params = best_result['params']
        
        logger.info("\n" + "="*60)
        logger.info("GRID SEARCH COMPLETE")
        logger.info("="*60)
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {metric}: {best_result[metric]:.1%}")
        logger.info("="*60)
        
        # Save results
        self._save_results(results_sorted[:top_n], best_params)
        
        return best_params, results_sorted[:top_n]
    
    def _save_results(self, top_results: List[Dict], best_params: Dict):
        """Save optimization results to JSON."""
        output = {
            'strategy': self.strategy_class.__name__,
            'best_params': best_params,
            'top_results': top_results
        }
        
        output_path = self.output_dir / f'optimization_{self.strategy_class.__name__}.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved results to: {output_path}")
    
    def seasonal_optimization(self, param_grid: Dict[str, List]) -> Dict[str, Dict]:
        """
        Find optimal parameters for each season.
        
        Seasons:
        - Spring: Mar-May
        - Summer: Jun-Aug
        - Fall: Sep-Nov
        - Winter: Dec-Feb
        
        Args:
            param_grid: Parameter grid to search
            
        Returns:
            Dictionary of {season: best_params}
        """
        logger.info("="*60)
        logger.info("SEASONAL OPTIMIZATION")
        logger.info("="*60)
        
        # Define seasons
        seasons = {
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11],
            'Winter': [12, 1, 2]
        }
        
        seasonal_params = {}
        
        for season_name, months in seasons.items():
            logger.info(f"\nOptimizing for {season_name} (months: {months})")
            
            # Filter data for this season
            season_mask = self.rtm_df['timestamp'].dt.month.isin(months)
            season_rtm = self.rtm_df[season_mask].copy()
            
            season_peaks = self.peaks_df[
                pd.to_datetime(self.peaks_df['date']).dt.month.isin(months)
            ].copy()
            
            if len(season_peaks) < 10:
                logger.warning(f"Insufficient data for {season_name}, skipping")
                continue
            
            # Run grid search for this season
            temp_optimizer = ParameterOptimizer(season_rtm, season_peaks, self.strategy_class)
            best_params, _ = temp_optimizer.grid_search(param_grid, top_n=1)
            
            seasonal_params[season_name] = best_params
            logger.info(f"{season_name} best params: {best_params}")
        
        # Save seasonal results
        output_path = self.output_dir / f'seasonal_optimization_{self.strategy_class.__name__}.json'
        with open(output_path, 'w') as f:
            json.dump(seasonal_params, f, indent=2)
        
        logger.info(f"\nSaved seasonal results to: {output_path}")
        
        return seasonal_params


def optimize_all_strategies():
    """Run optimization for all strategies."""
    from src.data_loader import ERCOTDataLoader
    from src.strategies import PriceDropStrategy, VelocityReversalStrategy
    
    logger.info("="*60)
    logger.info("OPTIMIZING ALL STRATEGIES")
    logger.info("="*60)
    
    # Load data
    loader = ERCOTDataLoader()
    rtm_df = loader.load_rtm_data('data/rtm_prices.csv')
    peaks_df = loader.get_daily_peaks(rtm_df)
    
    # Optimize PriceDropStrategy
    logger.info("\n1. Optimizing PriceDropStrategy...")
    price_drop_grid = {
        'lookback_minutes': [10, 15, 20],
        'drop_threshold': [0.015, 0.02, 0.025],
        'min_price_multiplier': [1.1, 1.15, 1.2],
        'longterm_minutes': [90, 120, 150]
    }
    
    optimizer1 = ParameterOptimizer(rtm_df, peaks_df, PriceDropStrategy)
    best_params1, top_results1 = optimizer1.grid_search(price_drop_grid)
    
    logger.info(f"\n✅ PriceDrop Best: {best_params1}")
    logger.info(f"   Success: {top_results1[0]['success_rate']:.1%}, Precision: {top_results1[0]['precision']:.1%}")
    
    # Optimize VelocityReversalStrategy
    logger.info("\n2. Optimizing VelocityReversalStrategy...")
    velocity_grid = {
        'velocity_window_minutes': [5, 10, 15],
        'acceleration_threshold': [-1.0, -0.5, -0.25],
        'price_percentile': [70, 75, 80],
        'lookback_minutes': [45, 60, 90]
    }
    
    optimizer2 = ParameterOptimizer(rtm_df, peaks_df, VelocityReversalStrategy)
    best_params2, top_results2 = optimizer2.grid_search(velocity_grid)
    
    logger.info(f"\n✅ VelocityReversal Best: {best_params2}")
    logger.info(f"   Success: {top_results2[0]['success_rate']:.1%}, Precision: {top_results2[0]['precision']:.1%}")
    
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info("="*60)
    
    return {
        'PriceDrop': (best_params1, top_results1),
        'VelocityReversal': (best_params2, top_results2)
    }

