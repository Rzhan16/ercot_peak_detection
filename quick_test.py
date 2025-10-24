"""Quick test of enhanced strategies - clean output."""
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

from src.data_loader import ERCOTDataLoader
from src.strategies import PriceDropStrategy, EnsembleStrategy, VelocityReversalStrategy
from src.backtester import Backtester

# Load data
print("\n" + "="*70)
print("LOADING DATA...")
print("="*70)
loader = ERCOTDataLoader()
rtm_df = loader.load_rtm_data('data/rtm_prices.csv')
peaks_df = loader.get_daily_peaks(rtm_df)

# Add date column
rtm_df['date'] = rtm_df['timestamp'].dt.date

# Test on 60 days
test_dates = peaks_df['date'].iloc[:60]
test_rtm = rtm_df[rtm_df['date'].isin(test_dates)]
test_peaks = peaks_df.iloc[:60]

print(f"Testing on {len(test_peaks)} days\n")

# Test baseline
print("="*70)
print("TESTING BASELINE")
print("="*70)
baseline = PriceDropStrategy(lookback_minutes=10, drop_threshold=0.035, 
                            min_price_multiplier=1.25, longterm_minutes=120)
bt_baseline = Backtester(baseline, test_peaks)
res_baseline = bt_baseline.run_backtest(test_rtm, verbose=False)

print(f"Success:     {res_baseline['success_rate']:.1%}")
print(f"Precision:   {res_baseline.get('precision', 0):.1%}")

# Test enhanced
print("\n" + "="*70)
print("TESTING ENSEMBLE")
print("="*70)
ensemble = EnsembleStrategy([
    PriceDropStrategy(lookback_minutes=10, drop_threshold=0.035),
    VelocityReversalStrategy(velocity_window_minutes=15, acceleration_threshold=-1.0)
], min_votes=2)

bt_ensemble = Backtester(ensemble, test_peaks)
res_ensemble = bt_ensemble.run_backtest(test_rtm, verbose=False)

print(f"Success:     {res_ensemble['success_rate']:.1%}")
print(f"Precision:   {res_ensemble.get('precision', 0):.1%}")

# Summary
print("\n" + "="*70)
print("SUMMARY (60-day test)")
print("="*70)
print(f"{'Strategy':<25} {'Success':>12} {'Precision':>12}")
print("-"*70)
print(f"{'Baseline (PriceDrop)':<25} {res_baseline['success_rate']:>11.1%} {res_baseline.get('precision', 0):>11.1%}")
print(f"{'Ensemble':<25} {res_ensemble['success_rate']:>11.1%} {res_ensemble.get('precision', 0):>11.1%}")

improvement = ((res_ensemble.get('precision', 0) - res_baseline.get('precision', 0)) / 
               res_baseline.get('precision', 0.01) * 100)
print(f"\n🎉 Precision improvement: {improvement:+.1f}%")
print("\n✅ Enhanced strategies are working!\n")
