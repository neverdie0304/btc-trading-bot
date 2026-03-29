"""MaxPos × Risk 그리드 서치."""
import os, warnings, logging
warnings.filterwarnings('ignore')
os.environ['TQDM_DISABLE'] = '1'
logging.disable(logging.CRITICAL)

from scanner.config import SCANNER_SETTINGS
from config.settings import SETTINGS
from scanner.backtest_engine import run_multi_backtest
import numpy as np, pandas as pd, glob

start, end = '2026-01-01', '2026-03-29'
cache_dir = 'data/cache'

vol_files = sorted(glob.glob(f'{cache_dir}/daily_volumes_*.parquet'))
vol_dfs = [pd.read_parquet(f) for f in vol_files]
vol_df = pd.concat(vol_dfs)
vol_df = vol_df[~vol_df.index.duplicated(keep='last')]
vol_df.index = pd.to_datetime(vol_df.index)
vol_df.sort_index(inplace=True)
vol_df = vol_df.loc[start:end]

min_vol = 50_000_000
daily_active = {}
for date_idx, row in vol_df.iterrows():
    d = str(date_idx)[:10]
    vols = row.dropna().sort_values(ascending=False)
    top = vols[vols >= min_vol]
    daily_active[d] = top.index.tolist()[:50]

unique = sorted(set(s for syms in daily_active.values() for s in syms))
all_data = {}
for sym in unique:
    path = f'{cache_dir}/{sym}_5m.parquet'
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if 'open_time' in df.columns:
            df.set_index('open_time', inplace=True)
        df.sort_index(inplace=True)
        mask = (df.index >= start) & (df.index <= end)
        df = df[mask]
        if len(df) > 50:
            all_data[sym] = df

print(f'Loaded {len(all_data)} symbols, {len(daily_active)} days\n')

max_positions = [1, 2, 3]
risk_rates = [0.015, 0.025, 0.035, 0.05]
results = []
total_combos = len(max_positions) * len(risk_rates)
done = 0

for mp in max_positions:
    cap_pct = 1.0 / mp
    for rr in risk_rates:
        SCANNER_SETTINGS['max_concurrent_positions'] = mp
        SCANNER_SETTINGS['max_capital_per_position_pct'] = cap_pct
        SETTINGS['risk_per_trade'] = rr
        p = run_multi_backtest(all_data=all_data, daily_active=daily_active, capital=10000, max_positions=mp)
        t = p.trades
        total = len(t)
        wins = sum(1 for x in t if x.pnl > 0)
        wr = wins / total * 100 if total else 0
        gp = sum(x.pnl for x in t if x.pnl > 0)
        gl = abs(sum(x.pnl for x in t if x.pnl <= 0)) or 1
        pf = gp / gl
        ret = (p.capital / 10000 - 1) * 100
        eq = p.equity_curve
        pk = eq[0] if eq else 0
        mdd = 0
        for e in eq:
            if e > pk: pk = e
            dd = (e - pk) / pk * 100
            if dd < mdd: mdd = dd
        actual_risk = rr / mp * 100
        ratio = ret / abs(mdd) if mdd != 0 else 0
        results.append((mp, rr * 100, actual_risk, total, wr, pf, ret, p.capital, mdd, ratio))
        done += 1
        print(f'  [{done}/{total_combos}] MaxPos={mp} Risk={rr*100:.1f}% -> {total} trades, PF={pf:.2f}, Ret={ret:+.0f}%, MDD={mdd:.1f}%')

# 복원
SCANNER_SETTINGS['max_concurrent_positions'] = 3
SCANNER_SETTINGS['max_capital_per_position_pct'] = 0.33
SETTINGS['risk_per_trade'] = 0.05

results.sort(key=lambda x: x[9], reverse=True)

print()
print('=' * 100)
print(f'{"MaxPos":>6} {"Risk%":>6} {"ActRsk":>7} {"Trades":>7} {"WR%":>6} {"PF":>6} {"Return":>11} {"Final":>13} {"MDD":>7} {"Ret/MDD":>8}')
print('-' * 100)
for r in results:
    ratio = f'{r[9]:.1f}x'
    print(f'{r[0]:>6} {r[1]:>5.1f}% {r[2]:>6.2f}% {r[3]:>7} {r[4]:>5.1f}% {r[5]:>6.2f} {r[6]:>+10.1f}% ${r[7]:>11,.0f} {r[8]:>6.1f}% {ratio:>8}')
print('=' * 100)
