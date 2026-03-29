"""상위 4개 조합 2025년 월별 성과 분석."""
import os, warnings, logging
from collections import defaultdict

warnings.filterwarnings('ignore')
os.environ['TQDM_DISABLE'] = '1'
logging.disable(logging.CRITICAL)

from scanner.config import SCANNER_SETTINGS
from config.settings import SETTINGS
from scanner.backtest_engine import run_multi_backtest
import numpy as np, pandas as pd, glob

start, end = '2025-01-01', '2025-12-31'
cache_dir = 'data/cache'

# 데이터 로드
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

# 상위 4개 조합: (max_pos, risk, cap_pct, label)
configs = [
    (1, 0.05, 1.00, "MP1 R5.0%"),
    (2, 0.05, 0.50, "MP2 R5.0%"),
    (1, 0.035, 1.00, "MP1 R3.5%"),
    (3, 0.05, 0.33, "MP3 R5.0%"),
]

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

for idx, (mp, rr, cap_pct, label) in enumerate(configs):
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

    # 월별 집계
    monthly_pnl = defaultdict(float)
    monthly_trades = defaultdict(int)
    monthly_wins = defaultdict(int)
    for trade in t:
        m = str(trade.entry_time)[:7]  # "2025-01"
        monthly_pnl[m] += trade.pnl
        monthly_trades[m] += 1
        if trade.pnl > 0:
            monthly_wins[m] += 1

    # 월별 수익률 (복리 기반)
    monthly_returns = {}
    bal = 10000
    for m_idx in range(1, 13):
        m_key = f'2025-{m_idx:02d}'
        pnl = monthly_pnl.get(m_key, 0)
        m_ret = pnl / bal * 100 if bal > 0 else 0
        monthly_returns[m_key] = (m_ret, monthly_trades.get(m_key, 0), monthly_wins.get(m_key, 0), pnl)
        bal += pnl

    print(f'{"=" * 80}')
    print(f'  [{idx+1}/4] {label}  |  ActRisk: {rr/mp*100:.2f}%')
    print(f'  Trades: {total} | WR: {wr:.1f}% | PF: {pf:.2f} | Return: {ret:+.1f}% | MDD: {mdd:.1f}%')
    print(f'  $10,000 -> ${p.capital:,.0f}')
    print(f'{"-" * 80}')
    print(f'  {"Month":>7} {"Return%":>9} {"PnL$":>10} {"Trades":>7} {"Wins":>5} {"WR%":>6}')
    print(f'  {"-" * 50}')

    for m_idx in range(1, 13):
        m_key = f'2025-{m_idx:02d}'
        m_ret, m_cnt, m_win, m_pnl = monthly_returns[m_key]
        m_wr = m_win / m_cnt * 100 if m_cnt > 0 else 0
        print(f'  {months[m_idx-1]:>7} {m_ret:>+8.1f}% ${m_pnl:>9,.0f} {m_cnt:>7} {m_win:>5} {m_wr:>5.0f}%')

    print()

# 복원
SCANNER_SETTINGS['max_concurrent_positions'] = 3
SCANNER_SETTINGS['max_capital_per_position_pct'] = 0.33
SETTINGS['risk_per_trade'] = 0.05
print('Done.')
