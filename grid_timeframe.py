"""타임프레임별 백테스트 비교 (5m → 15m, 1h, 4h, 1d 리샘플링)."""
import os, warnings, logging
warnings.filterwarnings('ignore')
os.environ['TQDM_DISABLE'] = '1'
logging.disable(logging.CRITICAL)

from config.settings import SETTINGS
from scanner.config import SCANNER_SETTINGS
from scanner.backtest_engine import run_multi_backtest
import numpy as np, pandas as pd, glob

start, end = '2025-01-01', '2025-12-31'
cache_dir = 'data/cache'

# ── 데이터 로드 (5분봉 캐시) ──
print('Loading cached data...')
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

all_data_5m = {}
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
            all_data_5m[sym] = df

print(f'Loaded {len(all_data_5m)} symbols\n')


def resample_ohlcv(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """5분봉을 더 큰 타임프레임으로 리샘플링한다."""
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    if 'quote_volume' in df_5m.columns:
        agg['quote_volume'] = 'sum'

    resampled = df_5m.resample(rule).agg(agg).dropna(subset=['open'])
    return resampled


# ── 타임프레임별 데이터 준비 ──
timeframes = [
    ('5m', None),       # 원본
    ('15m', '15min'),
    ('1h', '1h'),
    ('4h', '4h'),
    ('1d', '1D'),
]

# 현재 설정 저장 (MP1, R5%)
SCANNER_SETTINGS['max_concurrent_positions'] = 1
SCANNER_SETTINGS['max_capital_per_position_pct'] = 1.00

results = []

for tf_label, resample_rule in timeframes:
    print(f'[{tf_label}] Preparing data...', end=' ', flush=True)

    if resample_rule is None:
        all_data = all_data_5m
    else:
        all_data = {}
        for sym, df in all_data_5m.items():
            resampled = resample_ohlcv(df, resample_rule)
            if len(resampled) > 50:
                all_data[sym] = resampled

    print(f'{len(all_data)} symbols.', end=' ', flush=True)

    # 백테스트 실행
    p = run_multi_backtest(
        all_data=all_data,
        daily_active=daily_active,
        capital=10000,
        max_positions=1,
    )

    t = p.trades
    total = len(t)
    wins = sum(1 for x in t if x.pnl > 0)
    wr = wins / total * 100 if total else 0
    gp = sum(x.pnl for x in t if x.pnl > 0)
    gl = abs(sum(x.pnl for x in t if x.pnl <= 0)) or 1
    pf = gp / gl
    avg_r = np.mean([x.r_multiple for x in t]) if t else 0
    ret = (p.capital / 10000 - 1) * 100

    eq = p.equity_curve
    pk = eq[0] if eq else 0
    mdd = 0
    for e in eq:
        if e > pk: pk = e
        dd = (e - pk) / pk * 100
        if dd < mdd: mdd = dd

    # 평균 보유 시간
    avg_hold = np.mean([x.hold_candles for x in t]) if t else 0

    results.append((tf_label, total, wr, pf, avg_r, ret, p.capital, mdd, avg_hold))
    print(f'{total} trades, PF={pf:.2f}, Ret={ret:+.0f}%, MDD={mdd:.1f}%', flush=True)

# ── 결과 출력 ──
print()
print('=' * 95)
print(f'{"TF":>5} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgR":>7} {"Return":>11} {"Final":>13} {"MDD":>7} {"AvgHold":>10}')
print('-' * 95)
for r in results:
    ar = f'{r[4]:.2f}R'
    print(f'{r[0]:>5} {r[1]:>7} {r[2]:>5.1f}% {r[3]:>6.2f} {ar:>7} {r[5]:>+10.1f}% ${r[6]:>11,.0f} {r[7]:>6.1f}% {r[8]:>7.0f} bars')
print('=' * 95)

# 설정 복원
SCANNER_SETTINGS['max_concurrent_positions'] = 1
SCANNER_SETTINGS['max_capital_per_position_pct'] = 1.00
