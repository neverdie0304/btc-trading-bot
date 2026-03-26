---
name: optimization_results_2025_2026q1
description: Grid search optimization results - best params by PF, Return, Sharpe, Fitness on 2025 train + 2026 Q1 validation
type: project
---

# Parameter Optimization Results (2026-03-26)

## Search Space
- SL: [0.8, 1.0, 1.2, 1.5, 2.0]
- TP: [2.0, 2.5, 3.0, 3.6, 4.0, 5.0]
- Short: [ON, OFF]
- Volume Threshold: [1.0, 1.2, 1.5, 2.0]
- Total: 240 combinations

## Train: 2025 full year (4 quarters) / Validation: 2026-01-01 ~ 2026-03-25

## Validation 기준별 최적 조합

### PF 최적 (보수적, 안정적)
- SL=2.0, TP=5.0, Short=OFF, Vol=2.0
- Val: PF 1.81, +94%, MDD -9.7%, Sharpe 4.71, Fitness 15.1
- 특성: 거래 85회 (적음), 매우 낮은 MDD, 높은 PF

### 수익률 최적 (공격적, 성장형)
- SL=1.2, TP=5.0, Short=OFF, Vol=1.0
- Val: PF 1.36, +1365%, MDD -40.0%, Sharpe 6.46, Fitness 19.6
- 특성: 거래 583회, 높은 MDD 감수하고 최대 수익

### Sharpe 최적 (리스크 대비 수익 효율)
- SL=1.5, TP=4.0, Short=ON, Vol=1.0
- Val: PF 1.36, +1006%, MDD -37.1%, Sharpe 7.82, Fitness 21.1
- 특성: 거래 604회, Sharpe와 Fitness 모두 상위권

### Fitness 최적 (종합 밸런스)
- SL=2.0, TP=2.5, Short=ON, Vol=1.0
- Val: PF 1.30, +318%, MDD -14.4%, Sharpe 6.20, Fitness 27.7
- 특성: 거래 619회, 낮은 MDD + 높은 거래수 균형

## 현재 설정 (SL=1.2, TP=3.6, Short=ON, Vol=1.0)
- Val: PF 1.31, +907%, MDD -42.1%, Sharpe 6.70, Fitness 19.1

## 시드 크기별 추천 전략
- 저시드 (공격적): SL=1.2, TP=5.0, Short=OFF, Vol=1.0 → 수익률 극대화
- 중시드 (균형): SL=1.5, TP=4.0, Short=ON, Vol=1.0 → Sharpe 최적
- 고시드 (방어적): SL=2.0, TP=3.0~5.0, Short=OFF, Vol=2.0 → PF/MDD 최적
