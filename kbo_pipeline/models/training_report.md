# KBO What-if 모델 학습 리포트

**실행 일시**: 2026-06-14 17:50:16  
**피처 모드**: mvp  
**타깃 라벨**: batting_team_win_label  
**검증 시즌**: 마지막 2시즌 자동 선택  
**사용 피처 수**: 31개  

---

## 모델 성능

| 모델 | AUC | Brier | LogLoss |
|------|-----|-------|---------|
| Logistic | 0.8501 | 0.1583 | 0.4719 |
| LightGBM | 0.8615 | 0.1732 | 0.5293 |

---

## LightGBM 피처 중요도 (gain 기준 Top 20)

| 순위 | 피처 | Gain | Split |
|------|------|------|-------|
| 1 | `state_we` | 1285609 | 223 |
| 2 | `batting_score_diff_before` | 55244 | 33 |
| 3 | `pitcher_pre_whip_before` | 9150 | 119 |
| 4 | `pitcher_pre_ip_before` | 5381 | 94 |
| 5 | `pitcher_pre_k9_before` | 4869 | 98 |
| 6 | `pitcher_pre_era_before` | 4760 | 115 |
| 7 | `pitcher_pre_bb9_before` | 4615 | 85 |
| 8 | `inning` | 4469 | 46 |
| 9 | `pitcher_pre_games_before` | 3093 | 64 |
| 10 | `is_top_bool` | 2689 | 17 |
| 11 | `pitcher_pre_cum_hr` | 2203 | 52 |
| 12 | `late_clutch` | 1703 | 9 |
| 13 | `scoring_position_before` | 1669 | 2 |
| 14 | `batter_pre_games_before` | 516 | 15 |
| 15 | `batter_pre_cum_ab` | 403 | 7 |
| 16 | `state_re` | 240 | 4 |
| 17 | `batter_pre_cum_bb` | 133 | 4 |
| 18 | `is_home_batting` | 99 | 3 |
| 19 | `batter_pre_slg_before` | 29 | 1 |
| 20 | `batter_pre_cum_hr` | 25 | 1 |

---

## 사용 피처 목록

- `inning`
- `is_top_bool`
- `outs_before`
- `batting_score_diff_before`
- `runners_on_before`
- `base1_before`
- `base2_before`
- `base3_before`
- `scoring_position_before`
- `late_clutch`
- `is_home_batting`
- `state_we`
- `state_re`
- `batter_pre_games_before`
- `batter_pre_cum_ab`
- `batter_pre_avg_before`
- `batter_pre_obp_approx_before`
- `batter_pre_slg_before`
- `batter_pre_ops_before`
- `batter_pre_cum_hr`
- `batter_pre_cum_bb`
- `batter_pre_cum_kk`
- `pitcher_pre_games_before`
- `pitcher_pre_ip_before`
- `pitcher_pre_era_before`
- `pitcher_pre_whip_before`
- `pitcher_pre_k9_before`
- `pitcher_pre_bb9_before`
- `pitcher_pre_cum_hr`
- `same_hand_matchup`
- `batter_platoon_advantage`