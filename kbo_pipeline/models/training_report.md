# KBO What-if 모델 학습 리포트

**실행 일시**: 2026-06-12 15:51:17  
**피처 모드**: mvp  
**타깃 라벨**: batting_team_win_label  
**검증 시즌**: 마지막 2시즌 자동 선택  
**사용 피처 수**: 31개  

---

## 모델 성능

| 모델 | AUC | Brier | LogLoss |
|------|-----|-------|---------|
| Logistic | 0.8703 | 0.1461 | 0.4350 |
| LightGBM | 1.0000 | 0.0002 | 0.0012 |

---

## LightGBM 피처 중요도 (gain 기준 Top 20)

| 순위 | 피처 | Gain | Split |
|------|------|------|-------|
| 1 | `batting_score_diff_before` | 4582131 | 1437 |
| 2 | `pitcher_pre_k9_before` | 1997156 | 1700 |
| 3 | `pitcher_pre_bb9_before` | 1541288 | 1358 |
| 4 | `pitcher_pre_whip_before` | 1373061 | 1577 |
| 5 | `pitcher_pre_era_before` | 1295703 | 1454 |
| 6 | `pitcher_pre_ip_before` | 1045033 | 1258 |
| 7 | `batter_pre_games_before` | 699309 | 1264 |
| 8 | `is_top_bool` | 622626 | 704 |
| 9 | `batter_pre_obp_approx_before` | 502395 | 1333 |
| 10 | `pitcher_pre_cum_hr` | 467290 | 346 |
| 11 | `pitcher_pre_games_before` | 331755 | 639 |
| 12 | `batter_pre_avg_before` | 328428 | 978 |
| 13 | `batter_pre_slg_before` | 257964 | 902 |
| 14 | `batter_pre_ops_before` | 251979 | 1043 |
| 15 | `batter_pre_cum_ab` | 236142 | 1233 |
| 16 | `inning` | 199991 | 1241 |
| 17 | `batter_pre_cum_bb` | 181211 | 529 |
| 18 | `is_home_batting` | 170899 | 196 |
| 19 | `batter_pre_cum_kk` | 120834 | 528 |
| 20 | `state_re` | 71187 | 664 |

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