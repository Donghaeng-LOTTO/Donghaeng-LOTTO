# KBO What-if 모델 학습 리포트

**실행 일시**: 2026-06-12 17:10:54  
**피처 모드**: mvp  
**타깃 라벨**: batting_team_win_label  
**검증 시즌**: 마지막 2시즌 자동 선택  
**사용 피처 수**: 31개  

---

## 모델 성능

| 모델 | AUC | Brier | LogLoss |
|------|-----|-------|---------|
| Logistic | 0.9129 | 0.1213 | 0.3670 |
| LightGBM | 0.8703 | 0.1503 | 0.4598 |

---

## LightGBM 피처 중요도 (gain 기준 Top 20)

| 순위 | 피처 | Gain | Split |
|------|------|------|-------|
| 1 | `batting_score_diff_before` | 17753 | 181 |
| 2 | `pitcher_pre_k9_before` | 2766 | 186 |
| 3 | `pitcher_pre_era_before` | 2651 | 140 |
| 4 | `pitcher_pre_bb9_before` | 2604 | 140 |
| 5 | `pitcher_pre_whip_before` | 2224 | 142 |
| 6 | `pitcher_pre_ip_before` | 1941 | 139 |
| 7 | `is_top_bool` | 1896 | 56 |
| 8 | `batter_pre_games_before` | 945 | 75 |
| 9 | `batter_pre_obp_approx_before` | 650 | 46 |
| 10 | `is_home_batting` | 620 | 17 |
| 11 | `batter_pre_ops_before` | 590 | 48 |
| 12 | `batter_pre_slg_before` | 571 | 45 |
| 13 | `batter_pre_avg_before` | 551 | 48 |
| 14 | `pitcher_pre_cum_hr` | 438 | 25 |
| 15 | `pitcher_pre_games_before` | 390 | 39 |
| 16 | `batter_pre_cum_ab` | 300 | 32 |
| 17 | `batter_pre_cum_kk` | 170 | 18 |
| 18 | `inning` | 164 | 39 |
| 19 | `batter_pre_cum_bb` | 158 | 29 |
| 20 | `state_re` | 22 | 9 |

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