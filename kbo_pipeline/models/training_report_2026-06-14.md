# KBO What-if 모델 학습 리포트

**실행 일시**: 2026-06-14 08:36:09  
**피처 모드**: mvp  
**타깃 라벨**: batting_team_win_label  
**검증 시즌**: 마지막 2시즌 자동 선택  
**사용 피처 수**: 31개  

---

## 모델 성능

| 모델 | AUC | Brier | LogLoss |
|------|-----|-------|---------|
| LightGBM | 0.8431 | 0.1651 | 0.5016 |

---

## LightGBM 피처 중요도 (gain 기준 Top 20)

| 순위 | 피처 | Gain | Split |
|------|------|------|-------|
| 1 | `batting_score_diff_before` | 127513 | 185 |
| 2 | `pitcher_pre_k9_before` | 12379 | 321 |
| 3 | `pitcher_pre_era_before` | 9050 | 249 |
| 4 | `pitcher_pre_ip_before` | 8507 | 240 |
| 5 | `pitcher_pre_whip_before` | 8140 | 231 |
| 6 | `pitcher_pre_bb9_before` | 7740 | 208 |
| 7 | `pitcher_pre_games_before` | 6085 | 174 |
| 8 | `inning` | 5252 | 89 |
| 9 | `pitcher_pre_cum_hr` | 3589 | 104 |
| 10 | `is_top_bool` | 1872 | 59 |
| 11 | `batter_pre_games_before` | 1252 | 45 |
| 12 | `state_re` | 736 | 27 |
| 13 | `is_home_batting` | 541 | 14 |
| 14 | `batter_pre_cum_ab` | 315 | 13 |
| 15 | `late_clutch` | 245 | 6 |
| 16 | `batter_pre_slg_before` | 122 | 6 |
| 17 | `batter_pre_cum_kk` | 94 | 3 |
| 18 | `batter_pre_avg_before` | 88 | 4 |
| 19 | `batter_pre_cum_bb` | 74 | 3 |
| 20 | `base3_before` | 39 | 2 |

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