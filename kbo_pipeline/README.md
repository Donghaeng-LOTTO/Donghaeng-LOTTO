GIANTS AI Analytics
딥러닝 기반 야구 의사결정 지원 시스템 (Decision Support System)

경기 데이터 연산 시뮬레이션을 통해 벤치의 선택과 AI의 최적 제안을 비교 분석하고, 팀 로스터 및 시즌 성적을 관리하는 데이터 분석 플랫폼입니다.

주요 기능
1. 경기 일정 분석 (Game Schedule Analysis)
경기 분기점(Turning Point) 진단: 승패를 가른 핵심 상황과 실제 벤치의 선택을 정밀 기록합니다.

기대 승률 변동 시뮬레이션: Plotly 시각화를 통해 실제 선택과 AI 최적 제안 간의 승률 기댓값 차이를 시각적으로 비교합니다.

단정적 AI 리포트: 딥러닝 모델 알고리즘에 기반하여 대타 기용, 구종 선택 등의 오판 가능성을 정량적으로 분석하고 진단 로그를 제공합니다.

2. 선수 데이터베이스 (Player Database)
핵심 로스터 데이터 분석: 주요 선수들의 시즌 타율, 출루율(OBP), 도루 성공률, 방어율(ERA), WHIP 등 핵심 세부 지표를 그리드 카드 레이아웃으로 직관적으로 노출합니다.

3. 시즌 리더보드 (Season Leaderboard)
고대비 데이터 테이블: 고해상도 다크 테마가 적용된 테이블을 통해 리그 내 팀별 승무패 및 승률 순위 현황을 명확하게 파악할 수 있습니다.


# KBO What-if 승리확률 데이터 파이프라인 (2008–2025)

네이버 스포츠 API(schedule / record / relay) 기반 KBO 전 경기 수집 → 정규화 →
선수 ID 매핑 → pre-game 누적 기록 생성까지의 ETL 초안.

## 디렉터리 구조

```
src/
  config.py                     # 경로/상수/팀코드/시즌범위
  clients/naver_api.py          # 세션, 재시도, raw JSON 캐시, 실패 로그
  collectors/collect_games.py   # schedule + record fallback -> games.csv
  collectors/collect_records.py # record -> games_detail, boxscores
  collectors/collect_relays.py  # relay raw + lineup 선수 마스터
  parsers/parse_records.py      # record JSON 파싱
  parsers/parse_relays.py       # relay JSON -> events, PA 그룹화
  parsers/state_reconstructor.py# 타석 전 상태(점수/아웃/주자) 재구성 + 검증
  features/player_id_map.py     # 네이버 pcode <-> Kaggle Id 매핑
  features/pre_game_stats.py    # shift(1) 기반 pre-game 누적 기록
  features/pa_features.py       # is_top, 롯데 플래그, late_clutch, 승패 라벨
  quality/quality_report.py     # dataset_quality_report.csv
scripts/
  run_collect_2008_2025.py      # 수집 (연/월 단위, 재시작 가능)
  run_build_features.py         # 파싱/피처/품질 리포트
```

## 사전 준비

Kaggle CSV 2개를 다음 위치에 둔다 (player_id_map용, 없어도 나머지는 동작):

```
data/external/kaggle/kbo_batting_stats_by_season_1982-2025.csv
data/external/kaggle/kbo_pitching_stats_by_season_1982-2025.csv
```

## 실행 (Windows + uv)

```powershell
# 의존성 설치
uv sync

# 1) 소규모 테스트 (2008년 6월만)
uv run python scripts/run_collect_2008_2025.py --start-year 2008 --end-year 2008 --months 6

# 2) 피처 빌드 테스트
uv run python scripts/run_build_features.py

# 3) 전체 수집 (2008~2025, 3~11월) — 며칠 걸릴 수 있음. 중단 후 재실행 가능.
uv run python scripts/run_collect_2008_2025.py

# 롯데 경기만 먼저 모으고 싶다면
uv run python scripts/run_collect_2008_2025.py --team-codes LT

# 단계별 스킵
uv run python scripts/run_collect_2008_2025.py --skip-games --skip-records   # relay만
uv run python scripts/run_build_features.py --skip-events                    # 피처만 재계산
```

## 산출물

| 파일 | 내용 |
|---|---|
| `data/processed/games.csv` | schedule 기반 경기 목록 |
| `data/processed/games_detail.csv` | record 기반 메타 + 최종 점수/승패 |
| `data/processed/naver_players_seen.csv` | relay lineup 기반 선수 마스터 |
| `data/processed/player_id_map.csv` | 네이버 pcode ↔ Kaggle Id |
| `data/processed/batter_game_boxscores.csv` | 경기별 타자 boxscore |
| `data/processed/pitcher_game_boxscores.csv` | 경기별 투수 boxscore |
| `data/processed/relay_events.csv` | 이벤트 단위 play-by-play |
| `data/processed/plate_appearances.csv` | 타석 단위 + 사전 상태 + 파생 컬럼 |
| `data/processed/batter_pre_game_stats.csv` | 경기 전 누적 (shift(1)) |
| `data/processed/pitcher_pre_game_stats.csv` | 경기 전 누적 (shift(1)) |
| `data/processed/score_validation.csv` | 상태 재구성 점수 검증 |
| `data/processed/dataset_quality_report.csv` | 경기별 품질 체크 |
| `data/logs/failed_requests.csv` | 실패한 요청 로그 |

## 설계 원칙

1. **raw 우선 저장**: 모든 API 응답은 파싱 전에 `data/raw/`에 JSON으로 저장.
   파싱 로직이 바뀌어도 재요청 없이 재처리 가능.
2. **캐시**: raw JSON이 이미 있으면 재요청하지 않음 → 중단/재시작 안전.
3. **rate limit**: 요청 간 `sleep_sec`(기본 0.4s) + 재시도 백오프.
4. **데이터 누수 금지**: pre-game 스탯은 날짜순 cumsum 후 `shift(1)`.
   같은 경기 boxscore가 같은 경기 피처로 절대 들어가지 않음.
5. **game_id 날짜 검증**: schedule 응답에 현재 날짜 경기가 섞이는 문제 →
   `game_id[:8] == 요청날짜` 강제 + 비정규 팀코드 제외.
6. **상태 재구성**: currentGameState가 전부 0인 과거 경기는 텍스트 기반 재구성.
   불확실 이벤트는 `state_parse_status`/`parse_warning`에 기록,
   `score_validation.csv`로 record 최종 점수와 대조.

## 알려진 한계 (MVP)

- **OBP**: boxscore에 HBP/SF가 없어 `(H+BB)/(AB+BB)` 근사 (`obp_approx_before`).
- **SLG**: 2/3루타는 plate_appearances에서 보강. PA가 없는 경기 구간은
  비HR 안타를 단타로 간주한 하한값 (`slg_is_lower_bound=True`).
- **텍스트 상태 재구성**: 대주자/견제사/특수 플레이 등은 불완전할 수 있음.
  `parse_warning`이 있는 타석은 모델 학습 시 필터링하거나 가중치를 낮출 것.
- **팀코드 변천**: SK→SSG(2021), 넥센/키움(WO) 등은 `config.team_codes_for_year`에서
  관리. 실제 game_id 표기와 다르면 여기만 고치면 됨.
- **더블헤더**: game_id 끝자리 0/1/2. fallback 후보는 0만 시도하지만
  record의 `games` 목록에 그날 전 경기가 포함되므로 더블헤더도 수집됨.

## 권장 검증 루틴

```powershell
# 수집 후 빠른 점검
uv run python -c "import pandas as pd; df=pd.read_csv('data/processed/dataset_quality_report.csv'); print(df[['has_record_raw','has_relay_raw','pa_count_plausible']].mean()); print(df[df['pa_count_plausible']==False].head(20))"
```