# 전지적 꼴데시점 ⚾
> *What if 예측 모델과 롯데 팬의 심리학*

**"그 상황에서 대타를 냈더라면 결과는 달랐을까?"**

매일 밤 반복되는 팬들의 '만약에' 시나리오. 단순한 미련이 아니라, 데이터로 검증할 수 있는 질문입니다.  
롯데 자이언츠 경기 데이터와 LightGBM 기반 승리확률 모델로 팬의 아쉬움을 분석으로 바꿉니다.

---

## 데모

| 메인 대시보드 | 경기 IF 분석 |
|:-:|:-:|
| ![메인](https://github.com/user-attachments/assets/0fa6b712-3106-4633-a1b0-a710dacb1733) | ![whatif](https://github.com/user-attachments/assets/b92288df-6f0e-4d30-851c-a69dbaa5567c) |

---

## 핵심 기능

**경기 IF 분석 (What-If 시뮬레이션)**  
타석 단위 승리확률(WP) 흐름을 시각화하고, 분기점마다 "이 결정이 달랐다면?" 시나리오를 AI로 시뮬레이션합니다.

**선수 개인 분석**  
롯데 타자·투수의 시즌 누적 스탯, OPS/ERA 트렌드, 상대팀별 성적을 조회합니다.

**시즌 대시보드**  
월별 승패, 누적 승률 추이, 상대팀 매트릭스 히트맵, 홈/원정 비교를 한 화면에서 확인합니다.

**경기 전 승리확률 예측 (예정)**  
선발 투수와 상대팀을 선택하면 경기 시작 전 예상 WP와 대체 선발 시나리오를 제공합니다.

**투수 교체 타이밍 분석**  
실제 교체 시점의 이닝·점수·WP 변화 분포로 최적 타이밍 인사이트를 도출합니다.

---

## 모델

승리확률(WP)은 **타석 단위**로 예측합니다.

```
WP = P(공격팀 승리 | 이닝, 초/말, 점수차, 아웃카운트, 주자상태, 타자 스탯, 투수 스탯, 매치업)
```

| 구성 요소 | 설명 |
|---|---|
| **WE (Win Expectancy)** | 같은 게임 상황에서 역사적으로 공격팀이 이긴 비율 |
| **RE (Run Expectancy)** | 아웃카운트·주자 상태 기반 기대 득점 |
| **Player Features** | 타자 OPS·타율, 투수 ERA·WHIP·K/9, 플래툰 매치업 |

**모델 선택: LightGBM**

| 모델 | AUC |
|---|---|
| LightGBM ✅ | ★★★★★ |
| XGBoost | ★★★★★ |
| Random Forest | ★★★★☆ |
| 로지스틱 회귀 | ★★★★☆ |
| Neural Net | ★★★★★ |

> AUC 86.2% — 학습 529,945 타석 / 검증 158,009 타석 / 피처 31개

---

## 데이터

데이터는 용량 문제로 Git에 포함되지 않습니다. 아래 링크에서 다운로드 후 `kbo_pipeline/data/processed/` 에 넣어주세요.

[📂 Google Drive 다운로드](https://drive.google.com/drive/folders/1LiBfLb6I6EWW7bvog-E2GwJ17MaQNAsp)

- **기간**: 2008~2025 (18시즌)
- **타석**: 649,419개 (`model_master_pa_eligible.csv`)
- **소스 테이블**: 19개 (games, batter/pitcher stats, WE table, RE table 등)

---

## 시작하기

```bash
git clone https://github.com/your-org/Donghaeng-LOTTO.git
cd Donghaeng-LOTTO
uv sync
uv run streamlit run app.py
```

---

## 스택

`Python` `LightGBM` `Streamlit` `Plotly` `Pandas` `Three.js`

---

## 팀

Base1 전해성(FlatBass FlatBass) : (데이터수집) · Base2 백락원 (모델학습) · Base3 정혁 (streamlit 구현, github 관리)
