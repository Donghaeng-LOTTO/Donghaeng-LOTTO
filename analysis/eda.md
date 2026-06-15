### 전지적 롯데시점
##### 2015–2025 KBO 흐름 속 희망고문의 데이터
---

![주요 수치 피처 분포](image-2.png)

![주요 피처 간 상관관계 히트맵](image-5.png)

![팀별 홈 승률](image.png)

![점수차별 공격팀 승리 확률](image-3.png)

![이닝별 공격팀 승리 확률](image-4.png)

![승리/패배 타석의 Win Expectancy 분포 비교](image-6.png)

![RunExpectancy](image-15.png)

![9회 뜯어보기](image-16.png)

- 9이닝 경기에서 초 / 말 비교
    1. (왼쪽) 9회 초 원정 공격(점수를 낸 후에도 공격을 막아야 하는 상황) , (오른쪽) 9회 말 홈 공격(점수만 내면 바로 끝낼 수 있는 상황)
    2. (왼쪽) 9회 초 원정팀이 공격 중일 때 점수차별 승리 확률(점수차, x = 공격팀(원정) 점수 - 수비팀(홈팀) 점수, 공격팀 기준)
        - 9회 초 이기 때문에 공격을 마친 후에도 홈팀 공격을 막아야 한다. 따라서 아웃카운트가 늘어날수록 승률이 떨어지는 것을 볼 수 있다.
        - 반면 원정팀이 9회초에 이미 1점 앞서고 있으면, 공격 상황의 아웃카운트보다 리드 자체가 훨씬 중요해지는 것을 볼 수 있다.
    3. (오른쪽) 9회 말 홈팀 공격 점수차별 승리 확률(점수차, x = 공격팀(홈팀) 점수 - 수비팀(원정) 점수, 공격팀 기준)
        - 9회 말 홈팀의 점수차가 + 상황이라면 경기 진행에 의미가 없기 때문에 그래프에 표현되지 않음.
        - 1점인 상황과 동점인 상황일 때 승률이 급격한 차이를 보이는 것은 홈팀이 한 점만 내도 이길 수 있는 상황을 보여주는 것으로 해석할 수 있다.
    4. 공통적으로 아웃카운트가 늘어나면 WE는 낮아진다.

---

- 요약하자면 9회 초는 “앞서야 산다”, 9회 말은 “동점만 돼도 홈팀에게 끝내기 프리미엄이 붙는다.”

![WE 히트맵](image-17.png)

- 이닝·점수차 별 공격팀이 최종적으로 이길 확률
    1. 점수차가 커질수록 승률이 커진다.
    2. 후반으로 갈수록 동일한 점수차의 의미가 커진다. (동점 상황에서는 거의 반반확률)
---
    - 히트맵은 이닝과 공격팀 기준 점수차에 따른 Win Expectancy(WE)를 표현했다. 점수차가 공격팀에게 유리할수록 승리확률은 높아지고, 불리할수록 낮아지는 뚜렷한 패턴이 나타난다. 또한 같은 점수차라도 경기 후반으로 갈수록 승리확률이 더 극단적으로 변한다. 예를 들어 +1점 리드는 2회에는 약 0.59 수준이지만 9회에는 약 0.87까지 상승한다. 반대로 -1점 열세는 2회에는 약 0.37이지만 9회에는 약 0.15까지 낮아진다. 이는 야구에서 남은 공격 기회가 줄어들수록 현재 점수차의 영향력이 커진다는 것을 보여준다.

![롯데 자이언츠 시즌별 승률 추이](image-1.png)

```table
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game_date</th>
      <th>season</th>
      <th>game_id</th>
      <th>away_team_code</th>
      <th>home_team_code</th>
      <th>score</th>
      <th>winner_team_code</th>
      <th>batting_team_code</th>
      <th>fielding_team_code</th>
      <th>is_actual_comeback_win</th>
      <th>inning</th>
      <th>outs_before</th>
      <th>state_we</th>
      <th>batting_score_diff_before</th>
      <th>low_we_win_pa_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-03-14</td>
      <td>2015</td>
      <td>20150314LTWO0</td>
      <td>LT</td>
      <td>WO</td>
      <td>LT 5 : 3 WO</td>
      <td>LT</td>
      <td>LT</td>
      <td>WO</td>
      <td>1</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>0.171946</td>
      <td>-2.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-03-17</td>
      <td>2015</td>
      <td>20150317NCOB0</td>
      <td>NC</td>
      <td>OB</td>
      <td>NC 5 : 4 OB</td>
      <td>NC</td>
      <td>NC</td>
      <td>OB</td>
      <td>1</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.192308</td>
      <td>-2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-03-19</td>
      <td>2015</td>
      <td>20150319SSNC0</td>
      <td>SS</td>
      <td>NC</td>
      <td>SS 6 : 5 NC</td>
      <td>SS</td>
      <td>SS</td>
      <td>NC</td>
      <td>1</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.191781</td>
      <td>-2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-03-21</td>
      <td>2015</td>
      <td>20150321HTKT0</td>
      <td>HT</td>
      <td>KT</td>
      <td>HT 4 : 3 KT</td>
      <td>HT</td>
      <td>HT</td>
      <td>KT</td>
      <td>1</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.010181</td>
      <td>-3.0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-03-28</td>
      <td>2015</td>
      <td>20150328HHWO0</td>
      <td>HH</td>
      <td>WO</td>
      <td>HH 4 : 5 WO</td>
      <td>WO</td>
      <td>WO</td>
      <td>HH</td>
      <td>1</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.137097</td>
      <td>-3.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1210</th>
      <td>2025-09-21</td>
      <td>2025</td>
      <td>20250921NCHT02025</td>
      <td>NC</td>
      <td>HT</td>
      <td>NC 7 : 6 HT</td>
      <td>NC</td>
      <td>NC</td>
      <td>HT</td>
      <td>1</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.134951</td>
      <td>-2.0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1211</th>
      <td>2025-09-26</td>
      <td>2025</td>
      <td>20250926SSLT02025</td>
      <td>SS</td>
      <td>LT</td>
      <td>SS 9 : 10 LT</td>
      <td>LT</td>
      <td>LT</td>
      <td>SS</td>
      <td>1</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.135697</td>
      <td>-2.0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1212</th>
      <td>2025-10-01</td>
      <td>2025</td>
      <td>20251001HHSK02025</td>
      <td>HH</td>
      <td>SK</td>
      <td>HH 5 : 6 SK</td>
      <td>SK</td>
      <td>SK</td>
      <td>HH</td>
      <td>1</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>0.005848</td>
      <td>-3.0</td>
      <td>96</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>2025-10-01</td>
      <td>2025</td>
      <td>20251001KTHT02025</td>
      <td>KT</td>
      <td>HT</td>
      <td>KT 9 : 3 HT</td>
      <td>KT</td>
      <td>KT</td>
      <td>HT</td>
      <td>1</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.177500</td>
      <td>-3.0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1214</th>
      <td>2025-10-04</td>
      <td>2025</td>
      <td>20251004SSHT02025</td>
      <td>SS</td>
      <td>HT</td>
      <td>SS 8 : 9 HT</td>
      <td>HT</td>
      <td>HT</td>
      <td>SS</td>
      <td>1</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.070312</td>
      <td>-5.0</td>
      <td>112</td>
    </tr>
  </tbody>
</table>
<p>1215 rows × 15 columns</p>
</div>
```

![롯데 7회 이후 역전승 후보 경기](image-7.png)

![롯데 시즌별 7회 이후 역전승/역전패 후보 경기 수](image-8.png)

- 2021~2023은 역전패 후보가 역전승 후보보다 거의 2배 수준
- 7회 이후 한때 롯데의 승리확률이 높았던 경기를 끝까지 닫지 못한 경우가 많았다.

![롯데 vs OB(최다승) 7회 이후 역전 성공률 / 역전패 허용률](image-9.png)

- 최다승 팀: OB / 897 승
- 중간 팀: WO / 817 승
- 최소승 팀: HH / 729 승
- 비교 팀: ['LT', 'OB', 'WO', 'HH']
- 최다승 - 최소승: 168
- 최다승 - 중간팀: 80
- 중간팀 - 최소승: 88

![팀별 7회 이후 역전 성공률 - 역전패 허용률 변화](image-11.png)

![KBO 전체 팀별 7회 이후 역전 성공률 - 역전패 허용률](image-12.png)

![KBO 전체 팀별 시즌 승률 히트맵](image-13.png)

- OB는 2015~2020 강팀이었다가 이후 하락이 보이고,
- LG는 2019 이후 확실히 강팀권으로 올라오고,
- KT는 초반 약팀에서 2020 이후 중상위권으로 체질 개선된 흐름이 보임.
- 그러나 롯데는 꾸줂 미지근함...희망고문...

- 롯데는 최근 구간에서 승률 변동성이 매우 낮은 팀으로 나타난다. 그러나 평균 승률 역시 45% 수준에 머무르기 때문에, 이는 안정적인 강팀이라기보다 중하위권에서 큰 반등 없이 유지되는 패턴에 가깝다. 즉 롯데는 급격히 무너지는 팀은 아니지만, 상위권으로 치고 올라가는 지속성도 부족한 ‘희망고문형 안정팀’으로 해석할 수 있다.

![주요 팀 7회 이후 후반 반전 지표](image-14.png)

승률 히트맵과 7회 이후 후반 반전 지표를 함께 보면, 팀별 성격 차이가 드러난다.

OB는 2015~2020년 높은 승률과 양호한 후반 반전 지표를 함께 보이며 과거 강팀의 특징을 보였다. 그러나 2021년 이후에는 승률과 후반 지표가 모두 약화되며 이전만큼의 안정성은 줄어든 모습이다.

LG는 2019년 이후 승률이 꾸준히 높아졌고, 일부 시즌에서는 후반 반전 지표도 크게 양수로 나타났다. 이는 최근 구간에서 LG가 단순히 승률만 높은 것이 아니라 후반 경기 운영에서도 강팀의 특징을 보이고 있음을 시사한다.

KT는 초기에는 낮은 승률을 보였지만 2020년 이후 승률이 중상위권으로 올라왔고, 후반 반전 지표도 비교적 안정적으로 양수 구간을 보인다. 따라서 KT는 시간이 지날수록 팀 안정성이 강화된 성장형 팀으로 해석할 수 있다.

롯데는 리그 최악 수준의 급락을 보인 팀은 아니지만, 2021년 이후 후반 반전 지표가 지속적으로 음수권에 머문다. 이는 롯데가 크게 무너지는 팀이라기보다, 후반에 승부를 뒤집는 힘보다 지키지 못하는 위험이 조금 더 큰 상태를 꾸준히 벗어나지 못한 팀으로 볼 수 있다.



