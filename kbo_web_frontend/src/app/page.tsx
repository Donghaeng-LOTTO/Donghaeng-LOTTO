"use client"

import React, { useState } from "react"

const KBO_GAMES = [
  { 
    id: "game_01", 
    date: "2026-06-11", 
    title: "KIA 타이거즈 vs 삼성 라이온즈", 
    stadium: "광주 챔피언스필드",
    finalScore: "KIA 5 : 6 삼성", 
    actualResult: "삼성 라이온즈 1점차 승리",
    turningPoints: [
      {
        id: "tp_01_1",
        inning: "8회말",
        situation: "1사 만루 (점수 4:5)",
        description: "KIA의 결정적인 역전 찬스. 당시 타석에 선 김도영 선수가 삼진으로 물러나며 기회가 무산된 승부처",
        defaultBatter: "김도영",
        defaultOuts: "1",
        defaultRunners: "full"
      },
      {
        id: "tp_01_2",
        inning: "6회초",
        situation: "2사 1,2루 (점수 3:2)",
        description: "삼성이 역전 투런 홈런을 치기 직전 상황. 투수 교체 타이밍이 아쉬웠던 분기점",
        defaultBatter: "구자욱",
        defaultOuts: "2",
        defaultRunners: "1_2nd"
      }
    ]
  },
  { 
    id: "game_02", 
    date: "2026-06-10", 
    title: "LG 트윈스 vs 두산 베어스", 
    stadium: "잠실 야구장",
    finalScore: "LG 4 : 3 두산", 
    actualResult: "LG 트윈스 승리",
    turningPoints: [
      {
        id: "tp_02_1",
        inning: "9회말",
        situation: "2사 2루 (점수 4:3)",
        description: "두산의 마지막 동점 주자가 득점권에 나갔으나 후속 타자 불발로 끝난 장면",
        defaultBatter: "양의지",
        defaultOuts: "2",
        defaultRunners: "2nd"
      }
    ]
  }
]

export default function KboWhatIfSimulator() {
  const [activeGame, setActiveGame] = useState<typeof KBO_GAMES[0] | null>(null)
  const [activeTp, setActiveTp] = useState<typeof KBO_GAMES[0]["turningPoints"][0] | null>(null)
  
  const [whatIfBatter, setWhatIfBatter] = useState("")
  const [whatIfOuts, setWhatIfOuts] = useState("1")
  const [whatIfRunners, setWhatIfRunners] = useState("full")
  
  const [isSimulating, setIsSimulating] = useState(false)
  const [report, setReport] = useState<any>(null)

  const handleSelectGame = (game: typeof KBO_GAMES[0]) => {
    setActiveGame(game)
    setActiveTp(null)
    setReport(null)
  }

  const handleSelectTp = (tp: typeof KBO_GAMES[0]["turningPoints"][0]) => {
    setActiveTp(tp)
    setWhatIfBatter(tp.defaultBatter)
    setWhatIfOuts(tp.defaultOuts)
    setWhatIfRunners(tp.defaultRunners)
    setReport(null)
  }

  const handleRunWhatIf = (e: React.FormEvent) => {
    e.preventDefault()
    setIsSimulating(true)
    
    setTimeout(() => {
      setIsSimulating(false)
      const baseOriginal = 28.4
      let bonus = 0
      if (whatIfBatter.includes("최형우") || whatIfBatter.includes("홈런")) bonus += 35
      if (whatIfOuts === "0") bonus += 15
      if (whatIfRunners === "full") bonus += 10

      const calculatedWinRate = Math.min(92.4, baseOriginal + bonus + Math.floor(Math.random() * 8))

      setReport({
        originalRate: baseOriginal,
        simulatedRate: calculatedWinRate,
        outcomeText: `${whatIfBatter} 선수 적시 타점 주자 일소 장타 작렬`,
        analysis: `당시 실제 경기 승리 확률은 ${baseOriginal}%로 좌절성이 짙었습니다. 그러나 핵심 분기점 데이터를 기반으로 타석을 [${whatIfBatter}] 선수로 교체하고 상황을 가상 시뮬레이션한 결과, 예상 승리 확률은 ${calculatedWinRate}%까지 급상승합니다.`
      })
    }, 1200)
  }

  return (
    <div style={{ minHeight: "100vh", backgroundColor: "#0f172a", color: "#f8fafc", fontFamily: "sans-serif", padding: "0", margin: "0" }}>
      
      {/* 상단 헤더 */}
      <header style={{ borderBottom: "1px solid #334155", backgroundColor: "#1e293b", padding: "16px 24px" }}>
        <div>
          <h1 style={{ fontSize: "20px", fontWeight: "bold", margin: "0 0 4px 0", color: "#fff" }}>KBO What-If 승패 분기점 시뮬레이터</h1>
          <p style={{ fontSize: "12px", color: "#94a3b8", margin: "0" }}>"그때 이랬으면 이겼을까?" 아쉬웠던 실제 과거 순간을 재구성합니다.</p>
        </div>
      </header>

      {/* 메인 레이아웃 구역 */}
      <main style={{ display: "flex", flexWrap: "wrap", gap: "24px", padding: "24px", maxWidth: "1200px", margin: "0 auto" }}>
        
        {/* 왼쪽 섹션: 경기 목록 및 분기점 */}
        <div style={{ flex: "1 1 350px", display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* 경기 선택 리스트 */}
          <section style={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "12px", padding: "20px" }}>
            <h2 style={{ fontSize: "14px", fontWeight: "bold", color: "#34d399", textTransform: "uppercase", margin: "0 0 16px 0" }}>01 분석할 경기 선택</h2>
            <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
              {KBO_GAMES.map((game) => (
                <div
                  key={game.id}
                  onClick={() => handleSelectGame(game)}
                  style={{
                    padding: "16px", borderRadius: "8px", border: activeGame?.id === game.id ? "2px solid #10b981" : "1px solid #334155",
                    backgroundColor: activeGame?.id === game.id ? "#064e3b" : "#0f172a", cursor: "pointer"
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#64748b", marginBottom: "6px" }}>
                    <span>📅 {game.date}</span>
                    <span>{game.stadium}</span>
                  </div>
                  <div style={{ fontWeight: "bold", fontSize: "14px", color: "#fff" }}>{game.title}</div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "#94a3b8", marginTop: "12px", paddingTop: "8px", borderTop: "1px solid #1e293b" }}>
                    <span>최종 {game.finalScore}</span>
                    <span style={{ color: "#ef4444" }}>{game.actualResult}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* 분기점 리스트 */}
          {activeGame && (
            <section style={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "12px", padding: "20px" }}>
              <h2 style={{ fontSize: "14px", fontWeight: "bold", color: "#34d399", margin: "0 0 16px 0" }}>02 승패의 결정적 분기점 (TP)</h2>
              <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                {activeGame.turningPoints.map((tp) => (
                  <div
                    key={tp.id}
                    onClick={() => handleSelectTp(tp)}
                    style={{
                      padding: "16px", borderRadius: "8px", border: activeTp?.id === tp.id ? "2px solid #10b981" : "1px solid #334155",
                      backgroundColor: activeTp?.id === tp.id ? "#064e3b" : "#0f172a", cursor: "pointer"
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                      <span style={{ fontSize: "16px", fontWeight: "bold", color: "#34d399" }}>{tp.inning}</span>
                      <span style={{ fontSize: "11px", padding: "2px 6px", borderRadius: "4px", backgroundColor: "#0f172a", border: "1px solid #334155", color: "#94a3b8" }}>{tp.situation}</span>
                    </div>
                    <p style={{ fontSize: "12px", color: "#94a3b8", margin: "0", lineHeight: "1.5" }}>{tp.description}</p>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>

        {/* 오른쪽 섹션: 조건 제어판 및 결과 리포트 */}
        <div style={{ flex: "2 1 500px" }}>
          {!activeTp ? (
            <div style={{ border: "2px dashed #334155", borderRadius: "16px", padding: "64px", textAlign: "center", backgroundColor: "#1e293b" }}>
              <div style={{ fontSize: "24px", marginBottom: "16px" }}>🔮</div>
              <h3 style={{ fontSize: "16px", fontWeight: "bold", color: "#cbd5e1", margin: "0 0 8px 0" }}>시뮬레이터 활성화 대기 중</h3>
              <p style={{ fontSize: "12px", color: "#64748b", margin: "0", lineHeight: "1.6" }}>좌측에서 경기 목록을 누르고 아쉬웠던 분기점을 선택하시면 기동 시뮬레이션 제어창이 열립니다.</p>
            </div>
          ) : (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "24px" }}>
              
              {/* 시나리오 설정 폼 */}
              <section style={{ flex: "1 1 240px", backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "12px", padding: "20px" }}>
                <h2 style={{ fontSize: "14px", fontWeight: "bold", color: "#34d399", margin: "0 0 16px 0" }}>03 시나리오 조건 변형</h2>
                <form onSubmit={handleRunWhatIf} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                  
                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    <label style={{ fontSize: "12px", color: "#94a3b8", fontWeight: "medium" }}>👤 타석의 타자 대치</label>
                    <input 
                      type="text" value={whatIfBatter} onChange={(e) => setWhatIfBatter(e.target.value)}
                      style={{ width: "100%", backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "6px", padding: "8px 12px", color: "#fff", fontSize: "13px", boxSizing: "border-box" }}
                    />
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    <label style={{ fontSize: "12px", color: "#94a3b8" }}>아웃 카운트 조건</label>
                    <div style={{ display: "flex", gap: "8px" }}>
                      {["0", "1", "2"].map((out) => (
                        <button
                          key={out} type="button" onClick={() => setWhatIfOuts(out)}
                          style={{
                            flex: 1, padding: "8px 0", borderRadius: "6px", border: "1px solid #334155", fontWeight: "bold", fontSize: "12px", cursor: "pointer",
                            backgroundColor: whatIfOuts === out ? "#10b981" : "#0f172a", color: whatIfOuts === out ? "#fff" : "#94a3b8"
                          }}
                        >
                          {out} 아웃
                        </button>
                      ))}
                    </div>
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                    <label style={{ fontSize: "12px", color: "#94a3b8" }}>주자 베이스 상황 레이아웃</label>
                    <select
                      value={whatIfRunners} onChange={(e) => setWhatIfRunners(e.target.value)}
                      style={{ width: "100%", backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "6px", padding: "8px 12px", color: "#fff", fontSize: "13px", cursor: "pointer" }}
                    >
                      <option value="empty">주자 없음</option>
                      <option value="1st">1루 주자 배치</option>
                      <option value="2nd">2루 주자 배치</option>
                      <option value="1_2nd">1, 2루 주자 배치</option>
                      <option value="full">주자 만루 (최대 찬스)</option>
                    </select>
                  </div>

                  <button
                    type="submit" disabled={isSimulating}
                    style={{
                      width: "100%", backgroundColor: "#10b981", color: "#fff", border: "none", fontWeight: "bold", padding: "12px 0", borderRadius: "8px", fontSize: "13px", cursor: "pointer", marginTop: "8px"
                    }}
                  >
                    {isSimulating ? "시뮬레이션 분석 중..." : "가상 시뮬레이션 가동"}
                  </button>
                </form>
              </section>

              {/* 분석 리포트 결과 패널 */}
              <section style={{ flex: "1 1 240px", backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "12px", padding: "20px", display: "flex", flexDirection: "column", justifyContent: "space-between", minHeight: "320px" }}>
                <div>
                  <h2 style={{ fontSize: "14px", fontWeight: "bold", color: "#34d399", margin: "0 0 16px 0" }}>04 가상 타임라인 분석 리포트</h2>
                  {!report ? (
                    <div style={{ textAlign: "center", padding: "48px 0", color: "#64748b", fontSize: "12px" }}>
                      <div style={{ fontSize: "20px", marginBottom: "8px" }}>📊</div>
                      조건 세팅 후 가동 버튼을 누르세요.
                    </div>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                      <div style={{ display: "flex", gap: "12px" }}>
                        <div style={{ flex: 1, backgroundColor: "#0f172a", padding: "10px", borderRadius: "8px", border: "1px solid #334155", textAlign: "center" }}>
                          <span style={{ fontSize: "10px", color: "#64748b", display: "block", marginBottom: "2px" }}>당시 실제 확률</span>
                          <span style={{ fontSize: "14px", fontWeight: "bold", color: "#94a3b8", textDecoration: "line-through" }}>{report.originalRate}%</span>
                        </div>
                        <div style={{ flex: 1, backgroundColor: "#064e3b", padding: "10px", borderRadius: "8px", border: "1px solid #047857", textAlign: "center" }}>
                          <span style={{ fontSize: "10px", color: "#34d399", display: "block", marginBottom: "2px" }}>가상 승리 확률</span>
                          <span style={{ fontSize: "16px", fontWeight: "black", color: "#34d399" }}>{report.simulatedRate}%</span>
                        </div>
                      </div>
                      <div>
                        <span style={{ fontSize: "10px", color: "#64748b", fontWeight: "bold" }}>시뮬레이션 플레이 결과</span>
                        <div style={{ backgroundColor: "#0f172a", padding: "10px", borderRadius: "6px", border: "1px solid #334155", fontSize: "12px", fontWeight: "bold", color: "#e2e8f0", marginTop: "4px" }}>⚡ {report.outcomeText}</div>
                      </div>
                      <p style={{ backgroundColor: "#0f172a", padding: "12px", borderRadius: "8px", fontSize: "12px", color: "#94a3b8", margin: "0", lineHeight: "1.6" }}>{report.analysis}</p>
                    </div>
                  )}
                </div>
                {report && (
                  <button onClick={() => setReport(null)} style={{ width: "100%", background: "none", border: "none", borderTop: "1px solid #334155", color: "#64748b", fontSize: "12px", paddingTop: "12px", marginTop: "16px", cursor: "pointer" }}>
                    🔄 시나리오 초기화
                  </button>
                )}
              </section>

            </div>
          )}
        </div>
      </main>
    </div>
  )
}