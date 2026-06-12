import React from "react"
import "./globals.css" // 전역 스타일 연동

export const metadata = {
  title: "KBO What-If Simulator",
  description: "KBO 경기 승패 분기점 시뮬레이터",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="ko">
      <body>
        {children}
      </body>
    </html>
  )
}