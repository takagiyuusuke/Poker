import { type NextRequest, NextResponse } from "next/server"

let gameState: any = null

export async function POST(request: NextRequest) {
  try {
    const response = await fetch("http://localhost:8000/start_game", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`)
    }

    const data = await response.json()
    gameState = data

    return NextResponse.json(data)
  } catch (error) {
    console.error("Start game API error:", error)
    return NextResponse.json({ error: "Failed to start game" }, { status: 500 })
  }
}
