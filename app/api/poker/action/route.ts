import { type NextRequest, NextResponse } from "next/server"

let gameState: any = null

export async function POST(request: NextRequest) {
  try {
    const { action_id } = await request.json()

    const response = await fetch("http://localhost:8000/make_action", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ action_id }),
    })

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status}`)
    }

    const data = await response.json()
    gameState = data

    return NextResponse.json(data)
  } catch (error) {
    console.error("Action API error:", error)
    return NextResponse.json({ error: "Failed to make action" }, { status: 500 })
  }
}
