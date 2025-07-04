import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const { action_id } = await request.json()

    // Send action to Python backend
    const response = await fetch("http://localhost:8000/make_action", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ action_id }),
    })

    if (!response.ok) {
      throw new Error("Failed to make action")
    }

    const gameState = await response.json()
    return NextResponse.json(gameState)
  } catch (error) {
    console.error("Error making action:", error)
    return NextResponse.json({ error: "Failed to make action" }, { status: 500 })
  }
}
