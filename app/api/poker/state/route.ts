import { NextResponse } from "next/server"

export async function GET() {
  try {
    // Get current game state from Python backend
    const response = await fetch("http://localhost:8000/get_state", {
      method: "GET",
    })

    if (!response.ok) {
      throw new Error("Failed to get game state")
    }

    const gameState = await response.json()
    return NextResponse.json(gameState)
  } catch (error) {
    console.error("Error getting game state:", error)
    return NextResponse.json({ error: "Failed to get game state" }, { status: 500 })
  }
}
