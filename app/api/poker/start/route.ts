import { NextResponse } from "next/server"

// Game state will be stored in memory for this demo
// In production, you'd want to use a database or session storage
let gameState: any = null

export async function POST() {
  try {
    // Initialize a new game by calling the Python backend
    const response = await fetch("http://localhost:8000/start_game", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      throw new Error("Failed to start game")
    }

    gameState = await response.json()
    return NextResponse.json(gameState)
  } catch (error) {
    console.error("Error starting game:", error)
    return NextResponse.json({ error: "Failed to start game" }, { status: 500 })
  }
}
