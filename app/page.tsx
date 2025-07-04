"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Loader2, Coins, TrendingUp, Clock } from "lucide-react"

interface GameState {
  current_player: number
  stage: string
  pot: number
  community_cards: string[]
  players: {
    id: number
    name: string
    stack: number
    hand: string[]
    in_chips: number
    folded: boolean
    is_human: boolean
    last_action?: string
  }[]
  legal_actions: { [key: number]: string }
  q_values?: { [key: string]: number }
  recommended_action?: string
  game_over: boolean
  winner?: number
  hand_over?: boolean
}

const SUIT_SYMBOLS = { C: "♣", D: "♦", H: "♥", S: "♠" }

function formatCard(card: string): string {
  if (card.length !== 2) return card
  const rank = card[0]
  const suit = card[1] as keyof typeof SUIT_SYMBOLS
  return `${rank}${SUIT_SYMBOLS[suit] || suit}`
}

function CardDisplay({ cards, hidden = false }: { cards: string[]; hidden?: boolean }) {
  if (hidden) {
    return (
      <div className="flex gap-1">
        {cards.map((_, i) => (
          <div
            key={i}
            className="w-12 h-16 bg-gradient-to-br from-blue-900 to-blue-800 border-2 border-blue-600 rounded-lg flex items-center justify-center shadow-lg transform hover:scale-105 transition-transform"
          >
            <div className="w-8 h-10 bg-gradient-to-br from-blue-800 to-blue-700 rounded border border-blue-600 shadow-inner"></div>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="flex gap-1">
      {cards.map((card, i) => (
        <div
          key={i}
          className="w-12 h-16 bg-gradient-to-br from-white to-gray-100 border-2 border-gray-300 rounded-lg flex items-center justify-center text-sm font-bold shadow-lg transform hover:scale-105 transition-transform"
        >
          <span className={card.includes("♥") || card.includes("♦") ? "text-red-600" : "text-black"}>
            {formatCard(card)}
          </span>
        </div>
      ))}
    </div>
  )
}

function ChipStack({ amount, size = "md" }: { amount: number; size?: "sm" | "md" | "lg" }) {
  const chipCount = Math.min(Math.floor(amount / 10) + 1, 6)
  const sizeClasses = {
    sm: "w-5 h-5",
    md: "w-6 h-6",
    lg: "w-8 h-8",
  }

  return (
    <div className="relative flex flex-col items-center justify-center">
      <div className="relative flex items-end justify-center mb-1">
        {Array.from({ length: chipCount }).map((_, i) => (
          <div
            key={i}
            className={`${sizeClasses[size]} rounded-full border-2 border-yellow-600 bg-gradient-to-br from-yellow-400 to-yellow-600 shadow-md absolute`}
            style={{
              bottom: `${i * 1.5}px`,
              zIndex: chipCount - i,
              transform: `rotate(${(i * 10) % 360}deg)`,
            }}
          >
            <div className="w-full h-full rounded-full bg-gradient-to-br from-yellow-300 to-yellow-500 border border-yellow-500 flex items-center justify-center">
              <Coins className="w-2 h-2 text-yellow-800" />
            </div>
          </div>
        ))}
      </div>
      <div
        className="text-xs font-bold text-center min-w-12 bg-black bg-opacity-70 text-white px-2 py-1 rounded z-20 relative"
        style={{ marginTop: `${chipCount * 1.5 + 4}px` }}
      >
        ${amount}
      </div>
    </div>
  )
}

function QValueDisplay({
  qValues,
  recommendedAction,
}: { qValues?: { [key: string]: number }; recommendedAction?: string }) {
  if (!qValues) return null

  const total = Object.values(qValues).reduce((sum, val) => sum + Math.exp(val), 0)
  const normalizedValues = Object.entries(qValues)
    .map(([action, value]) => ({
      action,
      value,
      probability: Math.exp(value) / total,
    }))
    .sort((a, b) => b.probability - a.probability)

  return (
    <Card className="w-64 bg-gradient-to-br from-purple-50 to-blue-50 border-purple-200">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <TrendingUp className="w-4 h-4" />
          AI Recommendation
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {normalizedValues.map(({ action, probability }) => (
          <div key={action} className="flex justify-between items-center">
            <span
              className={`text-xs font-medium ${action === recommendedAction ? "text-green-700 font-bold" : "text-gray-700"}`}
            >
              {action}
            </span>
            <div className="flex items-center gap-2">
              <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${action === recommendedAction ? "bg-gradient-to-r from-green-400 to-green-600" : "bg-gradient-to-r from-blue-400 to-blue-600"}`}
                  style={{ width: `${probability * 100}%` }}
                />
              </div>
              <span className="text-xs w-10 text-right font-mono">{(probability * 100).toFixed(1)}%</span>
            </div>
          </div>
        ))}
        {recommendedAction && (
          <Badge variant="outline" className="text-green-700 border-green-600 bg-green-50 text-xs">
            Recommended: {recommendedAction}
          </Badge>
        )}
      </CardContent>
    </Card>
  )
}

function PlayerActionIndicator({
  action,
  isVisible,
  isAI = false,
}: { action?: string; isVisible: boolean; isAI?: boolean }) {
  if (!isVisible || !action) return null

  return (
    <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 z-20">
      <div
        className={`px-4 py-2 rounded-full text-sm font-medium shadow-lg transition-all duration-500 ${
          isAI ? "bg-blue-600 text-white animate-pulse" : "bg-green-600 text-white"
        }`}
        style={{
          animation: isVisible ? "fadeInOut 10s ease-in-out" : "none",
        }}
      >
        {action}
      </div>
    </div>
  )
}

export default function PokerGame() {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [loading, setLoading] = useState(false)
  const [gameStarted, setGameStarted] = useState(false)
  const [lastAction, setLastAction] = useState<{ playerId: number; action: string } | null>(null)

  const startNewGame = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/poker/start", { method: "POST" })
      const data = await response.json()
      setGameState(data)
      setGameStarted(true)
      setLastAction(null)
    } catch (error) {
      console.error("Failed to start game:", error)
    } finally {
      setLoading(false)
    }
  }

  const makeAction = async (actionId: number) => {
    if (!gameState || gameState.current_player !== 0) return

    setLoading(true)
    const actionName = gameState.legal_actions[actionId]
    setLastAction({ playerId: 0, action: actionName })

    try {
      const response = await fetch("/api/poker/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action_id: actionId }),
      })
      const data = await response.json()
      setGameState(data)
    } catch (error) {
      console.error("Failed to make action:", error)
    } finally {
      setLoading(false)
    }
  }

  const pollGameState = async () => {
    if (!gameStarted || !gameState || gameState.current_player === 0 || gameState.game_over) return

    try {
      const response = await fetch("/api/poker/state")
      const data = await response.json()

      // Show AI action if current player changed
      if (data.current_player !== gameState.current_player && gameState.players) {
        const aiPlayer = gameState.players[gameState.current_player]
        if (aiPlayer && !aiPlayer.is_human) {
          // Determine the action taken by the AI
          const actionTaken = determineAIAction(gameState, data)
          setLastAction({ playerId: gameState.current_player, action: actionTaken })
        }
      }

      setGameState(data)
    } catch (error) {
      console.error("Failed to poll game state:", error)
    }
  }

  const determineAIAction = (prevState: GameState, newState: GameState): string => {
    if (!prevState.players || !newState.players) return "Action"

    const prevPlayer = prevState.players[prevState.current_player]
    const newPlayer = newState.players[prevState.current_player]

    if (!prevPlayer || !newPlayer) return "Action"

    if (newPlayer.folded && !prevPlayer.folded) return "Fold"
    if (newPlayer.in_chips > prevPlayer.in_chips) {
      const diff = newPlayer.in_chips - prevPlayer.in_chips
      return diff > 0 ? `Bet $${diff}` : "Call"
    }
    if (newPlayer.in_chips === prevPlayer.in_chips) return "Check"
    return "Action"
  }

  useEffect(() => {
    if (!gameStarted) return

    const interval = setInterval(pollGameState, 1000)
    return () => clearInterval(interval)
  }, [gameStarted, gameState])

  // Clear action indicator after 10 seconds with fade out
  useEffect(() => {
    if (lastAction) {
      const timer = setTimeout(() => setLastAction(null), 10000)
      return () => clearTimeout(timer)
    }
  }, [lastAction])

  if (!gameStarted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-800 via-green-700 to-green-900 flex items-center justify-center">
        <Card className="w-96 bg-gradient-to-br from-white to-gray-50 shadow-2xl">
          <CardHeader>
            <CardTitle className="text-center text-2xl font-bold text-gray-800">3-Player Poker</CardTitle>
          </CardHeader>
          <CardContent className="text-center space-y-4">
            <p className="text-gray-600">You vs 2 DQN AI Agents</p>
            <Button onClick={startNewGame} disabled={loading} className="w-full bg-green-600 hover:bg-green-700">
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Start Game
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!gameState) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-800 via-green-700 to-green-900 flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-white" />
      </div>
    )
  }

  // Add null checks to prevent runtime errors
  const humanPlayer = gameState.players?.find((p) => p.is_human)
  const aiPlayers = gameState.players?.filter((p) => !p.is_human) || []
  const isPlayerTurn = gameState.current_player === 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-800 via-green-700 to-green-900 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Game Info Header */}
        <div className="text-center mb-4">
          <h1 className="text-2xl font-bold text-white mb-2">Texas Hold'em Poker</h1>
          <div className="flex justify-center gap-6 text-white text-sm">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>Stage: {gameState.stage.charAt(0).toUpperCase() + gameState.stage.slice(1)}</span>
            </div>
            <div className="flex items-center gap-2">
              <Coins className="w-4 h-4" />
              <span>Pot: ${gameState.pot}</span>
            </div>
            <span>Current: {gameState.players?.[gameState.current_player]?.name}</span>
          </div>
        </div>

        {/* Main Game Area */}
        <div className="relative">
          {/* Community Cards */}
          <div className="text-center mb-6">
            <h3 className="text-white mb-3 text-lg font-semibold">Community Cards</h3>
            <div className="flex justify-center gap-2">
              {gameState.community_cards.length > 0 ? (
                <CardDisplay cards={gameState.community_cards} />
              ) : (
                <div className="text-white text-sm opacity-75">No community cards yet</div>
              )}
            </div>
            {/* Pot visualization */}
            <div className="mt-3 flex justify-center">
              <ChipStack amount={gameState.pot} size="lg" />
            </div>
          </div>

          {/* Players Layout */}
          <div className="relative w-full h-[350px]">
            {/* AI Player 1 (Top Left) */}
            <div className="absolute top-0 left-8">
              <div className="relative">
                <PlayerActionIndicator
                  action={lastAction?.playerId === 1 ? lastAction.action : undefined}
                  isVisible={lastAction?.playerId === 1}
                  isAI={true}
                />
                <Card className="w-48 bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200 shadow-xl">
                  <CardContent className="p-3">
                    <div className="text-center space-y-2">
                      <h4 className="font-bold text-blue-800 text-sm">{aiPlayers[0]?.name}</h4>
                      <CardDisplay cards={aiPlayers[0]?.hand || ["??", "??"]} hidden />
                      <div className="flex justify-between items-center text-xs">
                        <div className="text-center">
                          <div className="text-gray-600 mb-1">Stack</div>
                          <ChipStack amount={aiPlayers[0]?.stack || 0} size="sm" />
                        </div>
                        <div className="text-center">
                          <div className="text-gray-600 mb-1">Bet</div>
                          <ChipStack amount={aiPlayers[0]?.in_chips || 0} size="sm" />
                        </div>
                      </div>
                      {aiPlayers[0]?.folded && (
                        <Badge variant="destructive" className="text-xs">
                          Folded
                        </Badge>
                      )}
                      {gameState.current_player === 1 && <Badge className="bg-blue-600 text-xs">Current Player</Badge>}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* AI Player 2 (Top Right) */}
            <div className="absolute top-0 right-8">
              <div className="relative">
                <PlayerActionIndicator
                  action={lastAction?.playerId === 2 ? lastAction.action : undefined}
                  isVisible={lastAction?.playerId === 2}
                  isAI={true}
                />
                <Card className="w-48 bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 shadow-xl">
                  <CardContent className="p-3">
                    <div className="text-center space-y-2">
                      <h4 className="font-bold text-purple-800 text-sm">{aiPlayers[1]?.name}</h4>
                      <CardDisplay cards={aiPlayers[1]?.hand || ["??", "??"]} hidden />
                      <div className="flex justify-between items-center text-xs">
                        <div className="text-center">
                          <div className="text-gray-600 mb-1">Stack</div>
                          <ChipStack amount={aiPlayers[1]?.stack || 0} size="sm" />
                        </div>
                        <div className="text-center">
                          <div className="text-gray-600 mb-1">Bet</div>
                          <ChipStack amount={aiPlayers[1]?.in_chips || 0} size="sm" />
                        </div>
                      </div>
                      {aiPlayers[1]?.folded && (
                        <Badge variant="destructive" className="text-xs">
                          Folded
                        </Badge>
                      )}
                      {gameState.current_player === 2 && (
                        <Badge className="bg-purple-600 text-xs">Current Player</Badge>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Human Player (Bottom Center) */}
            <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2">
              <div className="relative">
                <PlayerActionIndicator
                  action={lastAction?.playerId === 0 ? lastAction.action : undefined}
                  isVisible={lastAction?.playerId === 0}
                  isAI={false}
                />
                <Card className="w-60 bg-gradient-to-br from-green-50 to-green-100 border-green-200 shadow-xl">
                  <CardContent className="p-3">
                    <div className="text-center space-y-2">
                      <h4 className="font-bold text-green-800 text-sm">{humanPlayer?.name} (You)</h4>
                      <CardDisplay cards={humanPlayer?.hand || []} />
                      <div className="flex justify-between items-center text-xs">
                        <div className="text-center">
                          <div className="text-gray-600 mb-1">Stack</div>
                          <ChipStack amount={humanPlayer?.stack || 0} size="sm" />
                        </div>
                        <div className="text-center">
                          <div className="text-gray-600 mb-1">Bet</div>
                          <ChipStack amount={humanPlayer?.in_chips || 0} size="sm" />
                        </div>
                      </div>
                      {humanPlayer?.folded && (
                        <Badge variant="destructive" className="text-xs">
                          Folded
                        </Badge>
                      )}
                      {isPlayerTurn && <Badge className="bg-green-600 text-xs">Your Turn</Badge>}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>

          {/* Action Buttons and Q-Values */}
          {isPlayerTurn && !gameState.game_over && (
            <div className="mt-6 text-center">
              <h3 className="text-white mb-3 text-lg font-semibold">Your Turn - Choose Action</h3>

              {/* Q-Values Display - Moved closer to action buttons */}
              <div className="flex justify-center mb-4">
                {gameState.q_values && (
                  <QValueDisplay qValues={gameState.q_values} recommendedAction={gameState.recommended_action} />
                )}
              </div>

              <div className="flex justify-center gap-3 flex-wrap">
                {Object.entries(gameState.legal_actions).map(([actionId, actionName]) => (
                  <Button
                    key={actionId}
                    onClick={() => makeAction(Number.parseInt(actionId))}
                    disabled={loading}
                    variant={actionName === gameState.recommended_action ? "default" : "outline"}
                    className={`px-6 py-3 text-lg font-semibold ${
                      actionName === gameState.recommended_action
                        ? "bg-green-600 hover:bg-green-700 border-green-500 shadow-lg"
                        : "bg-white hover:bg-gray-100 text-gray-800"
                    }`}
                  >
                    {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    {actionName}
                    {actionName === gameState.recommended_action && <TrendingUp className="ml-2 h-4 w-4" />}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {/* Hand Over / Game Over */}
          {(gameState.hand_over || gameState.game_over) && (
            <div className="mt-6 text-center">
              <Card className="w-96 mx-auto bg-gradient-to-br from-yellow-50 to-orange-50 border-yellow-200 shadow-xl">
                <CardContent className="p-6">
                  <h3 className="text-xl font-bold mb-4 text-gray-800">
                    {gameState.game_over ? "Game Over!" : "Hand Complete!"}
                  </h3>
                  {gameState.winner !== undefined && gameState.players && (
                    <p className="mb-4 text-lg font-semibold text-green-700">
                      Winner: {gameState.players[gameState.winner]?.name}
                    </p>
                  )}
                  {gameState.game_over ? (
                    <Button onClick={startNewGame} className="w-full bg-blue-600 hover:bg-blue-700">
                      Start New Game
                    </Button>
                  ) : (
                    <div className="text-gray-600">Next hand starting soon...</div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeInOut {
          0% { opacity: 0; transform: translateY(-10px) translateX(-50%); }
          10% { opacity: 1; transform: translateY(0) translateX(-50%); }
          90% { opacity: 1; transform: translateY(0) translateX(-50%); }
          100% { opacity: 0; transform: translateY(-10px) translateX(-50%); }
        }
      `}</style>
    </div>
  )
}
