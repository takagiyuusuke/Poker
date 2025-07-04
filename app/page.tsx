"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Loader2 } from "lucide-react"

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
  }[]
  legal_actions: { [key: number]: string }
  q_values?: { [key: string]: number }
  recommended_action?: string
  game_over: boolean
  winner?: number
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
            className="w-12 h-16 bg-blue-900 border border-white rounded-md flex items-center justify-center"
          >
            <div className="w-8 h-10 bg-blue-800 rounded border border-blue-700"></div>
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
          className="w-12 h-16 bg-white border border-gray-300 rounded-md flex items-center justify-center text-sm font-bold"
        >
          <span className={card.includes("♥") || card.includes("♦") ? "text-red-600" : "text-black"}>
            {formatCard(card)}
          </span>
        </div>
      ))}
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
    <Card className="w-64">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">AI推論結果</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {normalizedValues.map(({ action, probability }) => (
          <div key={action} className="flex justify-between items-center">
            <span className={`text-xs ${action === recommendedAction ? "font-bold text-green-600" : ""}`}>
              {action}
            </span>
            <div className="flex items-center gap-2">
              <div className="w-16 h-2 bg-gray-200 rounded">
                <div
                  className={`h-full rounded ${action === recommendedAction ? "bg-green-500" : "bg-blue-500"}`}
                  style={{ width: `${probability * 100}%` }}
                />
              </div>
              <span className="text-xs w-12 text-right">{(probability * 100).toFixed(1)}%</span>
            </div>
          </div>
        ))}
        {recommendedAction && (
          <Badge variant="outline" className="text-green-600 border-green-600">
            推奨: {recommendedAction}
          </Badge>
        )}
      </CardContent>
    </Card>
  )
}

export default function PokerGame() {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [loading, setLoading] = useState(false)
  const [gameStarted, setGameStarted] = useState(false)

  const startNewGame = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/poker/start", { method: "POST" })
      const data = await response.json()
      setGameState(data)
      setGameStarted(true)
    } catch (error) {
      console.error("Failed to start game:", error)
    } finally {
      setLoading(false)
    }
  }

  const makeAction = async (actionId: number) => {
    if (!gameState || gameState.current_player !== 0) return

    setLoading(true)
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
      setGameState(data)
    } catch (error) {
      console.error("Failed to poll game state:", error)
    }
  }

  useEffect(() => {
    if (!gameStarted) return

    const interval = setInterval(pollGameState, 1000)
    return () => clearInterval(interval)
  }, [gameStarted, gameState])

  if (!gameStarted) {
    return (
      <div className="min-h-screen bg-green-800 flex items-center justify-center">
        <Card className="w-96">
          <CardHeader>
            <CardTitle className="text-center">3人対戦ポーカー</CardTitle>
          </CardHeader>
          <CardContent className="text-center space-y-4">
            <p className="text-sm text-gray-600">あなた vs DQNエージェント2人でのポーカー対戦</p>
            <Button onClick={startNewGame} disabled={loading} className="w-full">
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              ゲーム開始
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (!gameState) {
    return (
      <div className="min-h-screen bg-green-800 flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-white" />
      </div>
    )
  }

  const humanPlayer = gameState.players.find((p) => p.is_human)
  const aiPlayers = gameState.players.filter((p) => !p.is_human)
  const isPlayerTurn = gameState.current_player === 0

  return (
    <div className="min-h-screen bg-green-800 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Game Info Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-white mb-2">3人対戦ポーカー</h1>
          <div className="flex justify-center gap-4 text-white">
            <span>ステージ: {gameState.stage}</span>
            <span>ポット: ${gameState.pot}</span>
            <span>現在のプレイヤー: {gameState.players[gameState.current_player]?.name}</span>
          </div>
        </div>

        {/* Main Game Area */}
        <div className="relative">
          {/* Community Cards */}
          <div className="text-center mb-8">
            <h3 className="text-white mb-2">コミュニティカード</h3>
            <div className="flex justify-center">
              <CardDisplay cards={gameState.community_cards} />
            </div>
          </div>

          {/* Players Layout */}
          <div className="relative w-full h-96">
            {/* AI Player 1 (Top Left) */}
            <div className="absolute top-0 left-8">
              <Card className="w-48">
                <CardContent className="p-4">
                  <div className="text-center space-y-2">
                    <h4 className="font-bold">{aiPlayers[0]?.name}</h4>
                    <CardDisplay cards={aiPlayers[0]?.hand || []} hidden />
                    <div className="text-sm">
                      <div>スタック: ${aiPlayers[0]?.stack}</div>
                      <div>ベット: ${aiPlayers[0]?.in_chips}</div>
                    </div>
                    {aiPlayers[0]?.folded && <Badge variant="destructive">フォールド</Badge>}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* AI Player 2 (Top Right) */}
            <div className="absolute top-0 right-8">
              <Card className="w-48">
                <CardContent className="p-4">
                  <div className="text-center space-y-2">
                    <h4 className="font-bold">{aiPlayers[1]?.name}</h4>
                    <CardDisplay cards={aiPlayers[1]?.hand || []} hidden />
                    <div className="text-sm">
                      <div>スタック: ${aiPlayers[1]?.stack}</div>
                      <div>ベット: ${aiPlayers[1]?.in_chips}</div>
                    </div>
                    {aiPlayers[1]?.folded && <Badge variant="destructive">フォールド</Badge>}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Human Player (Bottom Center) */}
            <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2">
              <Card className="w-64">
                <CardContent className="p-4">
                  <div className="text-center space-y-2">
                    <h4 className="font-bold text-blue-600">{humanPlayer?.name} (あなた)</h4>
                    <CardDisplay cards={humanPlayer?.hand || []} />
                    <div className="text-sm">
                      <div>スタック: ${humanPlayer?.stack}</div>
                      <div>ベット: ${humanPlayer?.in_chips}</div>
                    </div>
                    {humanPlayer?.folded && <Badge variant="destructive">フォールド</Badge>}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Q-Values Display (Right Side) */}
            {gameState.q_values && (
              <div className="absolute right-0 top-1/2 transform -translate-y-1/2">
                <QValueDisplay qValues={gameState.q_values} recommendedAction={gameState.recommended_action} />
              </div>
            )}
          </div>

          {/* Action Buttons */}
          {isPlayerTurn && !gameState.game_over && (
            <div className="mt-8 text-center">
              <h3 className="text-white mb-4">あなたのターンです</h3>
              <div className="flex justify-center gap-2 flex-wrap">
                {Object.entries(gameState.legal_actions).map(([actionId, actionName]) => (
                  <Button
                    key={actionId}
                    onClick={() => makeAction(Number.parseInt(actionId))}
                    disabled={loading}
                    variant={actionName === gameState.recommended_action ? "default" : "outline"}
                    className={actionName === gameState.recommended_action ? "bg-green-600 hover:bg-green-700" : ""}
                  >
                    {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    {actionName}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {/* Game Over */}
          {gameState.game_over && (
            <div className="mt-8 text-center">
              <Card className="w-96 mx-auto">
                <CardContent className="p-6">
                  <h3 className="text-xl font-bold mb-4">ゲーム終了</h3>
                  {gameState.winner !== undefined && (
                    <p className="mb-4">勝者: {gameState.players[gameState.winner]?.name}</p>
                  )}
                  <Button onClick={startNewGame} className="w-full">
                    新しいゲームを開始
                  </Button>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
