"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Loader2, Coins, TrendingUp, Clock, X } from "lucide-react"

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
  hand_results?: {
    winner: number
    payoffs: number[]
    final_stacks: number[]
    hand_ranks?: string[]
    community_cards?: string[]
    player_hands?: string[][]
    folded?: boolean[]
  }
}

interface HandResult {
  winner: number
  payoffs: number[]
  final_stacks: number[]
  player_names: string[]
  hand_ranks?: string[]
  community_cards?: string[]
  player_hands?: string[][]
  folded?: boolean[]
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
        {"$" + amount}
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
}: {
  action?: string
  isVisible: boolean
  isAI?: boolean
}) {
  if (!isVisible || !action) return null

  return (
    <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 z-20">
      <div
        className={`px-4 py-2 rounded-full text-sm font-medium shadow-lg transition-all duration-500 ${
          isAI
            ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white"
            : "bg-gradient-to-r from-green-600 to-green-700 text-white"
        }`}
        style={{
          animation: "fadeInOut 3s ease-in-out",
          opacity: isVisible ? 1 : 0,
        }}
      >
        {action}
      </div>
    </div>
  )
}

function HandResultModal({
  handResult,
  onClose,
}: {
  handResult: HandResult | null
  onClose: () => void
}) {
  if (!handResult) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50">
      <Card className="w-[500px] max-w-[90vw] bg-gradient-to-br from-yellow-50 to-orange-50 border-yellow-200 shadow-2xl">
        <CardHeader className="flex flex-row items-center justify-between pb-4">
          <CardTitle className="text-2xl font-bold text-gray-800">Hand Complete!</CardTitle>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="w-5 h-5" />
          </Button>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Winner Section */}
          <div className="text-center bg-green-100 p-4 rounded-lg">
            <h3 className="text-xl font-bold text-green-800 mb-2">
              🏆 Winner: {handResult.player_names[handResult.winner]}
            </h3>
            {/* Winning hand is no longer shown */}
          </div>

          {/* Community Cards */}
          {handResult.community_cards && handResult.community_cards.length > 0 && (
            <div className="text-center">
              <h4 className="font-bold text-gray-800 text-lg mb-2">Community Cards</h4>
              <CardDisplay cards={handResult.community_cards} />
            </div>
          )}

          {/* Results Section */}
          <div className="space-y-3">
            <h4 className="font-bold text-gray-800 text-lg border-b pb-2">Hand Results:</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {handResult.payoffs.map((payoff, index) => (
                <div key={index} className="bg-white p-4 rounded-lg border shadow-sm">
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex-1">
                      <span className="font-bold text-lg">{handResult.player_names[index]}</span>
                      {index === handResult.winner && (
                        <Badge className="ml-2 bg-yellow-500 text-yellow-900">Winner</Badge>
                      )}
                      {handResult.folded && handResult.folded[index] && (
                        <Badge variant="destructive" className="ml-2">Folded</Badge>
                      )}
                    </div>
                    <div className="text-right">
                      <div className={`text-xl font-bold ${payoff >= 0 ? "text-green-600" : "text-red-600"}`}>
                        {payoff >= 0 ? "+" : ""}
                        {payoff}
                      </div>
                      <div className="text-sm text-gray-500">New Stack: {"$" + handResult.final_stacks[index]}</div>
                    </div>
                  </div>

                  {/* Hand Rank Display */}
                  {handResult.hand_ranks && handResult.hand_ranks[index] && (
                    <div className="mt-2 pt-2 border-t border-gray-200">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-600">Hand:</span>
                        <span className="text-sm font-medium text-gray-800">{handResult.hand_ranks[index]}</span>
                      </div>
                      {handResult.player_hands && handResult.player_hands[index] && (
                        <CardDisplay cards={handResult.player_hands[index]} />
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Continue Button */}
          <div className="pt-4">
            <Button
              onClick={onClose}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 text-lg font-semibold"
            >
              Continue to Next Hand
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default function PokerGame() {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [loading, setLoading] = useState(false)
  const [gameStarted, setGameStarted] = useState(false)
  const [playerActions, setPlayerActions] = useState<{ [key: number]: { action: string; timestamp: number } }>({})
  const [handResult, setHandResult] = useState<HandResult | null>(null)

  const updatePlayerAction = (playerId: number, action: string) => {
    setPlayerActions((prev) => ({
      ...prev,
      [playerId]: { action, timestamp: Date.now() },
    }))
  }

  const startNewGame = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/poker/start", { method: "POST" })
      const data = await response.json()
      setGameState(data)
      setGameStarted(true)
      setPlayerActions({})
      setHandResult(null)
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
    updatePlayerAction(0, actionName)

    try {
      const response = await fetch("/api/poker/action", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action_id: actionId }),
      })
      const data = await response.json()

      // AIのアクションを表示
      if (data.players) {
        data.players.forEach((player: any) => {
          if (!player.is_human && player.last_action) {
            updatePlayerAction(player.id, player.last_action)
          }
        })
      }

      // ハンド結果の処理
      if (data.hand_results && gameState.players) {
        setHandResult({
          winner: data.hand_results.winner,
          payoffs: data.hand_results.payoffs,
          final_stacks: data.hand_results.final_stacks,
          player_names: gameState.players.map((p) => p.name),
          hand_ranks: data.hand_results.hand_ranks || [],
          community_cards: data.hand_results.community_cards || [],
          player_hands: data.hand_results.player_hands || [],
          folded: data.hand_results.folded || [],
        })
      }

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

      // AIのアクションを表示
      if (data.players) {
        data.players.forEach((player: any) => {
          if (!player.is_human && player.last_action) {
            updatePlayerAction(player.id, player.last_action)
          }
        })
      }

      // ハンド結果の処理
      if (data.hand_results && !handResult && gameState.players) {
        setHandResult({
          winner: data.hand_results.winner,
          payoffs: data.hand_results.payoffs,
          final_stacks: data.hand_results.final_stacks,
          player_names: gameState.players.map((p) => p.name),
          hand_ranks: data.hand_results.hand_ranks || [],
          community_cards: data.hand_results.community_cards || [],
          player_hands: data.hand_results.player_hands || [],
          folded: data.hand_results.folded || [],
        })
      }

      setGameState(data)
    } catch (error) {
      console.error("Failed to poll game state:", error)
    }
  }

  const handleHandResultClose = () => {
    setHandResult(null)
    // モーダルを閉じた後、少し待ってから次のハンドの状態を取得
    setTimeout(() => {
      pollGameState()
    }, 500)
  }

  useEffect(() => {
    if (!gameStarted) return
    const interval = setInterval(pollGameState, 1500)
    return () => clearInterval(interval)
  }, [gameStarted, gameState])

  // アクション表示を3秒後に消す
  useEffect(() => {
    const now = Date.now()
    const timeouts: NodeJS.Timeout[] = []

    Object.entries(playerActions).forEach(([playerId, { timestamp }]) => {
      const remainingTime = Math.max(0, 3000 - (now - timestamp))
      if (remainingTime > 0) {
        const timeout = setTimeout(() => {
          setPlayerActions((prev) => {
            const newActions = { ...prev }
            delete newActions[Number.parseInt(playerId)]
            return newActions
          })
        }, remainingTime)
        timeouts.push(timeout)
      }
    })

    return () => timeouts.forEach(clearTimeout)
  }, [playerActions])

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
          <div className="flex justify-center gap-6 text-white text-sm">
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>
                Stage:{" "}
                {gameState.stage ? gameState.stage.charAt(0).toUpperCase() + gameState.stage.slice(1) : "Unknown"}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Coins className="w-4 h-4" />
              <span>Pot: {"$" + gameState.pot}</span>
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
              {Array.isArray(gameState.community_cards) && gameState.community_cards.length > 0 ? (
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
            {gameState.players.map((player, idx) => {
              // 固定３人分の配置クラス
              const positionClasses = [
                "absolute top-0 left-8", // idx=0: AI1
                "absolute top-0 right-8", // idx=1: AI2
                "absolute bottom-0 left-1/2 transform -translate-x-1/2", // idx=2: You
              ]
              const isAI = !player.is_human
              // カード隠しやバッジ色なども idx で使い分け
              return (
                <div key={player.id} className={positionClasses[idx]}>
                  <div className="relative">
                    <PlayerActionIndicator
                      action={playerActions[player.id]?.action}
                      isVisible={!!playerActions[player.id]}
                      isAI={isAI}
                    />
                    <Card
                      className={`${
                        isAI
                          ? idx === 0
                            ? "w-48 bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200"
                            : "w-48 bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200"
                          : "w-60 bg-gradient-to-br from-green-50 to-green-100 border-green-200"
                      } shadow-xl`}
                    >
                      <CardContent className="p-3">
                        <div className="text-center space-y-2">
                          {/* プレイヤー名 */}
                          <h4
                            className={`font-bold text-sm ${
                              isAI ? (idx === 0 ? "text-blue-800" : "text-purple-800") : "text-green-800"
                            }`}
                          >
                            {player.name}
                            {!player.is_human && ""}
                            {player.is_human && " (You)"}
                          </h4>
                          {/* 手札 */}
                          <CardDisplay cards={player.hand} hidden={isAI} />
                          {/* スタック＆ベット */}
                          <div className="flex justify-between items-center text-xs">
                            <div>
                              <div className="text-gray-600 mb-1">Stack</div>
                              <ChipStack amount={player.stack} size="sm" />
                            </div>
                            <div>
                              <div className="text-gray-600 mb-1">Bet</div>
                              <ChipStack amount={player.in_chips} size="sm" />
                            </div>
                          </div>
                          {/* Folded バッジ */}
                          {player.folded && (
                            <Badge variant="destructive" className="text-xs">
                              Folded
                            </Badge>
                          )}
                          {/* Current Player バッジ */}
                          {gameState.current_player === idx && (
                            <Badge
                              className={`text-xs ${
                                isAI ? (idx === 0 ? "bg-blue-600" : "bg-purple-600") : "bg-green-600"
                              }`}
                            >
                              {player.is_human ? "Your Turn" : "Current"}
                            </Badge>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )
            })}
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

          {/* Game Over */}
          {gameState.game_over && (
            <div className="mt-6 text-center">
              <Card className="w-96 mx-auto bg-gradient-to-br from-yellow-50 to-orange-50 border-yellow-200 shadow-xl">
                <CardContent className="p-6">
                  <h3 className="text-xl font-bold mb-4 text-gray-800">Game Over!</h3>
                  {gameState.winner !== undefined && gameState.players && (
                    <p className="mb-4 text-lg font-semibold text-green-700">
                      Winner: {gameState.players[gameState.winner]?.name}
                    </p>
                  )}
                  <Button onClick={startNewGame} className="w-full bg-blue-600 hover:bg-blue-700">
                    Start New Game
                  </Button>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>

      {/* Hand Result Modal */}
      <HandResultModal handResult={handResult} onClose={handleHandResultClose} />

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
