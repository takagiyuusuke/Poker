#!/usr/bin/env python3
"""
FastAPI backend for the poker web application.
This script loads your trained DQN model and provides API endpoints
for the Next.js frontend to interact with the game.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import numpy as np
from pathlib import Path
import rlcard
import time
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple

# Import your DQN classes (assuming they're in the same directory or properly installed)
# You'll need to adjust these imports based on your actual file structure
import sys
sys.path.append('.')  # Add current directory to path

# Import from your poker training script
from poker3 import (
    DQNAgent, extract_state, map_id_to_name, map_action_to_id,
    LimitholdemRuleAgentV1, START_BANKROLL, DEVICE, STATE_DIM
)

app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global game state
current_game = None
dqn_agents = []
# Keep results of the most recently finished hand so the frontend can
# display them even after the environment has been reset for the next
# hand.
last_hand_results = None

class ActionRequest(BaseModel):
    action_id: int

class GameState(BaseModel):
    current_player: int
    stage: str
    pot: int
    community_cards: List[str]
    players: List[Dict[str, Any]]
    legal_actions: Dict[int, str]
    q_values: Optional[Dict[str, float]] = None
    recommended_action: Optional[str] = None
    game_over: bool = False
    winner: Optional[int] = None
    hand_over: bool = False
    hand_results: Optional[Dict[str, Any]] = None
    last_action: Optional[Dict[str, str]] = None  # プレイヤーごとの最後のアクション

def load_dqn_agents(checkpoint_path: str) -> List[DQNAgent]:
    """Load two DQN agents from the checkpoint file."""
    agents = []
    for i in range(2):  # Create 2 AI agents
        agent = DQNAgent(action_dim=4)  # Assuming 4 actions for limit hold'em
        if Path(checkpoint_path).exists():
            agent.load_checkpoint(Path(checkpoint_path))
            print(f"Loaded DQN agent {i+1} from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained agent.")
        agents.append(agent)
    return agents

def format_card(card) -> str:
    """Convert RLCard card format to string."""
    if hasattr(card, 'suit') and hasattr(card, 'rank'):
        return f"{card.rank}{card.suit}"
    return str(card)

def get_q_values_and_recommendation(agent: DQNAgent, env, obs: dict, current_player_id: int) -> tuple:
    """Get Q-values and recommended action from DQN agent."""
    try:
        raw = obs['raw_obs']
        op1_id = (current_player_id + 1) % 3
        op2_id = (current_player_id + 2) % 3
        op1_raw = env.get_state(op1_id)['raw_obs']
        op2_raw = env.get_state(op2_id)['raw_obs']
        state_vec = extract_state(raw, op1_raw, op2_raw)
        legal_actions = list(obs['legal_actions'].keys())
        
        with torch.no_grad():
            q_values = agent.policy_net(state_vec.unsqueeze(0).to(DEVICE)).squeeze(0)
            
        # Create mask for legal actions
        mask = torch.full((agent.action_dim,), -float('inf'), device=DEVICE)
        mask[legal_actions] = 0
        q_values_legal = q_values + mask
        
        # Get action names and Q-values
        q_dict = {}
        for action_id in legal_actions:
            action_name = map_id_to_name(obs, action_id)
            q_dict[action_name] = float(q_values[action_id])
        
        # Get recommended action
        best_action_id = int(torch.argmax(q_values_legal))
        recommended_action = map_id_to_name(obs, best_action_id)
        
        return q_dict, recommended_action
    except Exception as e:
        print(f"Error getting Q-values: {e}")
        return None, None

def detect_winner(env) -> Optional[int]:
    """Detect the winner of the current hand or game."""
    try:
        if env.is_over():
            payoffs = env.get_payoffs()
            if payoffs:
                return int(np.argmax(payoffs))
    except:
        pass
    return None

def is_hand_over(env) -> bool:
    """Check if the current hand is over."""
    return env.is_over()

def get_hand_rank_name(rank_id: int) -> str:
    """Convert RLCard hand rank ID to a human readable name."""
    hand_ranks = {
        1: "High Card",
        2: "One Pair",
        3: "Two Pair",
        4: "Three of a Kind",
        5: "Straight",
        6: "Flush",
        7: "Full House",
        8: "Four of a Kind",
        9: "Straight Flush",
    }

    return hand_ranks.get(rank_id, f"Unknown Rank ({rank_id})")

def get_player_hand_rank(env, player_id):
    """Return player's hand rank as a string."""
    try:
        if env.is_over() and hasattr(env.game, 'public_cards'):
            player = env.game.players[player_id]
            if hasattr(player, 'hand') and player.hand:
                from rlcard.games.limitholdem.utils import Hand

                def to_eval_str(c):
                    """Return a card string in suit-first format for evaluation."""
                    if hasattr(c, "suit") and hasattr(c, "rank"):
                        return f"{c.suit}{c.rank}"
                    c = str(c)
                    # Convert rank-first strings like "AS" to "SA"
                    if len(c) == 2 and c[0] in "23456789TJQKA" and c[1] in "CDHS":
                        return f"{c[1]}{c[0]}"
                    return c

                cards = [to_eval_str(c) for c in player.hand + env.game.public_cards]
                if len(cards) == 7:
                    hand = Hand(cards)
                    hand.evaluateHand()
                    return get_hand_rank_name(int(hand.category))
    except Exception as e:
        print(f"Error getting hand rank for player {player_id}: {e}")

    # デフォルト値
    return "Unknown Hand"

def convert_game_state(env, obs: dict, current_player_id: int, ai_last_actions=None) -> GameState:
    try:
        stage = obs['raw_obs'].get('stage', 'preflop') if obs and 'raw_obs' in obs else 'preflop'
        pot = sum(p.in_chips for p in env.game.players) if hasattr(env.game, 'players') else 0
        community_cards = []
        if hasattr(env.game, 'public_cards'):
            community_cards = [format_card(card) for card in env.game.public_cards]
        
        players = []
        game_players = env.game.players if hasattr(env.game, 'players') else []
        
        for i in range(3):
            if i < len(game_players):
                player = game_players[i]
                is_human = (i == 0)
                player_name = "You" if is_human else f"AI Agent {i}"
                hand = []
                if is_human and hasattr(player, 'hand') and player.hand:
                    hand = [format_card(card) for card in player.hand]
                elif not is_human:
                    hand = ["??", "??"] if hasattr(player, 'hand') and player.hand and len(player.hand) > 0 else []
                
                last_action = None
                if ai_last_actions and i in ai_last_actions:
                    last_action = ai_last_actions[i]
                
                players.append({
                    "id": i,
                    "name": player_name,
                    "stack": getattr(player, 'stack', START_BANKROLL),
                    "hand": hand,
                    "in_chips": getattr(player, 'in_chips', 0),
                    "folded": getattr(player, 'folded', False),
                    "is_human": is_human,
                    "last_action": last_action,
                })
            else:
                players.append({
                    "id": i,
                    "name": f"AI Agent {i}",
                    "stack": START_BANKROLL,
                    "hand": ["??", "??"],
                    "in_chips": 0,
                    "folded": False,
                    "is_human": False,
                    "last_action": None,
                })
        
        legal_actions = {}
        if obs and 'legal_actions' in obs:
            for action_id in obs['legal_actions']:
                action_name = map_id_to_name(obs, action_id)
                legal_actions[action_id] = action_name
        
        q_values = None
        recommended_action = None
        if len(dqn_agents) > 0:
            agent = dqn_agents[current_player_id] if current_player_id < len(dqn_agents) else dqn_agents[0]
            q_values, recommended_action = get_q_values_and_recommendation(agent, env, obs, current_player_id)
        
        global last_hand_results

        winner = detect_winner(env)
        hand_over = is_hand_over(env)
        game_over = hand_over and any(p["stack"] <= 0 for p in players)

        hand_results = None
        if last_hand_results:
            hand_results = last_hand_results
            hand_over = True
            last_hand_results = None
        elif hand_over and hasattr(env, '_last_payoffs'):
            # 各プレイヤーの手札の強さを取得
            hand_ranks = []
            for i in range(3):
                hand_rank = get_player_hand_rank(env, i)
                hand_ranks.append(hand_rank)

            hand_results = {
                "winner": int(np.argmax(env._last_payoffs)),
                "payoffs": env._last_payoffs,
                "final_stacks": [p["stack"] for p in players],
                "hand_ranks": hand_ranks,
                "community_cards": [format_card(c) for c in env.game.public_cards],
                "player_hands": [[format_card(c) for c in pl.hand] for pl in env.game.players],
                "folded": [getattr(pl, "folded", False) for pl in env.game.players],
            }
        
        return GameState(
            current_player=current_player_id,
            stage=stage,
            pot=pot,
            community_cards=community_cards,
            players=players,
            legal_actions=legal_actions,
            q_values=q_values,
            recommended_action=recommended_action,
            game_over=game_over,
            winner=winner,
            hand_over=hand_over,
            hand_results=hand_results
        )
    except Exception as e:
        print(f"Error converting game state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert game state: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load the DQN agents on startup."""
    global dqn_agents
    # You'll need to specify the path to your checkpoint file
    checkpoint_path = "checkpoints/dqn_final.pth"  # Adjust this path
    dqn_agents = load_dqn_agents(checkpoint_path)

@app.post("/start_game")
async def start_game():
    global current_game
    try:
        env = rlcard.make(
            'limit-holdem',
            config={
                'seed': int(time.time()),          # 任意の乱数シード
                'game_num_players': 3,             # ← ここが最重要!!
                'allow_step_back': False,
            }
        )
        
        # 3人分のエージェントを必ずセット
        # フロントから行動が届くまで呼ばれないダミー
        class HumanProxyAgent:
            use_raw = True
            def step(self, obs):                       # never called
                return list(obs['legal_actions'].keys())[0]
    
        human_agent = HumanProxyAgent()
        ai_agent_1 = dqn_agents[0] if len(dqn_agents) > 0 else LimitholdemRuleAgentV1()
        ai_agent_2 = dqn_agents[1] if len(dqn_agents) > 1 else LimitholdemRuleAgentV1()
        env.set_agents([human_agent, ai_agent_1, ai_agent_2])
        obs, current_player_id = env.reset()
        
        # プレイヤー人数チェック
        if not hasattr(env.game, 'players') or len(env.game.players) != 3:
            raise RuntimeError(f"RLCard環境のプレイヤー数が3人ではありません: {len(env.game.players) if hasattr(env.game, 'players') else 'N/A'}")
        
        for i, player in enumerate(env.game.players):
            player.stack = START_BANKROLL
        
        current_game = {
            'env': env,
            'obs': obs,
            'current_player_id': current_player_id,
            'stacks': [START_BANKROLL] * 3
        }
        
        # reset直後にAIターンなら自動でAIターンを進める
        ai_last_actions = {}
        obs, current_player_id, ai_last_actions = advance_ai_turns(
            env, obs, current_player_id, context="start_game", ai_last_actions=ai_last_actions
        )
        
        # ハンドが終了していれば結果を保存
        if env.is_over():
            payoffs = env.get_payoffs()
            payoffs_list = payoffs.tolist() if hasattr(payoffs, "tolist") else list(payoffs)
            env._last_payoffs = payoffs_list

            # スタック更新
            for i, player in enumerate(env.game.players):
                player.stack += payoffs[i]

            hand_ranks = [get_player_hand_rank(env, i) for i in range(3)]

            global last_hand_results
            last_hand_results = {
                "winner": int(np.argmax(payoffs_list)),
                "payoffs": payoffs_list,
                "final_stacks": [p.stack for p in env.game.players],
                "hand_ranks": hand_ranks,
                "community_cards": [format_card(c) for c in env.game.public_cards],
                "player_hands": [[format_card(c) for c in p.hand] for p in env.game.players],
                "folded": [getattr(p, "folded", False) for p in env.game.players],
            }

            if not any(p.stack <= 0 for p in env.game.players):
                saved = [p.stack for p in env.game.players]
                obs, current_player_id = env.reset()
                for i, player in enumerate(env.game.players):
                    player.stack = saved[i]

        current_game['obs'] = obs
        current_game['current_player_id'] = current_player_id
        game_state = convert_game_state(env, obs, current_player_id, ai_last_actions=ai_last_actions)
        return game_state.model_dump()
        
    except Exception as e:
        print(f"Error starting game: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start game: {str(e)}")

async def handle_ai_turn(env, obs: dict, current_player_id: int, timeout: int = 10) -> Tuple[dict, int, Optional[str]]:
    """AIの行動を処理し、タイムアウトした場合はランダムな合法手を選択"""
    try:
        # タイムアウト付きでAIの行動を決定
        async def ai_action_task():
            legal_actions = list(obs['legal_actions'].keys())
            if not legal_actions:
                raise ValueError("No legal actions available")

            if current_player_id <= len(dqn_agents):
                agent = dqn_agents[current_player_id - 1]
                raw = obs['raw_obs']
                op1_raw = env.get_state((current_player_id + 1) % 3)['raw_obs']
                op2_raw = env.get_state((current_player_id + 2) % 3)['raw_obs']
                state_vec = extract_state(raw, op1_raw, op2_raw)
                ai_action = agent.select_action(state_vec, legal_actions, is_greedy=True)
            else:
                rule_agent = LimitholdemRuleAgentV1()
                ai_action_obj = rule_agent.step(obs)
                ai_action = map_action_to_id(obs, getattr(ai_action_obj, 'name', str(ai_action_obj)))
                if ai_action is None:
                    ai_action = legal_actions[0]

            ai_action_name = map_id_to_name(obs, ai_action)
            new_obs, new_player_id = env.step(ai_action)
            return new_obs, new_player_id, ai_action_name

        # タイムアウト処理
        try:
            return await asyncio.wait_for(ai_action_task(), timeout)
        except asyncio.TimeoutError:
            # タイムアウト時はランダムな合法手を選択
            legal_actions = list(obs['legal_actions'].keys())
            random_action = random.choice(legal_actions)
            random_action_name = map_id_to_name(obs, random_action)
            print(f"[AI] Timeout for player {current_player_id}, taking random action: {random_action_name}")
            new_obs, new_player_id = env.step(random_action)
            return new_obs, new_player_id, random_action_name

    except Exception as e:
        print(f"[AI] Error in handle_ai_turn: {e}")
        # エラー時もランダムな合法手を選択
        legal_actions = list(obs['legal_actions'].keys())
        random_action = random.choice(legal_actions)
        random_action_name = map_id_to_name(obs, random_action)
        print(f"[AI] Error recovery: taking random action: {random_action_name}")
        new_obs, new_player_id = env.step(random_action)
        return new_obs, new_player_id, random_action_name

def advance_ai_turns(env, obs: dict, current_player_id: int, *, context: str = "", ai_last_actions=None, max_steps: int = 20) -> Tuple[dict, int, Dict[int, str]]:
    """Advance AI turns until it's the human player's turn or the hand ends."""
    if ai_last_actions is None:
        ai_last_actions = {}

    step_count = 0
    prev_state = None

    while not env.is_over() and current_player_id != 0:
        step_count += 1
        print(
            f"[AI] ({context}) Turn {step_count}, Player {current_player_id}, "
            f"Stacks: {[p.stack for p in env.game.players]}, Bets: {[p.in_chips for p in env.game.players]}"
        )

        if step_count > max_steps:
            print("[AI] Max steps reached, breaking loop")
            break

        current_state = (
            current_player_id,
            [p.stack for p in env.game.players],
            [p.in_chips for p in env.game.players],
        )
        if current_state == prev_state:
            print("[AI] Detected state loop, breaking")
            break
        prev_state = current_state

        legal_actions = list(obs["legal_actions"].keys())
        if not legal_actions:
            print("[AI] No legal actions, breaking.")
            break

        if current_player_id <= len(dqn_agents):
            agent = dqn_agents[current_player_id - 1]
            raw = obs["raw_obs"]
            op1_raw = env.get_state((current_player_id + 1) % 3)["raw_obs"]
            op2_raw = env.get_state((current_player_id + 2) % 3)["raw_obs"]
            state_vec = extract_state(raw, op1_raw, op2_raw)
            ai_action = agent.select_action(state_vec, legal_actions, is_greedy=True)
            ai_action_name = map_id_to_name(obs, ai_action)
            ai_last_actions[current_player_id] = ai_action_name
        else:
            rule_agent = LimitholdemRuleAgentV1()
            ai_action_obj = rule_agent.step(obs)
            ai_action = map_action_to_id(obs, getattr(ai_action_obj, "name", str(ai_action_obj)))
            if ai_action is None:
                ai_action = legal_actions[0]
            ai_action_name = map_id_to_name(obs, ai_action)
            ai_last_actions[current_player_id] = ai_action_name

        print(
            f"[AI] ({context}) player={current_player_id}, action={ai_action_name}, "
            f"stack={[p.stack for p in env.game.players]}, in_chips={[p.in_chips for p in env.game.players]}"
        )
        obs, current_player_id = env.step(ai_action)
        print(
            f"[AI] ({context}) after step: stacks={[p.stack for p in env.game.players]}, "
            f"in_chips={[p.in_chips for p in env.game.players]}, hands={[getattr(p, 'hand', []) for p in env.game.players]}"
        )

    return obs, current_player_id, ai_last_actions

@app.post("/make_action")
async def make_action(request: ActionRequest):
    global current_game
    if not current_game:
        raise HTTPException(status_code=400, detail="No active game")
    try:
        env = current_game['env']
        action_id = request.action_id
        # 現在のスタックを保存
        current_stacks = [p.stack for p in env.game.players]
        
        obs, current_player_id = env.step(action_id)
        ai_last_actions = {}
        max_ai_steps = 20
        ai_step_count = 0
        prev_state = None

        while not env.is_over() and current_player_id != 0:
            ai_step_count += 1
            print(f"[AI] Turn {ai_step_count}, Player {current_player_id}, Stacks: {[p.stack for p in env.game.players]}, Bets: {[p.in_chips for p in env.game.players]}")
            
            if ai_step_count > max_ai_steps:
                print("[AI] Max steps reached, breaking loop")
                break

            # 状態が変化しないループを検出
            current_state = (current_player_id, [p.stack for p in env.game.players], [p.in_chips for p in env.game.players])
            if current_state == prev_state:
                print("[AI] Detected state loop, breaking")
                break
            prev_state = current_state

            try:
                # AIの行動をタイムアウト付きで処理
                new_obs, new_player_id, action_name = await handle_ai_turn(env, obs, current_player_id)
                if action_name:
                    ai_last_actions[current_player_id] = action_name
                obs, current_player_id = new_obs, new_player_id

            except Exception as e:
                print(f"[AI] Error during AI turn: {e}")
                break

        if env.is_over():
            payoffs = env.get_payoffs()
            payoffs_list = payoffs.tolist() if hasattr(payoffs, "tolist") else list(payoffs)
            env._last_payoffs = payoffs_list

            print(f"[HAND_END] Payoffs: {payoffs}")
            print(f"[HAND_END] Current stacks: {current_stacks}")

            # スタックの更新と保存
            for i, player in enumerate(env.game.players):
                old_stack = current_stacks[i]
                player.stack = old_stack + payoffs[i]
                if player.stack < 0:
                    player.stack = 0
                print(f"[HAND_END] Player {i}: {old_stack} + {payoffs[i]} = {player.stack}")

            # 手札評価
            hand_ranks = []
            for i in range(3):
                hand_ranks.append(get_player_hand_rank(env, i))

            global last_hand_results
            last_hand_results = {
                "winner": int(np.argmax(payoffs_list)),
                "payoffs": payoffs_list,
                "final_stacks": [p.stack for p in env.game.players],
                "hand_ranks": hand_ranks,
                "community_cards": [format_card(c) for c in env.game.public_cards],
                "player_hands": [[format_card(c) for c in p.hand] for p in env.game.players],
                "folded": [getattr(p, "folded", False) for p in env.game.players],
            }

            # 新しいハンドの開始（誰も破産していない場合）
            if not any(p.stack <= 0 for p in env.game.players):
                saved_stacks = [p.stack for p in env.game.players]
                print(f"[NEW_HAND] Starting new hand with stacks: {saved_stacks}")
                obs, current_player_id = env.reset()
                for i, player in enumerate(env.game.players):
                    player.stack = saved_stacks[i]
                print(f"[NEW_HAND] Restored stacks: {[p.stack for p in env.game.players]}")
                # 新しいハンド開始直後にAIターンなら進める
                obs, current_player_id, new_ai_actions = advance_ai_turns(
                    env, obs, current_player_id, context="new_hand", ai_last_actions=ai_last_actions
                )
                ai_last_actions.update(new_ai_actions)

        current_game['obs'] = obs
        current_game['current_player_id'] = current_player_id
        
        game_state = convert_game_state(env, obs, current_player_id, ai_last_actions=ai_last_actions)
        return game_state.model_dump()

    except Exception as e:
        print(f"Error in make_action: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_state")
async def get_state():
    """Get the current game state."""
    global current_game
    
    if not current_game:
        raise HTTPException(status_code=400, detail="No active game")
    
    try:
        env = current_game['env']
        obs = current_game['obs']
        current_player_id = current_game['current_player_id']
        
        game_state = convert_game_state(env, obs, current_player_id)
        return game_state.dict()
        
    except Exception as e:
        print(f"Error getting state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get state: {str(e)}")

if __name__ == "__main__":
    print("Starting Poker Backend Server...")
    print("Make sure you have your checkpoint file at: checkpoints/dqn_final.pth")
    uvicorn.run(app, host="0.0.0.0", port=8000)
