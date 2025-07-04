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
from typing import Dict, List, Any, Optional
import copy
import time

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

def get_q_values_and_recommendation(agent: DQNAgent, obs: dict) -> tuple:
    """Get Q-values and recommended action from DQN agent."""
    try:
        state_vec = extract_state(obs['raw_obs'])
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

def convert_game_state(env, obs: dict, current_player_id: int) -> GameState:
    """Convert RLCard game state to our API format."""
    try:
        # Get basic game info
        stage = obs['raw_obs'].get('stage', 'preflop') if obs and 'raw_obs' in obs else 'preflop'
        pot = sum(p.in_chips for p in env.game.players) if hasattr(env.game, 'players') else 0
        
        # Get community cards
        community_cards = []
        if hasattr(env.game, 'public_cards'):
            community_cards = [format_card(card) for card in env.game.public_cards]
        
        # Get player info - ensure all 3 players are included
        players = []
        game_players = env.game.players if hasattr(env.game, 'players') else []
        
        for i in range(3):  # Explicitly handle 3 players
            if i < len(game_players):
                player = game_players[i]
                is_human = (i == 0)  # Player 0 is human
                player_name = "You" if is_human else f"AI Agent {i}"
                
                # Get player's hand (only show human player's cards)
                hand = []
                if is_human and hasattr(player, 'hand') and player.hand:
                    hand = [format_card(card) for card in player.hand]
                elif not is_human:
                    # For AI players, show placeholder cards if they have cards
                    hand = ["??", "??"] if hasattr(player, 'hand') and player.hand and len(player.hand) > 0 else []
                
                players.append({
                    "id": i,
                    "name": player_name,
                    "stack": getattr(player, 'stack', START_BANKROLL),
                    "hand": hand,
                    "in_chips": getattr(player, 'in_chips', 0),
                    "folded": getattr(player, 'folded', False),
                    "is_human": is_human
                })
            else:
                # Fallback for missing players
                players.append({
                    "id": i,
                    "name": f"AI Agent {i}",
                    "stack": START_BANKROLL,
                    "hand": ["??", "??"],
                    "in_chips": 0,
                    "folded": False,
                    "is_human": False
                })
        
        # Get legal actions
        legal_actions = {}
        if obs and 'legal_actions' in obs:
            for action_id in obs['legal_actions']:
                action_name = map_id_to_name(obs, action_id)
                legal_actions[action_id] = action_name
        
        # Get Q-values if it's the human player's turn (not just AI)
        q_values = None
        recommended_action = None
        if current_player_id == 0 and len(dqn_agents) > 0:  # Human player's turn
            # Use first AI agent to provide recommendations
            agent = dqn_agents[0]
            q_values, recommended_action = get_q_values_and_recommendation(agent, obs)
        
        # Detect winner and game state
        winner = detect_winner(env)
        hand_over = is_hand_over(env)
        game_over = hand_over and any(p["stack"] <= 0 for p in players)
        
        # Check for hand results
        hand_results = None
        if hand_over and hasattr(env, '_last_payoffs'):
            hand_results = {
                "winner": winner if winner is not None else 0,
                "payoffs": env._last_payoffs,
                "final_stacks": [p["stack"] for p in players]
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
    """Start a new poker game."""
    global current_game
    
    try:
        # Create new RLCard environment
        env = rlcard.make('limit-holdem', config={'seed': None, 'num_players': 3})
        
        # Create agents: Human (index 0), AI Agent 1 (index 1), AI Agent 2 (index 2)
        human_agent = None  # Human player doesn't need an agent object
        ai_agent_1 = dqn_agents[0] if len(dqn_agents) > 0 else LimitholdemRuleAgentV1()
        ai_agent_2 = dqn_agents[1] if len(dqn_agents) > 1 else LimitholdemRuleAgentV1()
        
        # Set agents (human player will be handled separately)
        env.set_agents([human_agent, ai_agent_1, ai_agent_2])
        
        # Initialize game
        obs, current_player_id = env.reset()
        
        # Set initial bankrolls
        for i, player in enumerate(env.game.players):
            player.stack = START_BANKROLL
        
        # Store game state
        current_game = {
            'env': env,
            'obs': obs,
            'current_player_id': current_player_id,
            'initial_stacks': [START_BANKROLL] * 3
        }
        
        # Convert and return game state
        game_state = convert_game_state(env, obs, current_player_id)
        return game_state.dict()
        
    except Exception as e:
        print(f"Error starting game: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start game: {str(e)}")

@app.post("/make_action")
async def make_action(request: ActionRequest):
    """Make an action in the current game."""
    global current_game
    
    if not current_game:
        raise HTTPException(status_code=400, detail="No active game")
    
    try:
        env = current_game['env']
        action_id = request.action_id
        
        # Store previous stacks for payoff calculation
        prev_stacks = [p.stack for p in env.game.players]
        
        # Make the action
        obs, current_player_id = env.step(action_id)
        
        # Process AI turns automatically with small delays
        while not env.is_over() and current_player_id != 0:
            # Add small delay to make AI actions visible
            time.sleep(0.5)
            
            # AI player's turn
            if current_player_id <= len(dqn_agents):
                agent = dqn_agents[current_player_id - 1]
                state_vec = extract_state(obs['raw_obs'])
                legal_actions = list(obs['legal_actions'].keys())
                ai_action = agent.select_action(state_vec, legal_actions, is_greedy=True)
            else:
                # Fallback to rule-based agent
                rule_agent = LimitholdemRuleAgentV1()
                ai_action_obj = rule_agent.step(obs)
                ai_action = map_action_to_id(obs, ai_action_obj.name if hasattr(ai_action_obj, 'name') else ai_action_obj)
                if ai_action is None:
                    ai_action = legal_actions[0] if legal_actions else 0
            
            obs, current_player_id = env.step(ai_action)
        
        # Check if hand is over
        if env.is_over():
            # Get payoffs and update stacks
            payoffs = env.get_payoffs()
            
            # Store payoffs for hand results
            env._last_payoffs = payoffs
            
            # Update stacks based on payoffs
            for i, player in enumerate(env.game.players):
                # Payoffs are already in chips, add to stack
                player.stack += payoffs[i]
                # Ensure stack doesn't go below 0
                if player.stack < 0:
                    player.stack = 0
            
            # Check if anyone is bankrupt
            bankrupt_players = [i for i, p in enumerate(env.game.players) if p.stack <= 0]
            
            if not bankrupt_players:
                # Start new hand if no one is bankrupt
                obs, current_player_id = env.reset()
        
        # Update game state
        current_game['obs'] = obs
        current_game['current_player_id'] = current_player_id
        
        # Convert and return game state
        game_state = convert_game_state(env, obs, current_player_id)
        return game_state.dict()
        
    except Exception as e:
        print(f"Error making action: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to make action: {str(e)}")

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
