import streamlit as st
import gymnasium as gym
from PIL import Image
import pandas as pd
import altair as alt
import os, time

##For PPO
import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────────────────────
# .  PPO - policy network
# ────────────────────────────────
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        #input -> first hidden layer: 4 inputs -> 64 neurons
        self.fc1 = nn.Linear(4, 64)
        #first hidden layer -> second hidden layer: 64 -> 64
        self.fc2 = nn.Linear(64, 64)
        #second hidden layer -> output action probabilities
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        #Pass through first layer with ReLU
        x = F.relu(self.fc1(x))
        #Pass through second layer with ReLU
        x = F.relu(self.fc2(x))
        #Output layer with softmax for probabilities
        x = F.softmax(self.fc3(x), dim= -1)
        return x


# ────────────────────────────────
# .  PPO - value function network
# ────────────────────────────────
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        #input -> first hidden layer: 4 inputs -> 64 neurons
        self.fc1 = nn.Linear(4, 64)
        #first hidden layer -> second hidden layer: 64 -> 64
        self.fc2 = nn.Linear(64, 64)
        #second hidden layer -> output predicted future value for given state
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        #Pass through first layer with ReLU
        x = F.relu(self.fc1(x))
        #Pass through second layer with ReLU
        x = F.relu(self.fc2(x))
        #raw single output (no activation function since they would restrict it)
        x = self.fc3(x)
        return x


# ────────────────────────────────
# 0.  Config
# ────────────────────────────────
os.environ["SDL_VIDEODRIVER"] = "dummy"
st.set_page_config(layout="wide")
st.title("CartPole Environment Visualization")
st.write("Watch CartPole balance itself with random left/right actions.")

# ────────────────────────────────
# 1.  Layout
# ────────────────────────────────
with st.container():
    img_col, charts_col = st.columns(2)

    # Q1: live RGB frame
    with img_col:
        frame_ph = st.empty()

    # Q2-Q4: placeholders for four charts (2 × 2 grid)
    with charts_col:
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        pos_ph  = r1c1.empty()
        vel_ph  = r1c2.empty()
        ang_ph  = r2c1.empty()
        angv_ph = r2c2.empty()

        # Add training metrics section
        st.markdown("### Training Metrics")
        r3c1, r3c2 = st.columns(2)
        r4c1, r4c2 = st.columns(2)

        action_probs_ph = r3c1.empty()
        value_pred_ph = r3c2.empty()
        advantages_ph = r4c1.empty()
        td_errors_ph = r4c2.empty()

# ────────────────────────────────
# 2.  DataFrames for streaming
# ────────────────────────────────
pos_df  = pd.DataFrame(columns=["step", "pos"])
vel_df  = pd.DataFrame(columns=["step", "vel"])
ang_df  = pd.DataFrame(columns=["step", "angle"])
angv_df = pd.DataFrame(columns=["step", "ang_vel"])

# New DataFrames for training metrics
action_probs_df = pd.DataFrame(columns=["step", "left_prob", "right_prob"])
value_pred_df = pd.DataFrame(columns=["step", "value"])
advantages_df = pd.DataFrame(columns=["step", "advantage"])
td_errors_df = pd.DataFrame(columns=["step", "td_error"])

def make_chart(df, y_field, title):
    """Return a 150-px-tall Altair line chart for the given DataFrame."""
    if df.empty:                                       # Altair needs at least 1 row
        df = pd.DataFrame({ "step": [0], y_field: [0] })
    return (
        alt.Chart(df)
           .mark_line()
           .encode(x="step:Q", y=f"{y_field}:Q")
           .properties(height=250, title=title)
    )

# ────────────────────────────────
# 3.  Gym loop with live updates
# ────────────────────────────────
env = gym.make("CartPole-v1", render_mode="rgb_array")
policy_net = PolicyNetwork() #initialize PolicyNetwork (state -> NN -> action probabilities)
value_net = ValueNetwork() #initialize ValueNetwork (state -> NN -> predicted future value)
num_episodes = 10 #number of policy iterations
max_steps_per_episode = 200 #max number of steps (actions taken) in between policy iterations
discount_factor = 0.99 #how much to discount future rewards when calculating advantage for a time step
lambda_gae = 0.95 #controls the balance between bias and variance for GAE (how far into the future does each advantage calculation look)

#lists to record trajectory for current episode (reset after each episode)
states = []
actions = []
rewards = []
next_states = []
dones = []
log_probs = [] #log probabilities of actions taken before policy update

obs, _ = env.reset()

try:
    for episode in range(num_episodes):
        #reset environment and trajectory recordings between episodes
        obs, _= env.reset()
        pos_df  = pos_df.iloc[0:0]
        vel_df  = vel_df.iloc[0:0]
        ang_df  = ang_df.iloc[0:0]
        angv_df = angv_df.iloc[0:0]
        
        # Reset training metric DataFrames
        action_probs_df = action_probs_df.iloc[0:0]
        value_pred_df = value_pred_df.iloc[0:0]
        advantages_df = advantages_df.iloc[0:0]
        td_errors_df = td_errors_df.iloc[0:0]
        
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            ##action = env.action_space.sample() #random action
            #take an observation of the environment, call it state
            state = obs
            #convert state to tensor for PolicyNetwork
            state_tensor = torch.tensor(state, dtype=torch.float32)
            #run forward pass to get action probabilities
            action_probs = policy_net(state_tensor)
            #get the distribution of action probabilities (needed to calculate log probs)
            dist = torch.distributions.Categorical(action_probs)
            #sample action from action probabilities distribution
            action = dist.sample().item()
            #calculate log probabilities of action probs (needed for policy loss)
            log_prob = dist.log_prob(torch.tensor(action)).item()
            #run forward pass to predict future value ("numerical score of how good is your current situation")
            predicted_value = value_net(state_tensor)
            
            #step the environment with the chosen action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            #record trajectory for current episode
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            log_probs.append(log_prob) #record log_probs for policy loss calculation

            # Append latest observation (update dataframes that record state for visualization)
            pos_df.loc[len(pos_df)]  = [step, state[0]]
            vel_df.loc[len(vel_df)]  = [step, state[1]]
            ang_df.loc[len(ang_df)]  = [step, state[2]]
            angv_df.loc[len(angv_df)] = [step, state[3]]

            # Update training metric DataFrames
            action_probs_df.loc[len(action_probs_df)] = [step, action_probs[0].item(), action_probs[1].item()]
            value_pred_df.loc[len(value_pred_df)] = [step, predicted_value.item()]
            advantages_df.loc[len(advantages_df)] = [step, 0]  # Will be updated after episode
            td_errors_df.loc[len(td_errors_df)] = [step, 0]    # Will be updated after episode

            # Update each chart (150 px tall → never spills)
            pos_ph.altair_chart( make_chart(pos_df,  "pos",       "Cart Position"), use_container_width=True )
            vel_ph.altair_chart( make_chart(vel_df,  "vel",       "Cart Velocity"), use_container_width=True )
            ang_ph.altair_chart( make_chart(ang_df,  "angle",     "Pole Angle"),    use_container_width=True )
            angv_ph.altair_chart( make_chart(angv_df, "ang_vel",  "Pole Angular Velocity"), use_container_width=True )

            # Update training metric charts
            action_probs_ph.altair_chart( make_chart(action_probs_df, "left_prob", "Action Probabilities"), use_container_width=True )
            value_pred_ph.altair_chart( make_chart(value_pred_df, "value", "Value Predictions"), use_container_width=True )
            advantages_ph.altair_chart( make_chart(advantages_df, "advantage", "Advantages"), use_container_width=True )
            td_errors_ph.altair_chart( make_chart(td_errors_df, "td_error", "TD Errors"), use_container_width=True )

            # Update frame in Quadrant 1
            frame_ph.image(
                Image.fromarray(env.render()),
                caption=f"Step {step} — {'Left' if action==0 else 'Right'}",
                use_container_width=True,
            )

            time.sleep(0.05)
            obs = next_state
            step += 1
        #(convert to tensor) state and next_state lists for advantage calculation
        states_tensor = torch.tensor(states, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        #predict value for each state using ValueNetwork
        values = value_net(states_tensor).squeeze()
        next_values = value_net(next_states_tensor).squeeze()
        #(convert to tensor) rewards and dones lists for temporal difference calculation
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        #compute TD errors
        td_errors = rewards_tensor + (1 - dones_tensor) * discount_factor * next_values - values
        #compute ADVANTAGES using GAE (generalized advantage estimation)
        advantages = torch.zeros_like(td_errors) #create advantages tensor same size as td_errors
        advantage = 0
        for t in reversed(range(len(td_errors))): #loop through time steps in reverse order
            if dones[t]: #if pole falls or episode ends
                advantage = td_errors[t] #advantage is just current advantage (no future to account for)
            else:
                advantage = td_errors[t] + discount_factor * lambda_gae * advantage
            advantages[t] = advantage
        
        #normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update advantages and TD errors in DataFrames
        for t in range(len(advantages)):
            advantages_df.loc[t, "advantage"] = advantages[t].item()
            td_errors_df.loc[t, "td_error"] = td_errors[t].item()
        
        #print advantages
        print(f"Episode {episode} advantages: {advantages}")

        #compute POLICY LOSS
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        current_action_probs = policy_net(states_tensor)
        new_policy_prob = current_action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        old_policy_prob = torch.exp(torch.tensor(log_probs))
        ratio = new_policy_prob / old_policy_prob

        epsilon = 0.2   #clipping parameter
        term1 = ratio * advantages
        term2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

        #take the minimum and average
        policy_loss = torch.min(term1, term2)
        policy_loss = -policy_loss.mean()
        
        #print policy loss
        print(f"Episode {episode} policy loss: {policy_loss.item()}")

        #compute VALUE LOSS
        returns = torch.zeros_like(rewards_tensor) #create tensor to store actual returns
        return_at_time_step = 0
        
        #compute actual (not predicted) future rewards for each time step
        for time_step in reversed(range(len(rewards_tensor))):
            if dones_tensor[time_step]:
                return_at_time_step = rewards_tensor[time_step]
            else:
                return_at_time_step = rewards_tensor[time_step] + discount_factor * return_at_time_step
                returns[time_step] = return_at_time_step
        
        #compute current predicted values
        current_values = value_net(states_tensor).squeeze()
        #compute value loss (MSE between predicted and actual values)
        value_loss = F.mse_loss(current_values, returns)

        #print value loss
        print(f"Episode {episode} value loss: {value_loss.item()}")

finally:
    env.close()
