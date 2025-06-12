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
st.write("Watch CartPole learn to balance the pole by moving the cart left/right.")

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
        # Training metrics section at the top
        st.markdown("### Training Metrics")
        r3c1, r3c2 = st.columns(2)
        r4c1, r4c2 = st.columns(2)
        r5c1, r5c2 = st.columns(2)

        action_probs_ph = r3c1.empty()
        value_pred_ph = r3c2.empty()
        advantages_ph = r4c1.empty()
        td_errors_ph = r4c2.empty()
        policy_loss_ph = r5c1.empty()
        value_loss_ph = r5c2.empty()

        # State charts at the bottom
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        pos_ph  = r1c1.empty()
        vel_ph  = r1c2.empty()
        ang_ph  = r2c1.empty()
        angv_ph = r2c2.empty()

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
advantages_df = pd.DataFrame(columns=["step", "advantage"]).astype({"step": "int64", "advantage": "float64"})
td_errors_df = pd.DataFrame(columns=["step", "td_error"]).astype({"step": "int64", "td_error": "float64"})
policy_loss_df = pd.DataFrame(columns=["episode", "loss"])
value_loss_df = pd.DataFrame(columns=["episode", "loss"])

global_step = 0  # Global step counter for advantages/td_errors

def make_chart(df, y_field, title, x_field="step"):
    """Return a 150-px-tall Altair line chart for the given DataFrame."""
    if df.empty:
        df = pd.DataFrame({ x_field: [0], y_field: [0] })
    return (
        alt.Chart(df)
           .mark_line()
           .encode(x=f"{x_field}:Q", y=f"{y_field}:Q")
           .properties(height=250, title=title)
    )

# ────────────────────────────────
# 3.  Gym loop with live updates
# ────────────────────────────────
env = gym.make("CartPole-v1", render_mode="rgb_array")
policy_net = PolicyNetwork() #initialize PolicyNetwork (state -> NN -> action probabilities)
value_net = ValueNetwork() #initialize ValueNetwork (state -> NN -> predicted future value)
num_episodes = 500 #number of policy iterations
max_steps_per_episode = 200 #max number of steps (actions taken) in between policy iterations
discount_factor = 0.99 #how much to discount future rewards when calculating advantage for a time step
lambda_gae = 0.95 #controls the balance between bias and variance for GAE (how far into the future does each advantage calculation look)
learning_rate = 0.0003 #learning rate for unified loss
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate) #use Adam optimizer for policy net
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=learning_rate) #use Adam optimizer for value net

# PPO-specific parameters
ppo_epochs = 4  # Number of optimization epochs per batch
batch_size = 4  # Number of episodes to collect before updating
clip_epsilon = 0.2  # PPO clipping parameter

# Storage for multiple episodes
batch_states = []
batch_actions = []
batch_rewards = []
batch_next_states = []
batch_dones = []
batch_old_log_probs = []

obs, _ = env.reset()

try:
    episode = 0
    while episode < num_episodes:
        # Collect a batch of episodes
        for batch_episode in range(batch_size):
            if episode >= num_episodes:
                break
                
            #reset environment and trajectory recordings between episodes
            obs, _= env.reset()
            pos_df  = pos_df.iloc[0:0]
            vel_df  = vel_df.iloc[0:0]
            ang_df  = ang_df.iloc[0:0]
            angv_df = angv_df.iloc[0:0]
            
            # Reset training metric DataFrames (but DO NOT reset advantages_df or td_errors_df)
            action_probs_df = action_probs_df.iloc[0:0]
            value_pred_df = value_pred_df.iloc[0:0]
            
            # Current episode trajectory
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_next_states = []
            episode_dones = []
            episode_old_log_probs = []
            
            done = False
            step = 0
            episode_adv_indices = []  # Track indices for this episode in advantages_df/td_errors_df

            while not done and step < max_steps_per_episode:
                #take an observation of the environment, call it state
                state = obs
                #convert state to tensor for PolicyNetwork
                state_tensor = torch.tensor(state, dtype=torch.float32)
                
                # Use the current policy network to get action probabilities
                with torch.no_grad():  # Don't track gradients during data collection
                    action_probs = policy_net(state_tensor)
                    #get the distribution of action probabilities (needed to calculate log probs)
                    dist = torch.distributions.Categorical(action_probs)
                    #sample action from action probabilities distribution
                    action = dist.sample().item()
                    #calculate log probabilities of action probs (needed for policy loss)
                    old_log_prob = dist.log_prob(torch.tensor(action)).item()
                
                #run forward pass to predict future value ("numerical score of how good is your current situation")
                with torch.no_grad():
                    predicted_value = value_net(state_tensor)
                
                #step the environment with the chosen action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                #record trajectory for current episode
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_next_states.append(next_state)
                episode_dones.append(done)
                episode_old_log_probs.append(old_log_prob) #record OLD log_probs from CURRENT policy

                # Append latest observation (update dataframes that record state for visualization)
                pos_df.loc[len(pos_df)]  = [step, state[0]]
                vel_df.loc[len(vel_df)]  = [step, state[1]]
                ang_df.loc[len(ang_df)]  = [step, state[2]]
                angv_df.loc[len(angv_df)] = [step, state[3]]

                # Update training metric DataFrames
                action_probs_df.loc[len(action_probs_df)] = [step, action_probs[0].item(), action_probs[1].item()]
                value_pred_df.loc[len(value_pred_df)] = [step, predicted_value.item()]
                advantages_df.loc[len(advantages_df)] = [global_step, 0]  # Will be updated after episode
                td_errors_df.loc[len(td_errors_df)] = [global_step, 0]    # Will be updated after episode
                episode_adv_indices.append(len(advantages_df) - 1)

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
                policy_loss_ph.altair_chart( make_chart(policy_loss_df, "loss", "Policy Loss", x_field="episode"), use_container_width=True )
                value_loss_ph.altair_chart( make_chart(value_loss_df, "loss", "Value Loss", x_field="episode"), use_container_width=True )

                # Update frame in Quadrant 1
                frame_ph.image(
                    Image.fromarray(env.render()),
                    caption=f"Episode {episode}, Step {step} — {'Left' if action==0 else 'Right'}",
                    use_container_width=True,
                )

                #time.sleep(0.005)
                obs = next_state
                step += 1
                global_step += 1
            
            # Add episode data to batch
            batch_states.extend(episode_states)
            batch_actions.extend(episode_actions)
            batch_rewards.extend(episode_rewards)
            batch_next_states.extend(episode_next_states)
            batch_dones.extend(episode_dones)
            batch_old_log_probs.extend(episode_old_log_probs)
            
            # Compute advantages for this episode and update DataFrames
            states_tensor = torch.tensor(episode_states, dtype=torch.float32)
            next_states_tensor = torch.tensor(episode_next_states, dtype=torch.float32)
            
            with torch.no_grad():
                values = value_net(states_tensor).squeeze()
                next_values = value_net(next_states_tensor).squeeze()
            
            rewards_tensor = torch.tensor(episode_rewards, dtype=torch.float32)
            dones_tensor = torch.tensor(episode_dones, dtype=torch.float32)
            
            # Compute TD errors
            td_errors = rewards_tensor + (1 - dones_tensor) * discount_factor * next_values - values
            
            # Compute advantages using GAE
            advantages = torch.zeros_like(td_errors)
            advantage = 0
            for t in reversed(range(len(td_errors))):
                if episode_dones[t]:
                    advantage = td_errors[t]
                else:
                    advantage = td_errors[t] + discount_factor * lambda_gae * advantage
                advantages[t] = advantage
            
            # Update advantages and TD errors in DataFrames
            for idx, t in zip(episode_adv_indices, range(len(advantages))):
                advantages_df.loc[idx, "advantage"] = advantages[t].item()
                td_errors_df.loc[idx, "td_error"] = td_errors[t].item()
            
            # Update charts
            advantages_ph.altair_chart( make_chart(advantages_df, "advantage", "Advantages"), use_container_width=True )
            td_errors_ph.altair_chart( make_chart(td_errors_df, "td_error", "TD Errors"), use_container_width=True )
            
            episode += 1
        
        # Now update the policy and value networks using the collected batch
        if len(batch_states) > 0:
            # Convert batch data to tensors
            states_tensor = torch.tensor(batch_states, dtype=torch.float32)
            actions_tensor = torch.tensor(batch_actions, dtype=torch.long)
            rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32)
            next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32)
            dones_tensor = torch.tensor(batch_dones, dtype=torch.float32)
            old_log_probs_tensor = torch.tensor(batch_old_log_probs, dtype=torch.float32)
            
            # Compute values and advantages for the entire batch
            with torch.no_grad():
                values = value_net(states_tensor).squeeze()
                next_values = value_net(next_states_tensor).squeeze()
            
            # Compute TD errors
            td_errors = rewards_tensor + (1 - dones_tensor) * discount_factor * next_values - values
            
            # Compute advantages using GAE (recompute for consistency)
            advantages = torch.zeros_like(td_errors)
            advantage = 0
            episode_start = 0
            
            for i in range(len(batch_dones)):
                if i == 0 or batch_dones[i-1]:  # Start of new episode
                    episode_start = i
                    advantage = 0
            
            # Process advantages in reverse for each episode in the batch
            i = len(batch_dones) - 1
            while i >= 0:
                if batch_dones[i]:
                    advantage = td_errors[i]
                else:
                    advantage = td_errors[i] + discount_factor * lambda_gae * advantage
                advantages[i] = advantage
                i -= 1
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute returns for value loss
            returns = torch.zeros_like(rewards_tensor)
            return_val = 0
            for i in reversed(range(len(rewards_tensor))):
                if batch_dones[i]:
                    return_val = rewards_tensor[i]
                else:
                    return_val = rewards_tensor[i] + discount_factor * return_val
                returns[i] = return_val
            
            # PPO Update for multiple epochs
            for ppo_epoch in range(ppo_epochs):
                # Compute current action probabilities and values
                current_action_probs = policy_net(states_tensor)
                current_values = value_net(states_tensor).squeeze()
                
                # Compute new log probabilities
                new_dist = torch.distributions.Categorical(current_action_probs)
                new_log_probs = new_dist.log_prob(actions_tensor)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                
                # Compute policy loss (PPO clipped loss)
                term1 = ratio * advantages
                term2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                policy_loss = -torch.min(term1, term2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(current_values, returns)
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                
                # Backward pass
                policy_optimizer.zero_grad()
                value_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=0.5)
                
                policy_optimizer.step()
                value_optimizer.step()
            
            # Print losses for the current batch
            print(f"Batch ending at episode {episode-1}: Policy Loss = {policy_loss.item():.6f}, Value Loss = {value_loss.item():.2f}")
            
            # Update loss dataframes (use the episode number for x-axis)
            policy_loss_df.loc[len(policy_loss_df)] = [episode-1, policy_loss.item()]
            value_loss_df.loc[len(value_loss_df)] = [episode-1, value_loss.item()]
            
            # Clear the batch for next iteration
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_next_states = []
            batch_dones = []
            batch_old_log_probs = []

finally:
    env.close()
