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

# ────────────────────────────────
# 2.  DataFrames for streaming
# ────────────────────────────────
pos_df  = pd.DataFrame(columns=["step", "pos"])
vel_df  = pd.DataFrame(columns=["step", "vel"])
ang_df  = pd.DataFrame(columns=["step", "angle"])
angv_df = pd.DataFrame(columns=["step", "ang_vel"])

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

#lists to record trajectory for current episode (reset after each episode)
states = []
actions = []
rewards = []
next_states = []
dones = []

obs, _ = env.reset()

try:
    for episode in range(num_episodes):
        #reset environment and trajectory recordings between episodes
        obs, _= env.reset()
        pos_df  = pos_df.iloc[0:0]
        vel_df  = vel_df.iloc[0:0]
        ang_df  = ang_df.iloc[0:0]
        angv_df = angv_df.iloc[0:0]
        
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
            #run forward pass to predict future value ("numerical score of how good is your current situation")
            predicted_value = value_net(state_tensor)
            #sample action from action probabilities
            action = torch.multinomial(action_probs, num_samples=1).item()
            #step the environment with the chosen action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            #record trajectory for current episode
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            # Append latest observation (update dataframes that record state for visualization)
            pos_df.loc[len(pos_df)]  = [step, state[0]]
            vel_df.loc[len(vel_df)]  = [step, state[1]]
            ang_df.loc[len(ang_df)]  = [step, state[2]]
            angv_df.loc[len(angv_df)] = [step, state[3]]

            # Update each chart (150 px tall → never spills)
            pos_ph.altair_chart( make_chart(pos_df,  "pos",       "Cart Position"), use_container_width=True )
            vel_ph.altair_chart( make_chart(vel_df,  "vel",       "Cart Velocity"), use_container_width=True )
            ang_ph.altair_chart( make_chart(ang_df,  "angle",     "Pole Angle"),    use_container_width=True )
            angv_ph.altair_chart( make_chart(angv_df, "ang_vel",  "Pole Angular Velocity"), use_container_width=True )

            # Update frame in Quadrant 1
            frame_ph.image(
                Image.fromarray(env.render()),
                caption=f"Step {step} — {'Left' if action==0 else 'Right'}",
                use_container_width=True,
            )

            time.sleep(0.05)
            obs = next_state
            step += 1


finally:
    env.close()
