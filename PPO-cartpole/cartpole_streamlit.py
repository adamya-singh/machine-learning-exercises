import streamlit as st
import gymnasium as gym
import numpy as np
from PIL import Image
import time
import os
import pandas as pd

# Set environment variable to prevent SDL from trying to create a window
os.environ['SDL_VIDEODRIVER'] = 'dummy'

#set streamlit to wide mode
st.set_page_config(layout="wide")

# Set up the Streamlit app
st.title("CartPole Environment Visualization")
st.write("Watch the CartPole environment with random actions. The cart moves left or right to balance the pole.")

# Create placeholders for visualization
image_placeholder = st.empty()

# Create placeholders for charts
cart_position_chart = st.empty()
cart_velocity_chart = st.empty()
pole_angle_chart = st.empty()
pole_angular_velocity_chart = st.empty()

# Initialize data storage
cart_position_data = []
cart_velocity_data = []
pole_angle_data = []
pole_angular_velocity_data = []

# Initialize the CartPole environment with a different render mode
env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset()

try:
    # Run the simulation
    for step in range(1000):
        # Select a random action (0: left, 1: right)
        action = env.action_space.sample()
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Store the observation data
        cart_position_data.append(observation[0])
        cart_velocity_data.append(observation[1])
        pole_angle_data.append(observation[2])
        pole_angular_velocity_data.append(observation[3])
        
        # Create DataFrames for each metric
        cart_position_df = pd.DataFrame({'Cart Position': cart_position_data})
        cart_velocity_df = pd.DataFrame({'Cart Velocity': cart_velocity_data})
        pole_angle_df = pd.DataFrame({'Pole Angle': pole_angle_data})
        pole_angular_velocity_df = pd.DataFrame({'Pole Angular Velocity': pole_angular_velocity_data})
        
        # Update charts
        cart_position_chart.line_chart(cart_position_df)
        cart_velocity_chart.line_chart(cart_velocity_df)
        pole_angle_chart.line_chart(pole_angle_df)
        pole_angular_velocity_chart.line_chart(pole_angular_velocity_df)
        
        # Render the frame as an RGB array
        frame = env.render()
        # Convert the frame (numpy array) to a PIL Image
        image = Image.fromarray(frame)
        # Display the image in Streamlit
        image_placeholder.image(image, caption=f"Step {step}, Action: {'Left' if action == 0 else 'Right'}", use_container_width=True)
        
        # Add a small delay to make the animation visible
        time.sleep(0.05)
        
        # Reset the environment if the episode ends
        if terminated or truncated:
            observation, info = env.reset()
            # Clear the data when episode resets
            cart_position_data = []
            cart_velocity_data = []
            pole_angle_data = []
            pole_angular_velocity_data = []
finally:
    # Ensure proper cleanup
    env.close()
