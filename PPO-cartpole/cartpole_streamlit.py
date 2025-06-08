import streamlit as st
import gymnasium as gym
import numpy as np
from PIL import Image
import time
import os

# Set environment variable to prevent SDL from trying to create a window
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Set up the Streamlit app
st.title("CartPole Environment Visualization")
st.write("Watch the CartPole environment with random actions. The cart moves left or right to balance the pole.")

# Create a placeholder for the image
image_placeholder = st.empty()

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
finally:
    # Ensure proper cleanup
    env.close()
