import streamlit as st
import gymnasium as gym
from PIL import Image
import os, time, pandas as pd

# ────────────────────────────────
# 0.  Config
# ────────────────────────────────
os.environ["SDL_VIDEODRIVER"] = "dummy"
st.set_page_config(layout="wide")
st.title("CartPole Environment Visualization")
st.write("Watch CartPole balance itself with random left/right actions.")

# ────────────────────────────────
# 1.  Layout — keep frame in Quadrant 1
# ────────────────────────────────
with st.container():
    img_col, charts_col = st.columns(2)   # 50 % | 50 %

    # Q1: live RGB frame
    with img_col:
        frame_ph = st.empty()

    # Q2-Q4: four small charts (2 × 2)
    with charts_col:
        # row 1
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.subheader("Cart Position")
            pos_chart = st.line_chart(pd.DataFrame([], columns=["pos"]))
        with r1c2:
            st.subheader("Cart Velocity")
            vel_chart = st.line_chart(pd.DataFrame([], columns=["vel"]))

        # row 2
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.subheader("Pole Angle")
            ang_chart = st.line_chart(pd.DataFrame([], columns=["angle"]))
        with r2c2:
            st.subheader("Pole Angular Velocity")
            angv_chart = st.line_chart(pd.DataFrame([], columns=["ang_vel"]))

# ────────────────────────────────
# 2.  Gym loop with streaming updates
# ────────────────────────────────
env = gym.make("CartPole-v1", render_mode="rgb_array")
obs, _ = env.reset()

try:
    for step in range(1_000):
        action = env.action_space.sample()
        obs, _, done, trunc, _ = env.step(action)

        # Push the latest point to each chart
        pos_chart.add_rows({"pos": [obs[0]]})
        vel_chart.add_rows({"vel": [obs[1]]})
        ang_chart.add_rows({"angle": [obs[2]]})
        angv_chart.add_rows({"ang_vel": [obs[3]]})

        # Update frame in Quadrant 1
        frame_ph.image(
            Image.fromarray(env.render()),
            caption=f"Step {step} — {'Left' if action == 0 else 'Right'}",
            use_container_width=True,
        )

        time.sleep(0.05)

        if done or trunc:
            obs, _ = env.reset()
finally:
    env.close()
