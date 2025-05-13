import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt

# Config
BALL_DIAMETER = 6.35
R = BALL_DIAMETER / 2
AVERAGE_FRAMERATE = 100

st.title("FOAB Data Visualizer")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Extract odor pulse indices
    start_idx, stop_idx = [], []
    for i in range(len(df) - 1):
        if df['odor_state'].iloc[i+1] - df['odor_state'].iloc[i] == 1:
            start_idx.append(i + 1)
        elif df['odor_state'].iloc[i+1] - df['odor_state'].iloc[i] == -1:
            stop_idx.append(i)

    pulse_df = pd.DataFrame({
        "start_idx": start_idx,
        "stop_idx": stop_idx,
        "Odor": [df["odor_name"].iloc[j].strip() for j in start_idx]
    })

    # Inputs
    size_window = st.number_input("Window size", min_value=1, value=500)
    filter_window = st.number_input("Filter window (odd)", min_value=3, step=2, value=25)
    center = st.slider("Center Frame", min_value=0, max_value=len(df)-1, value=1000)

    start = max(center - size_window, 0)
    stop = min(center + size_window, len(df)-1)

    # Setup
    x = 'frames'
    x_pos = df.columns[14]
    y_pos = df.columns[15]
    heading_dir = df.columns[16]
    movement_speed = df.columns[18]

    # Precompute filtered signals
    ground_speed = medfilt(df[movement_speed] * R * AVERAGE_FRAMERATE)
    ground_speed_filtered = savgol_filter(ground_speed, filter_window, 3)

    upwind_velocity = df[x_pos].diff()
    upwind_velocity_filtered = savgol_filter(medfilt(upwind_velocity * R), filter_window, 3)

    heading = df[heading_dir]
    angular_velocity = np.diff(heading) * 180 / np.pi
    angular_velocity = [(b - 360 if b > 180 else b + 360 if b < -180 else b) for b in angular_velocity]
    angular_velocity_filtered = savgol_filter(medfilt(angular_velocity), filter_window, 3)

    integrated_heading = df[df.columns[16]] * R

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    axs = axs.flatten()

    # Trajectory
    tail = max(center - 500, 0)
    axs[0].plot(df[x_pos][tail:center]*R, df[y_pos][tail:center]*R, color='gray', alpha=0.5)
    axs[0].scatter(df[x_pos][center]*R, df[y_pos][center]*R, color='red')
    axs[0].set_title("Trajectory")

    # Add colored odor segments
    def draw_odor_patches(ax):
        for _, row in pulse_df.iterrows():
            if row["stop_idx"] >= start and row["start_idx"] <= stop:
                color = {'MO': 'blue', 'IAA10% x 0.1': 'green', 'BEN10% x 0.1': 'red'}.get(row["Odor"].strip(), None)
                if color:
                    ax.axvspan(max(row["start_idx"], start), min(row["stop_idx"], stop), color=color, alpha=0.3)

    # Ground speed
    axs[1].plot(ground_speed_filtered)
    axs[1].axvline(center, color='red', linestyle=':')
    axs[1].set_title("Ground Speed")
    draw_odor_patches(axs[1])

    # Upwind velocity
    axs[2].plot(upwind_velocity_filtered)
    axs[2].axvline(center, color='red', linestyle=':')
    axs[2].set_title("Upwind Velocity")
    draw_odor_patches(axs[2])

    # Angular velocity
    axs[3].plot(angular_velocity_filtered)
    axs[3].axvline(center, color='red', linestyle=':')
    axs[3].set_title("Angular Velocity")
    draw_odor_patches(axs[3])

    # Integrated heading
    axs[4].plot(integrated_heading)
    axs[4].axvline(center, color='red', linestyle=':')
    axs[4].set_title("Integrated Heading")
    draw_odor_patches(axs[4])

    for ax in axs:
        ax.set_xlim(start, stop)
        ax.axhline(0, linestyle='--', color='black')

    st.pyplot(fig)
