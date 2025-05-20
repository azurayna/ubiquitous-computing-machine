import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import random

st.set_page_config(page_title="Lotka-Volterra Simulator", layout="wide")

st.title("🐺🐇 Lotka-Volterra Simulator with Gillespie Algorithm")

st.sidebar.header("Simulation Parameters")
a = st.sidebar.slider("Prey birth rate (a)", 0.01, 2.0, 1.0, 0.01)
b = st.sidebar.slider("Predation rate (b)", 0.01, 2.0, 0.1, 0.01)
c = st.sidebar.slider("Predator death rate (c)", 0.01, 2.0, 1.5, 0.01)
d = st.sidebar.slider("Predator efficiency (d)", 0.01, 2.0, 0.075, 0.01)

prey0 = st.sidebar.slider("Initial prey population", 1, 500, 100)
predator0 = st.sidebar.slider("Initial predator population", 1, 500, 50)

time_end = st.sidebar.slider("Simulation time", 10, 500, 100)
simulation_type = st.sidebar.selectbox("Simulation Type", ["Deterministic (ODE)", "Stochastic (Gillespie)"])

# Deterministic Lotka-Volterra model
def lotka_volterra(t, z):
    x, y = z
    dxdt = a * x - b * x * y
    dydt = d * b * x * y - c * y
    return [dxdt, dydt]

# Gillespie stochastic simulation
def gillespie(prey0, predator0, a, b, c, d, max_time):
    time = 0.0
    prey = prey0
    predator = predator0

    t_series = [time]
    prey_series = [prey]
    predator_series = [predator]

    while time < max_time and prey > 0 and predator > 0:
        rate_birth = a * prey
        rate_predation = b * prey * predator
        rate_death = c * predator

        total_rate = rate_birth + rate_predation + rate_death
        if total_rate == 0:
            break

        time += np.random.exponential(1 / total_rate)
        event = random.choices(
            population=["birth", "predation", "death"],
            weights=[rate_birth, rate_predation, rate_death],
            k=1
        )[0]

        if event == "birth":
            prey += 1
        elif event == "predation":
            prey -= 1
            predator += 1
        elif event == "death":
            predator -= 1

        t_series.append(time)
        prey_series.append(prey)
        predator_series.append(predator)

    return np.array(t_series), np.array(prey_series), np.array(predator_series)

# Run simulation
if simulation_type == "Deterministic (ODE)":
    sol = solve_ivp(
        lotka_volterra,
        [0, time_end],
        [prey0, predator0],
        t_eval=np.linspace(0, time_end, 1000)
    )
    t = sol.t
    prey = sol.y[0]
    predator = sol.y[1]
else:
    t, prey, predator = gillespie(prey0, predator0, a, b, c, d, time_end)

# DataFrame
results_df = pd.DataFrame({
    "Time": t,
    "Prey": prey,
    "Predator": predator
})

st.subheader("☆ Time Series")
fig, ax = plt.subplots()
ax.plot(t, prey, label="Prey")
ax.plot(t, predator, label="Predator")
ax.set_xlabel("Time")
ax.set_ylabel("Population")
ax.legend()
st.pyplot(fig)

st.subheader("🐰🥕 Phase Plot (Prey vs Predator)")
fig2, ax2 = plt.subplots()
ax2.plot(prey, predator, lw=1)
ax2.set_xlabel("Prey Population")
ax2.set_ylabel("Predator Population")
ax2.set_title("Phase Plot")
st.pyplot(fig2)

st.subheader("☆ Population Statistics")
st.write("### Prey")
st.write(results_df["Prey"].describe())

st.write("### Predator")
st.write(results_df["Predator"].describe())

st.subheader("🐰🥕 Download Data")
st.download_button("Download CSV", data=results_df.to_csv(index=False), file_name="lotka_volterra_data.csv")

st.subheader("☆ Histograms")
fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
ax3.hist(prey, bins=30, color="skyblue", edgecolor="black")
ax3.set_title("Prey Histogram")
ax3.set_xlabel("Population")
ax3.set_ylabel("Frequency")

ax4.hist(predator, bins=30, color="salmon", edgecolor="black")
ax4.set_title("Predator Histogram")
ax4.set_xlabel("Population")
ax4.set_ylabel("Frequency")
st.pyplot(fig3)

st.subheader("🐰🥕 Raw Data Table")
st.dataframe(results_df)

st.markdown("---")
st.markdown("Deployed online thanks to Streamlit.")
