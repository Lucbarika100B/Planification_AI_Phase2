import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/dqn_action_distribution.csv")

plt.figure(figsize=(10, 6))
plt.bar(df["Action"], df["Frequency"], color="skyblue")
plt.title("Distribution des actions prises par DQN")
plt.xlabel("Actions")
plt.ylabel("Fréquence")
plt.grid(True)
plt.savefig("results/dqn_action_plot.png")
plt.show()
