import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Données simulées (à remplacer plus tard par des vraies logs DQN) ===
data = {
    'Action': ['BUY', 'SELL', 'HOLD', 'SCALP_BUY', 'SCALP_SELL', 'SWAP', 'TAKE_PROFIT', 'STOP_LOSS'],
    'Frequency': [320, 410, 85, 300, 290, 160, 120, 70],
    'Success_Rate': [0.73, 0.78, 0.60, 0.71, 0.74, 0.58, 0.85, 0.91]
}

df = pd.DataFrame(data)

# === Création de la figure ===
fig, ax1 = plt.subplots(figsize=(12, 6))

# === Barplot pour les fréquences ===
bar_plot = sns.barplot(
    x='Action', y='Frequency', data=df, ax=ax1, color='skyblue'
)
ax1.set_ylabel('Fréquence (exécutions)', fontsize=12)
ax1.set_xlabel('Action DQN', fontsize=12)
ax1.set_title('Fréquence & Taux de Succès des Actions DQN', fontsize=15)

# === Deuxième axe Y pour le taux de succès ===
ax2 = ax1.twinx()
line_plot = sns.lineplot(
    x='Action', y='Success_Rate', data=df, ax=ax2, color='red', marker='o'
)
ax2.set_ylabel('Taux de Succès (%)', fontsize=12)
ax2.set_ylim(0, 1.05)
ax2.axhline(0.7, color='gray', linestyle='--', linewidth=1)

# === Ajout des légendes combinées ===
bar_label = ax1.bar([0], [0], color='skyblue', label='Fréquence')
line_label = ax2.plot([], [], color='red', marker='o', label='Taux de Succès')[0]
threshold_line = ax2.plot([], [], color='gray', linestyle='--', label='Seuil 70%')[0]

ax2.legend(handles=[bar_label[0], line_label, threshold_line], loc='upper left')

# Format de l'axe X
plt.xticks(rotation=15)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.savefig("results/dqn_action_plot.png")
plt.show()
