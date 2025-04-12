- Objectif du projet
Ce projet vise à développer un agent intelligent capable de gérer dynamiquement un portefeuille financier en adaptant sa stratégie en fonction des conditions de marché.
L’approche repose sur un Meta-Agent capable de choisir dynamiquement entre DQN (Deep Q-Learning) et PPO (Proximal Policy Optimization) selon la volatilité, la tendance du marché et d'autres indicateurs.
_____________________________________________________________________________________________________________________________________

- L’objectif principal est de :

- Tirer parti de la réactivité de DQN et de la stabilité de PPO.

- Permettre au Meta-Agent de s’adapter intelligemment en temps réel aux différentes phases du marché.

- Améliorer la robustesse et réduire les risques liés à l'entraînement traditionnel d'un seul modèle.

- Contenu du projet
meta_agent.py : Définition complète du Meta-Agent (sélection dynamique d’actions et apprentissage).

- train_meta_agent.py : Script d'entraînement du Meta-Agent hybride sur des données financières simulées.

plot_hybrid_performance.py : Visualisation graphique de la performance comparative (DQN vs PPO vs Meta-Agent).

- Librairies utilisées
Voici la liste complète des dépendances nécessaires pour exécuter ce projet :

- Librairie	Version conseillée	Description
numpy	>= 1.21	Manipulation de matrices et calculs numériques
pandas	>= 1.3	Gestion des séries temporelles de données financières
matplotlib	>= 3.4	Visualisation des courbes de performance
stable-baselines3	>= 1.5	Implémentation de DQN, PPO et outils RL
gym	>= 0.26	Création d'environnements personnalisés pour l'agent
shimmy	>= 0.2.0	Compatibilité entre Gym et Gymnasium (patch automatique)

- Installation de l’environnement
Je recommande d'utiliser un environnement virtuel pour éviter les conflits de versions.

- Créer un environnement virtuel :

python -m venv rl_env
Activer l'environnement virtuel :

- Windows :

.\rl_env\Scripts\activate
Mac/Linux :

source rl_env/bin/activate
Installer les dépendances :

pip install numpy pandas matplotlib stable-baselines3 gym shimmy

___________________________________________________________________________________________________________________________

- Organisation des fichiers
Fichier	Rôle
src/meta_agent.py	Contient la définition du Meta-Agent hybride
src/train_meta_agent.py	Script d'entraînement et interaction avec l'environnement
src/plot_hybrid_performance.py	Génère les graphiques de performance
models/	Dossier pour sauvegarder les modèles entraînés
data/	Données financières utilisées pour l’entraînement (ex : BTC_Daily.csv)

- Points techniques clés du Meta-Agent
Sélection contextuelle : RSI, MACD, volatilité, tendance.

Adaptation dynamique : Choix entre DQN et PPO selon le contexte du marché.

Entraînement intelligent : Apprentissage progressif basé sur les conditions réelles simulées.

- Pourquoi ce projet est pertinent ?
Les marchés financiers sont très dynamiques. Un agent statique (DQN seul ou PPO seul) risque :
De mal réagir aux changements rapides de marché.D'être soit trop lent (PPO) soit trop instable (DQN).
Le Meta-Agent hybride tire parti des forces combinées des deux algorithmes :
- Rapidité de décision grâce à DQN.
- Robustesse et stabilité à long terme grâce à PPO.

Cette approche est très inspirée de la finance comportementale où les traders adaptent instinctivement leurs stratégies selon les conditions économiques.

- Après avoir installé toutes les librairies, pour entraîner le Meta-Agent :

cd src
python train_meta_agent.py



