# Deep Reinforcement Learning 

L’objectif de ce projet était de nous familiariser avec les techniques d’apprentissages profond par renforcement, en particulier le Deep-Q-Learning. Cette technique récente permet à un agent d’apprendre des politiques efficaces même avec des données d’entrées à nombreuses dimensions.
Nous nous sommes grandement inspirés du papier de recherche pointé à l'adresse https://arxiv.org/pdf/1605.02097.pdf, [Kempka et al., 2013], pour l’implémentation de l’algorithme, ces différentes optimisations et les paramètres des modèles. En effet un tel algorithme requiert d’être entraîné longtemps sur un grand nombre de données et, dans certains cas, il n’est pas garanti de converger. Pour adresser ce problème, le projet nous guide vers différentes méthodes inspirées du papier de recherche, à savoir l’Expérience epsilon-greedy et l’Expérience Replay qui améliorent grandement les performances du DQL.
Ce projet est constitutié de deux parties:

- Cartpole : Le but était de maîtriser l'environnement cartpole de la bibliothèque du gymnase pour s'assurer que nous avions une implémentation fonctionnelle et une bonne compréhension de l'algorithme DQN.
- VizDoom : Le but de la deuxième partie était de maîtriser l'environnement de jeu vidéo en utilisant des méthodes plus avancées telles que Dueling, Actor-Critic, etc.

Lien Démo : https://drive.google.com/file/d/197Pcv4Pheh4OCIzIHcMJJTrDdPBpCwo1/view?usp=sharing
