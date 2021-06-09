## Projet The Ms PacMan Vs Ghosts Competition

Ce projet, intitulé [The Ms. PacMan VS Ghost Team Competition](http://www.pacmanvghosts.co.uk/), entre dans le cadre de l’UE Environnements Virtuels Interactifs et Jeux Vidéos du Master 2 Informatique parcours ANDROIDE de la faculté d’ingénierie de Sorbonne Université. Il consiste à implémenter une Intelligence Artificielle dans un jeu vidéo existant et ce, en mettant en oeuvre les notions d’apprentissage, d’optimisation et de décision acquises au cours de la formation.

Le projet a été réalisé par le trinôme composé des étudiants Clément Boisson, Jehyankaa Jeyarajaratnam et Yasmine Hamdani.

## Code

Le code du projet se trouve dans src/main/java et dans src/main/java/entrants/pacman.CBYHJJ

* Dans src/main/java/entrants/ghosts.CBYHJJ, le comportement des fantômes du jeu est défini (on n'y a pas touché)

* Dans src/main/java/entrants/pacman.CBYHJJ nous trouvons plusieurs classes que nous avons créées pour implémenter notre solution.

  * La classe __PacManPlayer__ est une classe qui charge le réseau de neurones "network.txt" et permet
            de choisir la direction à prendre selon les inputs décrits dans le rapport.

  * La classe __PacManTraining__ est une classe qui sert à entraîner le réseau de nerones.
            Suivant les inputs, on détermine la direction à prendre via un softmax, et on apprend au réseau à
            faire le lien entre la valeur du déplacement et les inputs

  * La classe __QLearning__ contient l'algorithme du QLearning et ce qui s'y rapporte comme le calcul des récompenses

  * La classe __Tools__ contient des méthodes utiles lors du calcul des paramètres (comme le nombre d'intersections proches de Ms PacMan)

  * La classe __DistributedRandomNumberGenerator__ contient une méthode pour tirer aléatoirement selon une distribution de probabilités donnée (utilisée pour le softmax). Cette classe provient de [ce repo Github](https://github.com/mikeroelens/HungerMoji).

* Dans src/main/java on trouve deux classes selon le mode de lancement du jeu.

  * La classe __Training__ sert à lancer plusieurs parties avec PacManTraining afin d'entraîner le réseau
            sans afficher le jeu.

  * La classe __Play__ sert à lancer une partie avec PacManPlayer en affichant le jeu.
  

