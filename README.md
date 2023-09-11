# memoire


## TODO
### Tâches à accomplir :
- [x] À chaque intervalle de temps, générer un nombre de requêtes.
- [x] Générer des utilisateurs, et assigne le aux requêtes .

- [x] Générer des positions pour les utilisateurs.
- [x] Appliquer la formule de Shannon pour calculer la capacité de données.
  - Formule de Shannon : Snr = P / (distance ^ 2) / bruit

- [x] Utiliser des échéances flexibles (soft deadlines).
  - skip pen
  - Une tache qui passe 1
  - depase le dealine une fraction de 1
- [x] Plotter la valeur de fitness pour chaque génération.
- [x] Écrire des cas pour le problème.
  - Supposons que plusieurs requêtes ont le même temps de réponse. Pour que la solution soit valide :
    - Si nous choisissions d'exécuter une tâche sur le serveur :
    Toutes les requêtes doivent demander la même tâche avec la même entrée. Le serveur exécute cette tâche une seule fois pour répondre à toutes les requêtes simultanées. 
    - Si nous choisissions de répondre avec l'entrée :
    Toutes les requêtes doivent avoir la même entrée. Le serveur répond en utilisant cette entrée pour toutes les requêtes simultanées.
    ```python
    individu = [
    (2, 5),    # Requête 1 : User(Local), Temps 5
    (SERVER_LOCATION, 8),    # Requête 2 : Serveur, Temps 8
    None    # Requête 3 : Non planifiée
    ]
    ```
- [ ] Transformer le `None` en `(-1, -1)`
- [ ] Implémenter une fonction de correction.
  - Une correction posdible c'Est avoir la fitness de tous les combinaison de groupe
    On decide de ne pas executer des requetes et voir quel combinaison donne la meilleur valeur
- [ ] Transformer min en max
- [x] Une fonction pour donner le nombre de requêtes execute
- [ ] utilise github(git)
- les données d'articles
#### Jeudi

- courbe de convergence
- Moins d'users (40) et entre (0,10) requetes per time slot
- Change le deadline (3, et 8) joue sur max (courbes)
Apres
- Nombre de requete / 2 ou sur 4


- Évolution de la courbe de convergence au fil des generations.
- Réduction du nombre d'utilisateurs à 40, avec une fréquence de requêtes par intervalle de temps allant de 0 à 10.
- Variation des délais impartis, avec des valeurs entre 3 et 8, impactant les valeurs maximales des courbes.
Ensuite,
- Réduction du nombre de requêtes de moitié ou par un facteur de 4.
- 

### Parameters
| Parameter                   | Explanation                                                                                                   |
|---------------------------- |-------------------------------------------------------------------------------------------------------------- |
| N_USERS                     | Number of users in the system.                                                                               |
| N_TASKS                     | Number of different tasks available in the system.                                                           |
| N_INPUT                     | Number of different input data sizes available.                                                             |
| N_REQUESTS                  | Number of requests generated for scheduling.                                                                 |
| MIN_DATA_RATE               | Minimum data transfer rate for transmission.                                                                 |
| MAX_DATA_RATE               | Maximum data transfer rate for transmission.                                                                 |
| MIN_INPUT_SIZE              | Minimum size of input data for tasks.                                                                        |
| MAX_INPUT_SIZE              | Maximum size of input data for tasks.                                                                        |
| MIN_OUTPUT_TIMES            | Minimum time taken to process output data.                                                                   |
| MAX_OUTPUT_TIMES            | Maximum time taken to process output data.                                                                   |
| USER_COMPUTATION_CAPACITY   | Computation capacity of each user's local device.                                                           |
| MIN_TASK_PROCESSING_REQ     | Minimum processing requirement for a task.                                                                   |
| MAX_TASK_PROCESSING_REQ     | Maximum processing requirement for a task.                                                                   |
| MIN_ARRIVAL                 | Minimum arrival time of requests.                                                                            |
| MAX_ARRIVAL                 | Maximum arrival time of requests.                                                                            |
| MIN_DEADLINE                | Minimum deadline for request completion.                                                                     |
| MAX_DEADLINE                | Maximum deadline for request completion.                                                                     |
| SERVER_COMPUTATION_CAPACITY | Computation capacity of the server (MEC).                                                                    |
| POPULATION_SIZE             | Number of individuals in the population for the genetic algorithm.                                          |
| GENERATIONS                | Number of generations for the genetic algorithm.                                                            |
| MUTATION_RATE               | Probability of mutation during crossover in the genetic algorithm.                                          |
| PROBABILITY_SKIP            | Probability of not responding to a request in an individual's schedule during mutation in the genetic algorithm. |



#### Pas bon
Si plusieurs requêtes ont le même temps de réponse  nous avons une solution pas valide :
- Si nous choisissons d'exécuter une tâche sur le serveur :
Toutes les requêtes demandent des tâches différentes ou des entrées différentes.  les requêtes nécessitent différentes tâches ou entrées, ce qui ne peut pas être satisfait.
- Si nous choisissons de répondre avec l'entrée :
Les requêtes demandent différentes entrées. Le serveur ne peut répondre qu'avec une seule entrée, ce qui ne peut pas satisfaire les demandes de tâches et d'entrées différentes.

Considérons les éléments suivants :
C1 : code 1, I1 : pour l'entrée 1, L1 : pour la location 1
Même chose pour C2, I2, L2.

Lorsque deux réponses (C1, I1, L1) et (C2, I2, L2) pour deux requêtes sont envoyées au même time slot, nous avons les possibilités suivantes :

| Possibilité           | Quoi faire                                                 |
|-----------------------|------------------------------------------------------------|
| C1=C2, I1=I2, L1!=L2  | Pas bon, changer L1 ou L2 de manière à avoir L1=L2        |
| C1=C2, I1!=I2, L1!=L2 | Pas bon, différentes entrées, enlever l'une des réponses   |
| C1!=C2, I1!=I2, L1!=L2| Pas bon, différentes entrées et codes, enlever l'une des réponses |
| C1=C2, I1!=I2, L1=L2  | Pas bon, entrée différente, enlever l'une des réponses   |
| C1!=C2, I1!=I2, L1=L2 | Pas bon, différentes entrées, enlever l'une des réponses |
| C1!=C2, I1=I2, L1=L2  | Bon si L1=L2=LOCAL, sinon ajuster L1, L2 à LOCAL          |
| C1!=C2, I1=I2, L1!=L2 | Pas bon, ajuster L1, L2 à LOCAL                           |
| C1!=C2, I1!=I2, L1=L2 | Pas bon, enlever l'une des réponses                       |
| C1=C2, I1=I2, L1=L2   | Pas de problème, pas besoin d'ajuster  



Nouvau indivudu
choir seulement quand repondre



### Dynamic
A voir
Tableau[tache][users][input][location]






##### TODO
LES DEUX FITNESS SUR LA MEME COURBES ET MEME RESEAXU

Couper a 200 generations

### Heuristique 
- le  nombre diminu par iterations
- probleme je n'enleve pas des requete quand c'est ajouté a un group (j'enleve seulement s'il init the group)