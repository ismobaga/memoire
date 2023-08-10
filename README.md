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

