from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Spécifiez le chemin du fichier texte contenant vos données
file_path = 'data.txt'

# Initialisez une liste vide pour stocker les lignes lues
data = []

data = defaultdict(list)
clean = defaultdict(list)

with open(file_path, 'r') as file:
    # Parcourez chaque ligne du fichier
    for line in file:
        # Divisez la ligne en fonction de l'espace
        parts = line.strip().split()

        # La colonne "methode" peut avoir plusieurs parties séparées par des espaces,
        # alors nous la concaténons pour obtenir la valeur complète de la méthode.
        method_value = ' '.join(parts[2:-3])  # Indices de 2 jusqu'à l'avant-dernier élément

        # Réassemblez la ligne de données en respectant la structure attendue
        reconstructed_line = parts[:2] + [method_value] + parts[-3:]
        # print(data)

        # Ajoutez la ligne de données à la liste
        data[method_value].append(parts[:2] + parts[-3:])

lu = 0
users = []
for key in data:
    data[key] = [list(map(int, line))for line in data[key] ]
    d = []
    somme = 0
    i = 0
    for line in data[key]:
        if i != line[0]:
            i = line[0]
            d.append(somme/10)
            somme = 0
            print(i,)
        percent = 100 * line[4]/line[3]
        somme += percent

    d.append(somme/10)
    clean[key] = d
print(clean)
key =     'Soft deadline penalty'
for line in data[key]:
    if lu != line[2]:
        lu = line[2]
        users.append(lu)
# print(users)
# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
for key in clean:
    ax.plot(users, clean[key], label=key)

ax.set(xlabel='Time [$s$]', ylabel='Voltage [mV]',
       title='About as simple as it gets, folks')
ax.legend()
plt.show()


import tikzplotlib
tikzplotlib.save("figure.pgf")