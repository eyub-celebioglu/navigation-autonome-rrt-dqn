import numpy as np
import time
import pygame
import csv
from rrt import RRT
from rrt_sc import RRTSC
from rrt_star import RRTStar

from pathlib import Path

# chemins relatifs a la racine du depot
RACINE = Path(__file__).resolve().parent.parent
CHEMIN_CSV = RACINE / "results" / "data" / "benchmark_rrt.csv"

pygame.init()
liste_obstacles = []
liste_obstacles.append(pygame.Rect(200, 150, 130, 330))
liste_obstacles.append(pygame.Rect(500, 150, 130, 330))
liste_obstacles.append(pygame.Rect(350, 150, 130, 130))

depart = (50, 50)
arrivee = (700, 500)
long_branche = 20
theta = 30
nb_repetitions = 10

iterations_testees = [500,600, 700, 1000,2000,3000,4000,5000,6000]

def cout_chemin(chemin):
    if chemin is None or len(chemin) < 2:
        return 0
    total = 0
    for k in range(len(chemin) - 1):
        dx = chemin[k+1][0] - chemin[k][0]
        dy = chemin[k+1][1] - chemin[k][1]
        total += np.sqrt(dx**2 + dy**2)
    return total

with open(CHEMIN_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow([
        "Itérations",
        "Temps RRT", "Cout RRT", "Noeuds RRT",
        "Temps RRTSC", "Cout RRTSC", "Noeuds RRTSC", 
        "Temps RRT*", "Cout RRT*", "Noeuds RRT*", 
    ])

    for nb_iter in iterations_testees:
        print(f"Test avec {nb_iter} itérations...")
        ligne = [nb_iter]

        for solveur_class in [RRT, RRTSC, RRTStar]:
            runs_temps  = []
            runs_cout   = []
            runs_noeuds = []


            for _ in range(nb_repetitions):
                solveur = solveur_class(depart, arrivee, nb_iter, liste_obstacles, long_branche, theta)
                t = time.perf_counter()
                chemin, _ = solveur.compute_path()
                runs_temps.append(time.perf_counter() - t)

                if chemin is not None:
                    runs_cout.append(cout_chemin(chemin))
                    runs_noeuds.append(len(chemin))
                else:
                    runs_cout.append(0)
                    runs_noeuds.append(0)

            ligne += [
                f"{np.mean(runs_temps):.4f}",

                f"{np.mean(runs_cout):.1f}",

                f"{np.mean(runs_noeuds):.1f}",
                
            ]

        writer.writerow(ligne)

print(f"Fichier {CHEMIN_CSV} sauvegardé")