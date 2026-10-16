import torch
import torch.nn as nn
import pygame
import os
import sys
import math
from environnement import EnvironnementRobot
from rrt import RRT

from pathlib import Path

# chemins relatifs a la racine du depot
RACINE = Path(__file__).resolve().parent.parent

class DQN(nn.Module):
    def __init__(self, taille_etat, nb_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(taille_etat, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, nb_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

depart_robot = (40, 40)
arrivee_finale = (750, 550)

liste_obstacles = [
    pygame.Rect(230, 0,   70, 370),
    pygame.Rect(430, 200, 70, 400),
    pygame.Rect(600, 50,  70, 350),
]

# obstacles gonflés pour que le RRT laisse de la place au robot rayon 25 + marge
carte_pour_rrt = [mur.inflate(60, 60) for mur in liste_obstacles]

chemin_trouve = None
arbre_rrt = []
tentatives = 0

while chemin_trouve is None or len(chemin_trouve) == 0:
    tentatives += 1
    solveur_rrt = RRT(depart_robot, arrivee_finale, 3000, carte_pour_rrt, 30, None)
    chemin_trouve, arbre_rrt = solveur_rrt.compute_path()
    if tentatives > 10:
        print("RRT ne trouve pas de chemin sur cette carte")
        sys.exit()

chemin_trouve.pop(0)
if len(chemin_trouve) == 0:
    chemin_trouve = [arrivee_finale]

env = EnvironnementRobot(depart=depart_robot, waypoints=chemin_trouve)
env.liste_obstacles = liste_obstacles

taille_etat = 13
nb_actions = 5
reseau = DQN(taille_etat, nb_actions)

nom_fichier_modele = RACINE / "models" / "dqn_meilleur_modele.pth"

if os.path.exists(nom_fichier_modele):
    reseau.load_state_dict(torch.load(nom_fichier_modele, weights_only=True))
    print("charger")
else:
    print(f"modele introuvable : {nom_fichier_modele}")
    sys.exit()

# on met le reseau en mode evaluation pour qu'il ne calcule pas les gradients
reseau.eval()

def couleur_vitesse(vitesse, vitesse_max):
    # on force un vrai dégradé mm vitesse max
    ratio = min(vitesse / vitesse_max, 1.0) if vitesse_max > 0 else 0
    # du bleu a cyan a vert a jaune a rouge
    if ratio < 0.25:
        r, g, b = 0, int(ratio * 4 * 255), 255
    elif ratio < 0.5:
        r, g, b = 0, 255, int((0.5 - ratio) * 4 * 255)
    elif ratio < 0.75:
        r, g, b = int((ratio - 0.5) * 4 * 255), 255, 0
    else:
        r, g, b = 255, int((1.0 - ratio) * 4 * 255), 0
    return (r, g, b)

while True:
    etat = env.reset()
    done = False
    historique = []

    while not done:
        env.clock.tick(30)
        # pas de hasard
        with torch.no_grad():
            etat_tensor = torch.FloatTensor(etat).unsqueeze(0)
            q_values = reseau(etat_tensor)
            action = torch.argmax(q_values).item()

        nouvel_etat, reward, done = env.step(action)

        # on enregistre la position et la vitesse pour le tracé
        historique.append(((env.x, env.y), env.vitesse))

        # dessin manuel pour avoir le dégradé + arbre RRT
        env.screen.fill((230, 230, 230))

        for obs in env.liste_obstacles:
            pygame.draw.rect(env.screen, (255, 0, 0), obs)

        # arbre RRT en gris clair
        for branche in arbre_rrt:
            pygame.draw.line(env.screen, (180, 180, 180), branche[0], branche[1], 1)

        # chemin RRT planifié en vert foncé
        chemin_affiche = [depart_robot] + chemin_trouve
        if len(chemin_affiche) > 1:
            pygame.draw.lines(env.screen, (0, 140, 0), False, chemin_affiche, 2)
        for wp in env.waypoints:
            pygame.draw.circle(env.screen, (0, 200, 0), (int(wp[0]), int(wp[1])), 5)

        pygame.draw.circle(env.screen, (0, 255, 0), (int(env.arrivee[0]), int(env.arrivee[1])), 15)

        # tracer dégradé vitesse de la trajectoire réelle
        if len(historique) > 1:
            for k in range(len(historique) - 1):
                (x1, y1), v1 = historique[k]
                (x2, y2), _  = historique[k + 1]
                pygame.draw.line(env.screen, couleur_vitesse(v1, env.vitesse_max),
                                 (int(x1), int(y1)), (int(x2), int(y2)), 3)

        pygame.draw.circle(env.screen, (50, 50, 50), (int(env.x), int(env.y)), env.robot_rayon)
        end_x = env.x + math.cos(math.radians(env.angle)) * 35
        end_y = env.y + math.sin(math.radians(env.angle)) * 35
        pygame.draw.line(env.screen, (255, 255, 255), (int(env.x), int(env.y)), (int(end_x), int(end_y)), 3)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        etat = nouvel_etat