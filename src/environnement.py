import pygame
import math
import numpy as np

class EnvironnementRobot:
    def __init__(self, depart=(50, 50), waypoints=None):
        pygame.init()
        # configuration de la fenetre et des paramètres du robot
        self.largeur, self.hauteur = 800, 600
        self.screen = pygame.display.set_mode((self.largeur, self.hauteur))
        self.clock = pygame.time.Clock()

        # configuration de l'environnement carte, obstacles, etc 
        self.depart = depart
        if waypoints is None:
            waypoints = [(700, 500)]
        self.waypoints_originaux = waypoints.copy()  # on garde une copie pour pouvoir reset l'environnement
        
        self.vitesse_max = 5
        self.vitesse_rotation = 10 
        self.acceleration = 0.5    
        self.robot_rayon = 25
        
        # configuration de l'obastacle 
        self.liste_obstacles = [
            pygame.Rect(200, 0, 50, 400),    # Mur vertical qui descend du plafond
            pygame.Rect(200, 400, 300, 50),  # Mur horizontal en bas
            pygame.Rect(500, 150, 50, 300),  # Mur vertical qui remonte
        ]
        # on reset l'environnement pour initialiser les variables d'etat du robot et la premiere cible
        self.reset()

    # la fonction reset remet le robot a sa position de depart, 
    # remet les waypoints a leur ordre d'origine et choisit le premier waypoint comme cible
    def reset(self):
        self.x, self.y = self.depart
        self.angle = 0
        self.vitesse = 0    
        self.waypoints = self.waypoints_originaux.copy()
        self.arrivee = self.waypoints.pop(0)
        
        # on calcule la distance initiale pour pouvoir donner une recompose par rapport a l'eloignetment ou l'approche de la cible
        self.dist_precedente = math.dist((self.x, self.y), self.arrivee)
        
        # on retourne l'etat initial du robot pour que le reseau puisse commencer a apprendre
        return self.get_etat()

    def get_capteurs(self):
        # je repasse a 5 angles car plus precis, gauche, diag-gauche, devant, diag-droite, droite
        angles_capteurs = [-60, -30, 0, 30, 60]
        portee_max = 150 # il voit jusqu'à 150 pixels
        distances_normalisees = []
        
        for angle_relatif in angles_capteurs:
            angle_absolu = math.radians(self.angle + angle_relatif)
            dx = math.cos(angle_absolu)
            dy = math.sin(angle_absolu)
            
            distance = 0
            touche = False
            
            # on fait avancer le rayon pixel par pixel 
            while distance < portee_max:
                distance += 5
                test_x = self.x + dx * distance
                test_y = self.y + dy * distance
                
                # si le rayon sort de l'écran
                if test_x < 0 or test_x > self.largeur or test_y < 0 or test_y > self.hauteur:
                    touche = True
                    break
                    
                # si le rayon touche un obstacle
                point_test = pygame.Rect(test_x, test_y, 1, 1)
                if any(point_test.colliderect(obs) for obs in self.liste_obstacles):
                    touche = True
                    break
                    
            # on normalise entre 0 et 1 avec 1 voie libre et 0 dans le mur
            distances_normalisees.append(distance / portee_max)
            
        return distances_normalisees
    
    def collision_cercle_rect(self, rect):
        # point le plus proche du centre du robot sur le rectangle
        proche_x = max(rect.left, min(self.x, rect.right))
        proche_y = max(rect.top, min(self.y, rect.bottom))
        distance = math.sqrt((self.x - proche_x)**2 + (self.y - proche_y)**2)
        return distance < self.robot_rayon
    

    # la fonction get_etat calcule les 11 valeurs de l'etat a partir de la position,
    # l'angle, la vitesse du robot et la position de la cible et les obstacles capteur
    def get_etat(self):
        dx = self.arrivee[0] - self.x
        dy = self.arrivee[1] - self.y
        dist = math.sqrt(dx**2 + dy**2)
        angle_vers_cible = math.atan2(dy, dx)
        angle_robot_rad  = math.radians(self.angle)
        diff_angle = angle_vers_cible - angle_robot_rad
        
        # On recrup les 3 capteurs
        lidars = self.get_capteurs()
        
        # 8 valeurs de base
        etat_base = [
            self.x / self.largeur,
            self.y / self.hauteur,
            math.cos(angle_robot_rad), 
            math.sin(angle_robot_rad), 
            self.vitesse / self.vitesse_max,
            dist / math.sqrt(self.largeur**2 + self.hauteur**2),
            math.cos(diff_angle), 
            math.sin(diff_angle),      
        ]
        
        
        return np.array(etat_base + lidars, dtype=np.float32)
    
    # la fonction step applique l'action choisie par le reseau
    # met a jour la position du robot, calcule la recompense et 
    # retourne le nouvel etat, la recompense et si l episode est termine
    # les actions sont : 0 = accelerer, 1 = freiner, 2 = tourner a gauche, 3 = tourner a droite, 4 = ne rien faire
    def step(self, action):
        if action == 0: 
            self.vitesse = min(self.vitesse + self.acceleration, self.vitesse_max)
        elif action == 1: 
            self.vitesse = max(self.vitesse - self.acceleration, 0)
        elif action == 2: 
            self.angle = (self.angle - self.vitesse_rotation) % 360 
        elif action == 3: 
            self.angle = (self.angle + self.vitesse_rotation) % 360
       
        # physique standard pour faire avancer le robot dans la direction de son angle a une vitesse donnee
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.vitesse
        self.y += math.sin(rad) * self.vitesse
        
        reward = -0.1 # petite penalite de temps
        done = False 
        
        # on verifie les collisions avec les murs et les obstacles
        # si collision c'est une penalite et la fin de l episode
        # robot_rect = pygame.Rect(self.x-15, self.y-15, 30, 30)
        if (self.x < self.robot_rayon or self.x > self.largeur - self.robot_rayon or self.y < self.robot_rayon or self.y > self.hauteur - self.robot_rayon or any(self.collision_cercle_rect(obs) for obs in self.liste_obstacles)):
            return self.get_etat(), -200, True

        # on calcule la distance actuelle vers la cible pour donner  
        # une recompense positive s'il s'en approche et negative s'il s'en eloigne
        dist_actuelle = math.dist((self.x, self.y), self.arrivee)
        reward += (self.dist_precedente - dist_actuelle) * 0.5
        self.dist_precedente = dist_actuelle 

        # on donne une recompense supplementaire s
        # si le robot regarde dans la direction de la cible pour encourager un bon alignement 
        dx = self.arrivee[0] - self.x
        dy = self.arrivee[1] - self.y
        angle_vers_cible = math.atan2(dy, dx)
        alignement = math.cos(angle_vers_cible - math.radians(self.angle))
        reward += max(0, alignement) * 0.2

        # pour pas cheat on utilise une nouvelle logique simple
        # qui vont forcer le robot a apprendre a eviter les murs et les obstacles en lui donnant des penalites 
        # plus fortes plus il est proche d'eux plutot que de juste punir la collision apres qu'elle se soit produite 
        lidars = self.get_capteurs()
        distance_min = min(lidars)
        
        # donc si un mur est tres proche, moins de 40% de portee du lidar 
        if distance_min < 0.4:
            # plus il est proche de 0 donc le mur plus la penalitee est grossee
            reward -= (0.4 - distance_min) * 200.0
            
            # et pour choquer notre robot, on lui ajoute une penatlite en plus s'il fonce vite dans le mur
            if self.vitesse > self.vitesse_max * 0.5 and distance_min < 0.2:
                reward -= 50.0 

        # puis on regarde si les cible est atteinte, si oui on passe a la suivante ou on termine l episode si c'etait la derniere :

        # si cible atteinte avec une marge de 1.5 fois le rayon du robot pour que ce soit plus facile
        # on donne une grosse recompense et on passe a la cible suivante
        if dist_actuelle < self.robot_rayon * 1.5:
            if len(self.waypoints) > 0:
                self.arrivee = self.waypoints.pop(0)
                # on calcule la distance precedente pour la nouvelle cible pour que le robot puisse 
                # continuer a recevoir des recompenses d'approche ou d'eloignement
                self.dist_precedente = math.dist((self.x, self.y), self.arrivee)
                reward += 100 
            else:
                reward += 200
                done = True

        # on bride les petites recompenses mais on laisse passer les bonus d'arrivee
        #if reward < 50:
        #    reward = max(-10.0, min(10.0, reward))
        return self.get_etat(), reward, done
    
    def dessiner(self):
        self.screen.fill((230, 230, 230))
        
        # on dessine les obstacle
        for obs in self.liste_obstacles: 
            pygame.draw.rect(self.screen, (255, 0, 0), obs)
            
        # on dessine tout le chemin RRT
        # on cree une liste avec la position du robot la cible actuelle les cibles future
        points_chemin = [(self.x, self.y), self.arrivee] + self.waypoints
        if len(points_chemin) > 1:
            # Trace une ligne verte foncée qui relie tous les points
            pygame.draw.lines(self.screen, (0, 150, 0), False, points_chemin, 2)
            
        # on dessine les waypoints restant en petits points vert
        for wp in self.waypoints:
            pygame.draw.circle(self.screen, (0, 150, 0), (int(wp[0]), int(wp[1])), 5)
            
        # on dessine la cible actuelle en vert clair
        pygame.draw.circle(self.screen, (0, 255, 0), (int(self.arrivee[0]), int(self.arrivee[1])), 15)
        
        # on dessine le robot avec une ligne blanche qui indique sa direction
        pygame.draw.circle(self.screen, (50, 50, 50), (int(self.x), int(self.y)), self.robot_rayon)
        end_x = self.x + math.cos(math.radians(self.angle)) * 30
        end_y = self.y + math.sin(math.radians(self.angle)) * 30
        pygame.draw.line(self.screen, (255, 255, 255), (self.x, self.y), (end_x, end_y), 3)
        
        pygame.display.flip()