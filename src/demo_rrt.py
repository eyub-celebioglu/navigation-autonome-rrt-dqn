import pygame
import math
import sys
from rrt import RRT
from rrt_sc import RRTSC
from rrt_star import RRTStar

def arguments():
    try:
        algo = int(input("Choisir entre RRT taper 1 , RRTSC taper 2  ou RRTStart taper 3: "))
        if algo != 1 and algo != 2 and algo !=3:
            print("Veuillez choisir entre 1, 2 ou 3")
            return arguments()  
        nbIt = int(input("Nombre d'itération : "))
        lb = int(input("Longueur des branches : "))
        Inputangle = input("Angle des branches (mettre None si pas d'angle): ")
        angle = None if Inputangle == "None" else int(Inputangle)
        return algo, nbIt, lb, angle
    except ValueError:
        print("Entrée invalide, veuillez recommencer")
        return arguments()



algo,iterations,long_branche,teta=arguments()
largeur, hauteur = 800, 600
fenetre = pygame.display.set_mode((largeur, hauteur))
pygame.init()
# Ajout ligne de commande pour les paramètres (mettre plus si besoin de modifier plus de paramètre)


fond_gris = (200, 200, 200)
robot_couleur = (50, 50, 50) # couleur noire

class robot:
    def __init__(self, x, y, vitesse_max, vitesse_rotation):
        self.vitesse_max = vitesse_max
        self.vitesse_rotation = vitesse_rotation
        self.vitesse = 0  # notre robot a une vitesse de 0 au debut
        self.angle = 0 # l'angle permet de nous guider dans quels direction le robot va tourner
        self.x = x
        self.y = y
        self.robot_rayon = 30
        self.acceleration = 0.1 # chaque fois qu on veut acclerer

    def rotation(self, gauche=False, droite=False):
        if gauche:
            self.angle += self.vitesse_rotation # En reduisant la vitesse de rotation notre angle tend vers la gauche
        elif droite:
            self.angle -= self.vitesse_rotation
    
    def avancer_droit(self):
        self.vitesse = min(self.vitesse + self.acceleration, self.vitesse_max) # limitateur de vitesse pour pas allew a l'infini on est pas dans l'espace
        self.mouvement()
    
    def mouvement(self):
        rad = math.radians(self.angle) # conversion en radian
        # On veut savoir combien le robot avance vers le haut donc  cosinus = adjacent / hypotenuse -> cos(teta) = verticale / vitesse
        # donc verticale = cos(teta) * vitesse
        verticale = math.cos(rad) * self.vitesse
        # mm logique pour l'horizontale sauf qu'ici on a sinus = oppose / Hypotenuse -> sin(teta) = horizontale / vitesse
        # donc  horizontale = sin(teta) * vitesse
        horizontale = math.sin(rad) * self.vitesse

        self.y -= verticale
        self.x -= horizontale


# on initialise le robot ici
robot_test = robot(largeur//2, hauteur//2, 5, 5)

# Boucle qui fait afficher notre fenetre tant qu on le ferme pas
run = True
clock = pygame.time.Clock() # Horloge pour gerer les images par seconde
IPS = 30

liste_obstacles = []
rectangle1 = pygame.Rect(200, 150, 130, 330) # ajout du rayon du robot dans la longueur et la largeur des rectangles
liste_obstacles.append(rectangle1)
rectangle2 = pygame.Rect(500, 150, 130, 330)        
liste_obstacles.append(rectangle2)
rectangle3 = pygame.Rect(350, 150, 130, 130)
liste_obstacles.append(rectangle3)


depart = (50, 50)          # point de départ du robot
arrivee = (700, 500)       # objectif en bas à droite
          # longueur d'une branche
        # nombre d'essais pour trouver le chemin dans le code de RRT fixée par yoann
                   # angle de départ  (mettre a None pour comparer l'arbre avec et sans restriction d'angle )

# on place le robot au départ
robot_test.x = depart[0]
robot_test.y = depart[1]

if algo == 1:
    solveur = RRT(depart, arrivee,iterations, liste_obstacles, long_branche,teta)
elif algo == 2:
    solveur = RRTSC(depart, arrivee,iterations, liste_obstacles, long_branche,teta)
elif algo ==3:
    solveur = RRTStar(depart,arrivee,iterations,liste_obstacles,long_branche,teta)

    
chemin_trouve, arbre_complet = solveur.compute_path() # On récupère la liste de points du chemin trouvé pour l'afficher dans le simulateur


    

while run:
    clock.tick(IPS)

   
    fenetre.fill(fond_gris)
    for obstacle in liste_obstacles:
        pygame.draw.rect(fenetre, (255, 0, 0), obstacle) # on dessine les obstacles en rouge

    # on dessine les branches de l'arbre de recherche
    for branche in arbre_complet:
        # une branche contient ((x1, y1), (x2, y2))
        pygame.draw.line(fenetre, (150, 150, 150), branche[0], branche[1], 1)
    
    if chemin_trouve != None: # on dessine que si on a trouvé un chemin

        # pygame.draw.lines(surface, couleur, fermé ?, liste_points, épaisseur)
        pygame.draw.lines(fenetre, (0, 0, 255), False, chemin_trouve, 1)
        
    else: # Si pas de chemin trouvé affiche bulle de texte

        # Police et taille du texte
        font = pygame.font.SysFont('Arial', 24)  
        text_surface = font.render("Pas de chemin trouvé", True, (0,0,0))
        padding = 10
        bubble_rect = text_surface.get_rect()
        bubble_rect.inflate_ip(padding*2, padding*2)  # élargit le rectangle pour le padding
        bubble_rect.topleft = (300, 300)  # position à l'écran
        
        pygame.draw.rect(fenetre, (255,255,255), bubble_rect, border_radius=8)

        # Dessiner le texte
        fenetre.blit(text_surface, (bubble_rect.x + padding, bubble_rect.y + padding))

    # on dessiner le point d'arrivée en Vert
    pygame.draw.circle(fenetre, (0, 255, 0), arrivee, 10)
    pygame.draw.circle(fenetre,robot_couleur, (robot_test.x, robot_test.y),robot_test.robot_rayon)
    pygame.display.flip()

    

    for event in pygame.event.get(): # Fermer la fenetre
        if event.type == pygame.QUIT:
            run = False
            break
    
    touche_clavier = pygame.key.get_pressed()

    if touche_clavier[pygame.K_q]:
        robot_test.rotation(gauche=True)
    if touche_clavier[pygame.K_d]:
        robot_test.rotation(droite=True)
    if touche_clavier[pygame.K_z]:
        robot_test.avancer_droit()
pygame.quit()