import numpy as np
import random
import pygame  # pour les rectangles e

class treeNode():
    def __init__(self, X, Y,theta=None):
        self.X = X #position x
        self.Y = Y  # position y
        self.theta = theta # position angulaire (au départ nulle)
        self.children = [] #enfants des noeuds
        self.parent = None # parents des noeuds


    

class RRT():
    def __init__(self, start, goal, iteration, grid, stepSize,theta):
        self.randomTree = treeNode(start[0],start[1]) # position de départ
        self.goal = treeNode(goal[0],goal[1])          # objectif
        self.nearestNode = None                         # plus proche noeud
        self.iteration = iteration             # nombre total d'itération (limite a 200)
        self.grid = grid                                # la carte (liste de rectangles maintenant que c'est pygame)
        self.lengthBranch = stepSize                    # longueur des branches
        self.angleBranch = np.radians(theta) if theta is not None else None           # angle des branches (converti en radians)
        self.pathDist = 0                               # longueur total parcouru
        self.nearestDist = 10000                        # distance du plus proche noeud (intialiser a un trs gros nombre)
        self.numberWaypoints = 0                        # nombre de point de repère
        self.waypoints = []                             #liste des points de repères
        self.toutes_les_branches = []                    # liste de toutees les branches

    def addChild(self, x,y,theta=None):
        # Ajoute un nouveau noeud enfant au noeud le plus proche
        # si le noeud correspond au but il connecte directement au but
        if( x == self.goal.X):
            self.toutes_les_branches.append(((self.nearestNode.X, self.nearestNode.Y), (self.goal.X, self.goal.Y)))
            self.nearestNode.children.append(self.goal)
            self.goal.parent = self.nearestNode
            self.goal.theta = theta
        else:
            tampNode = treeNode(x,y,theta=theta)
            self.toutes_les_branches.append(((self.nearestNode.X, self.nearestNode.Y), (x, y)))
            self.nearestNode.children.append(tampNode)
            tampNode.parent = self.nearestNode

    def sampleAPoint(self):
        # Génère un point aléatoire dans la grille pour l’exploration
        x = random.randint(1, 800) # taille de la fenetre 800x600
        y = random.randint(1, 600)
        point = np.array([x,y])
        return point
    
    def steerToPoint(self, start,end):
        # Calcule un nouveau point dans la direction de end depuis star
        # à une distance égale à la longueur d’une branche 
        offset = self.lengthBranch*self.unitVector(start,end)
        point = np.array([start.X + offset[0], start.Y + offset[1]])

        #calcule de l'angle
        theta = np.arctan2(point[1] - start.Y, point[0] - start.X)
       
        
        # Si angle trop important restreindre l'angle
        if start.theta is not None and self.angleBranch is not None:
            diff = theta - start.theta

            diff = (diff + np.pi)% (2*np.pi) - np.pi # normalise dans [-pi,pi]

            diff = np.clip(diff,-self.angleBranch,self.angleBranch)
            theta = start.theta + diff

        

        # recalcule du point après restriction d'angle
        point[0] = start.X + self.lengthBranch * np.cos(theta) 
        point[1] = start.Y + self.lengthBranch * np.sin(theta)
        

        # On empeche le point de sortir de la fenetre
        if point[0] >= 800: point[0] = 800 - 1
        if point[1] >= 600: point[1] = 600 - 1
        if point[0] < 0: point[0] = 0
        if point[1] < 0: point[1] = 0

        return point,theta
    
    def isInObstacle(self,start,end):
        u_hat = self.unitVector(start,end) 
        testPoint = np.array([0.0,0.0])
        # elle regarde pixel par pixel si la branche touche un obstacle en avançant de 1 pixel à la fois dans la direction de la branche
        # donc impossible que la branche traverse un obstacle sans que cette fonction le détecte

        
        for i in range(self.lengthBranch):
            testPoint[0] = start.X+ i*u_hat[0]
            testPoint[1] = start.Y+ i*u_hat[1]
            for rect in self.grid:   # on parcourt les rectangles de la grille pour voir si le point de test est dans un obstacle
                if rect.collidepoint(testPoint[0], testPoint[1]):
                    return True #
        return False

    def unitVector(self,start,end):
        # Calcule et retourne le vecteur unitaire normalisé entre deux points
        vector = np.array([end[0]-start.X, end[1]-start.Y])
        u_hat = vector/np.linalg.norm(vector)
        return u_hat
    
    def findNearest(self,root,point):
        # Recherche récursive dans l’arbre le noeud le plus proche d’un point donné
        if not root:
            return
        dist = self.distance(root,point)
        if dist <= self.nearestDist:
            self.nearestNode = root
            self.nearestDist = dist
        for i in root.children:
            self.findNearest(i,point)


    def distance(self,noeud1,point):
         # Calcule la distance euclidienne entre un noeud et un point
        dist = np.sqrt((noeud1.X - point[0])**2 + (noeud1.Y - point[1])**2)
        return dist
    
    def goalFound(self,point):
        # Indique si un point est suffisamment proche du but pour considérer que le but est atteint
        if self.distance(self.goal,point) <= self.lengthBranch:
            return True

    def resetNearestValues(self):
        # Réinitialise les variables utilisées pour trouver le noeud le plus proche avant une nouvelle recherche
        self.nearestNode = None
        self.nearestDist = 10000
    
    def retraceRRTPath(self,goal):
        # Remonte récursivement depuis le noeud but jusqu’à la racine
        # en enregistrant le chemin et calculant la distance totale
        if goal.X == self.randomTree.X:
            return
        self.numberWaypoints += 1
        currentPoint = np.array([goal.X, goal.Y])
        self.waypoints.insert(0,currentPoint)
        self.pathDist += self.lengthBranch
        self.retraceRRTPath(goal.parent)

    def compute_path(self):
    
        # on remet à zéro pour que tout soit propre
        self.randomTree = treeNode(self.randomTree.X, self.randomTree.Y,theta=None)
        self.nearestNode = None
        self.pathDist = 0
        self.waypoints = []
        self.toutes_les_branches = []

        # Boucle principale de recherche
        for i in range(self.iteration):
            # On réinitialise la distance min à chaque tour
            self.resetNearestValues()

            # On tire un point au hasard
            point = self.sampleAPoint()
            
            # On trouve le noeud le plus proche dans l'arbre
            self.findNearest(self.randomTree, point)
            
            # On crée un nouveau point dans cette direction avec un angle
            new,theta = self.steerToPoint(self.nearestNode, point)
            
            # On vérifie si on touche un obstacle un rectangle pygame
            if not self.isInObstacle(self.nearestNode, new):
                # Si on a pas d'obstacle alors on ajoute le point à l'arbre
                self.addChild(new[0], new[1],theta = theta)
                
                # Si on est arrivé au but 
                if self.goalFound(new):
                    self.addChild(self.goal.X, self.goal.Y,theta = theta)
                    
                    # alors on reconstruit le chemin
                    self.retraceRRTPath(self.goal)
                    
                    # on convertit le chemin en liste simple pour le simulateur en pygame
                    path_for_pygame = []
                    # on ajoute le départ
                    path_for_pygame.append((self.randomTree.X, self.randomTree.Y)) # on l'apelle rendomTree mais cest bien un départ 
                    # on ajoute les autres points                                   # elle est fixée dans le constructeur
                    for p in self.waypoints:
                        path_for_pygame.append((p[0], p[1]))
                        
                    return path_for_pygame , self.toutes_les_branches # on renvoie les listes au simulateur
        return None, self.toutes_les_branches # Si pas de chemin au but trouvé renvoi toutes les branches de l'arbre et None
