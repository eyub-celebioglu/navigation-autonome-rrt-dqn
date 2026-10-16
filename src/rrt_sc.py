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


    

class RRTSC():
    def __init__(self, start, goal, iteration, grid, stepSize,theta):
        self.randomTree = treeNode(start[0],start[1]) # position de départ
        self.goal = treeNode(goal[0],goal[1])          # objectif
        self.nearestNode = None                         # plus proche noeud
        self.iteration = iteration            
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
    
    def steerToPoint(self, start,end,length):
        # Calcule un nouveau point dans la direction de end depuis star
        # à une distance égale à la longueur d’une branche 
        offset = length*self.unitVector(start,end)
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
        point[0] = start.X + length * np.cos(theta) 
        point[1] = start.Y + length * np.sin(theta)
        

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
    
    def isInObstacleFullPath(self, nodeI, pointJ):
        # Calcule la distance réelle entre les deux points
        dist = int(self.distance(nodeI, pointJ))
        if dist == 0:
            return False
        u_hat = self.unitVector(nodeI, pointJ)
        testPoint = np.array([0.0, 0.0])
        for i in range(dist):  # parcourt toute la distance du chemin
            testPoint[0] = nodeI.X + i * u_hat[0]
            testPoint[1] = nodeI.Y + i * u_hat[1]
            for rect in self.grid:
                if rect.collidepoint(testPoint[0], testPoint[1]):
                    return True
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

    def verifAngle(self,parentNode,childNode):
        if self.angleBranch is None:
            return True
        if parentNode.theta is None:
            return True
        angleBranche = np.arctan2(childNode.Y - parentNode.Y,childNode.X-parentNode.X)

        diffAngle = angleBranche - parentNode.theta
        diffAngle = (diffAngle+np.pi) % (2*np.pi) - np.pi

        return abs(diffAngle) <= self.angleBranch
    
    def shortcut_path(self, path, seuil):
        disk_radius = self.lengthBranch * 0.25

        cout_precedent = float('inf')
        ratio = 1.0

        while ratio > seuil: # Tant qu'on peut améliorer le chemin on continu
            
            ameliore = False
            i = 0
            cout_precedent = sum(
                self.distance(treeNode(path[k][0], path[k][1]), [path[k+1][0], path[k+1][1]])
                for k in range(len(path) - 1))

            while i < len(path) - 2:
                nodeI = treeNode(path[i][0], path[i][1])
                theta_i = np.arctan2(path[i+1][1] - path[i][1], path[i+1][0] - path[i][0])
                nodeI.theta = theta_i

                meilleur_j = None
                meilleure_branche = None

                # On tire des branches de longueur croissante 
                for multiplicateur in range(1, len(path) - i):
                    longueur = self.lengthBranch * multiplicateur

                    # On tire dans plusieurs directions
                    for j in range(i + 2, len(path)):
                        #On vise un noeud plus loin dans le chemin
                        cible = np.array([path[j][0], path[j][1]])
                        
                        
                        new_point, _ = self.steerToPoint(nodeI, cible, longueur)
                        branche_end = [new_point[0], new_point[1]]

                         # si on touche un obstacle on s'arrete
                        if self.isInObstacleFullPath(nodeI, branche_end): 
                            break 

                        dist_branche_disqueJ = self.distance(treeNode(new_point[0], new_point[1]), [path[j][0], path[j][1]])
                        if dist_branche_disqueJ <= disk_radius:
                            if meilleur_j is None or j > meilleur_j:
                                meilleur_j = j
                                meilleure_branche = (new_point[0], new_point[1])

                if meilleur_j is not None:
                    nodeI_verif = treeNode(path[i][0], path[i][1])
                    nodeI_verif.theta = theta_i

                    nodeBranche = treeNode(meilleure_branche[0], meilleure_branche[1])

                    # Angle du noeud meilleur_j dans le chemin
                    if meilleur_j < len(path) - 1:
                        theta_j = np.arctan2(path[meilleur_j+1][1] - path[meilleur_j][1],
                                            path[meilleur_j+1][0] - path[meilleur_j][0])
                    else:
                        theta_j = theta_i  # dernier noeud, pas de contrainte
                    nodeJ_verif = treeNode(path[meilleur_j][0], path[meilleur_j][1])
                    nodeJ_verif.theta = theta_j

                    if self.verifAngle(nodeI_verif, nodeBranche) and self.verifAngle(nodeBranche, nodeJ_verif):
                        path = path[:i+1] + [meilleure_branche] + path[meilleur_j:]
                        ameliore = True

                i += 1

            cout_actuel = sum(
                self.distance(treeNode(path[k][0], path[k][1]), [path[k+1][0], path[k+1][1]])
                for k in range(len(path) - 1)
            )

            if cout_precedent == float('inf'):
                ratio = 1.0
            else:
                ratio = (cout_precedent - cout_actuel) / cout_precedent if cout_actuel < cout_precedent else 0.0

            

            if not ameliore:
                break

        return path


    def compute_path(self):
        self.randomTree = treeNode(self.randomTree.X, self.randomTree.Y, theta=None)
        self.nearestNode = None
        self.pathDist = 0
        self.waypoints = []
        self.toutes_les_branches = []

        
        for i in range(self.iteration):
            
            self.resetNearestValues()
            point = self.sampleAPoint()
            self.findNearest(self.randomTree, point)
            new, theta = self.steerToPoint(self.nearestNode, point,self.lengthBranch)

            if not self.isInObstacle(self.nearestNode, new):
                self.addChild(new[0], new[1], theta=theta)

                if self.goalFound(new):
                    self.addChild(self.goal.X, self.goal.Y, theta=theta)
                    self.retraceRRTPath(self.goal)

                    # Chemin brut
                    path_for_pygame = [(self.randomTree.X, self.randomTree.Y)]
                    for p in self.waypoints:
                        path_for_pygame.append((p[0], p[1]))
    
                    # Phase d'amélioration par raccourcis
                    
                    path_for_pygame = self.shortcut_path(path_for_pygame,0.001) # Mettre 1.0 pour pas d'amélioration
                    
                    return path_for_pygame, self.toutes_les_branches

        return None, self.toutes_les_branches
