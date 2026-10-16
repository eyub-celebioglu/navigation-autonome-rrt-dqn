import numpy as np
import random
import pygame

class treeNode():
    def __init__(self, X, Y, theta=None):
        self.X = X              # position x
        self.Y = Y              # position y
        self.theta = theta      # position angulaire (au départ nulle)
        self.children = []      # enfants des noeuds
        self.parent = None      # parent du noeud
        self.cost = 0           # coût depuis la racine


class RRTStar():
    def __init__(self, start, goal, iterations, grid, stepSize, theta):
        self.randomTree = treeNode(start[0], start[1])  # position de départ
        self.goal = treeNode(goal[0], goal[1])           # objectif
        self.nearestNode = None                          # plus proche noeud
        self.iterations = iterations                     # nombre total d'itérations
        self.grid = grid                                 # liste de rectangles obstacles
        self.lengthBranch = stepSize                     # longueur des branches
        self.angleBranch = np.radians(theta) if theta is not None else None  # angle max des branches
        self.pathDist = 0                                # longueur totale parcourue
        self.nearestDist = 10000                         # distance du plus proche noeud
        self.numberWaypoints = 0                         # nombre de points de repère
        self.waypoints = []                              # liste des points de repères
        self.toutes_les_branches = []                    # liste de toutes les branches

    def sampleAPoint(self):
        # Génère un point aléatoire dans la grille pour l'exploration
        x = random.randint(1, 800)
        y = random.randint(1, 600)
        return np.array([x, y])

    def steerToPoint(self, start, end):
        # Calcule un nouveau point dans la direction de end depuis start
        # à une distance égale à la longueur d'une branche
        offset = self.lengthBranch * self.unitVector(start, end)
        point = np.array([start.X + offset[0], start.Y + offset[1]])

        # Calcule de l'angle
        theta = np.arctan2(point[1] - start.Y, point[0] - start.X)

        # Si angle trop important, restreindre l'angle
        if start.theta is not None and self.angleBranch is not None:
            diff = theta - start.theta
            diff = (diff + np.pi) % (2 * np.pi) - np.pi  # normalise dans [-pi, pi]
            diff = np.clip(diff, -self.angleBranch, self.angleBranch)
            theta = start.theta + diff

        # Recalcule du point après restriction d'angle
        point[0] = start.X + self.lengthBranch * np.cos(theta)
        point[1] = start.Y + self.lengthBranch * np.sin(theta)

        # On empêche le point de sortir de la fenêtre
        point[0] = np.clip(point[0], 0, 799)
        point[1] = np.clip(point[1], 0, 599)

        return point, theta

    def isInObstacle(self, start, end):
        # Vérifie pixel par pixel si la branche touche un obstacle
        u_hat = self.unitVector(start, end)
        testPoint = np.array([0.0, 0.0])
        for i in range(self.lengthBranch):
            testPoint[0] = start.X + i * u_hat[0]
            testPoint[1] = start.Y + i * u_hat[1]
            for rect in self.grid:
                if rect.collidepoint(testPoint[0], testPoint[1]):
                    return True
        return False

    def unitVector(self, start, end):
        # Calcule et retourne le vecteur unitaire normalisé entre deux points
        vector = np.array([end[0] - start.X, end[1] - start.Y])
        norme = np.linalg.norm(vector)
        if norme == 0:
            return np.array([0.0, 0.0])
        return vector / norme

    def findNearest(self, root, point, radius, result=None):
        # Recherche récursive dans l'arbre les noeuds dans un rayon donné
        # et met à jour self.nearestNode avec le plus proche
        if result is None:
            result = []
        if not root:
            return result
        dist = self.distance(root, point)
        if dist <= self.nearestDist and dist <= radius:
            self.nearestNode = root
            self.nearestDist = dist
            result.append(root)
        for child in root.children:
            self.findNearest(child, point, radius, result)
        return result

    def updateChildrenCost(self, node):
        # Met à jour récursivement le coût de tous les enfants après un rewiring
        for child in node.children:
            child.cost = node.cost + self.distance(node, [child.X, child.Y])
            self.updateChildrenCost(child)

    def distance(self, noeud1, point):
        # Calcule la distance euclidienne entre un noeud et un point
        return np.sqrt((noeud1.X - point[0])**2 + (noeud1.Y - point[1])**2)

    def goalFound(self, point):
        # Indique si un point est suffisamment proche du but
        if self.distance(self.goal, point) <= self.lengthBranch:
            return True

    def resetNearestValues(self):
        # Réinitialise les variables pour trouver le noeud le plus proche
        self.nearestNode = None
        self.nearestDist = 10000

    def retraceRRTPath(self, goal):
        # Remonte depuis le noeud but jusqu'à la racine en enregistrant le chemin
        node = goal
        while node.parent is not None:
            self.numberWaypoints += 1
            self.waypoints.insert(0, np.array([node.X, node.Y]))
            self.pathDist += self.lengthBranch
            node = node.parent

    def rewiring(self,Xneighbors,Xnew,Xbest):
        for n in Xneighbors:
            if n is Xbest or n is self.randomTree:
                continue
            newCost = Xnew.cost + self.distance(Xnew, [n.X, n.Y])
            if newCost < n.cost and not self.isInObstacle(Xnew, [n.X, n.Y]):
                if n.parent is not None and n in n.parent.children:
                    n.parent.children.remove(n)
                n.parent = Xnew
                n.cost = newCost
                Xnew.children.append(n)
                self.updateChildrenCost(n)

    def chooseParent(self,Xneighbors,Xnew,Xnearest):
        Xbest = Xnearest
        bestCost = Xnew.cost
        for n in Xneighbors:
            cout_via_n = n.cost + self.distance(n, [Xnew.X, Xnew.Y])
            if cout_via_n < bestCost and not self.isInObstacle(n, [Xnew.X, Xnew.Y]) and self.verifAngle(n, Xnew):
                Xbest = n
                bestCost = cout_via_n
            return Xbest,bestCost

    def verifAngle(self,parentNode,childNode):
        if self.angleBranch is None:
            return True
        if parentNode.theta is None:
            return True
        angleBranche = np.arctan2(childNode.Y - parentNode.Y,childNode.X-parentNode.X)

        diffAngle = angleBranche - parentNode.theta
        diffAngle = (diffAngle+np.pi) % (2*np.pi) - np.pi

        return abs(diffAngle) <= self.angleBranch

    def compute_path(self):
        self.randomTree = treeNode(self.randomTree.X, self.randomTree.Y, theta=None)
        self.nearestNode = None
        self.pathDist = 0
        cout_initial = 0
        self.waypoints = []
        self.toutes_les_branches = []
        meilleur_cout = float('inf')
        meilleur_chemin = None
        chemin_trouve = False

        # Boucle principale 
        
        for itr in range(self.iterations):

            
            point = self.sampleAPoint()

            # si obstacle on recommence
            obstacle = any(rect.collidepoint(point[0], point[1]) for rect in self.grid)
            if obstacle:
                continue

            
          
            self.resetNearestValues()
            tous = self.findNearest(self.randomTree, point, radius=float('inf'), result=None)
            Xnearest = self.nearestNode

            new, theta = self.steerToPoint(Xnearest, point)
            Xnew = treeNode(new[0], new[1], theta=theta)

            if self.isInObstacle(Xnearest, new):
                continue

            Xnew.cost = Xnearest.cost + self.distance(Xnearest, [Xnew.X, Xnew.Y])

            # Filtre les voisins dans le rayon 50 autour de Xnew
            Xneighbors = [n for n in tous if self.distance(n, [Xnew.X, Xnew.Y]) <= 50]

            # Trouve le meilleur parent parmi les voisins
            Xbest, bestCost = self.chooseParent(Xneighbors, Xnew, Xnearest)
           
            # ajoute au chemin le meilleur
            Xnew.cost = bestCost
            Xnew.parent = Xbest
            Xbest.children.append(Xnew)
            self.toutes_les_branches.append(((Xbest.X, Xbest.Y), (Xnew.X, Xnew.Y)))

            # recablage (pour nos amis quebecois) des noeuds 
            self.rewiring(Xneighbors, Xnew, Xbest)

            # on vérifie le but
            if self.goalFound(new):
                chemin_trouve = True
                self.numberWaypoints = 0
                self.waypoints = []
                self.retraceRRTPath(Xnew)
                cout_initial = Xnew.cost
                cout_actuel = Xnew.cost
                if cout_actuel < meilleur_cout:
                    
                    meilleur_cout = cout_actuel
                    meilleur_chemin = [(self.randomTree.X, self.randomTree.Y)]
                    for p in self.waypoints:
                        meilleur_chemin.append((p[0], p[1]))

            # Vérifie aussi si le RECABLAGE a amélioré un voisin déjà proche du but
            if chemin_trouve:
                for n in Xneighbors:
                    if self.goalFound([n.X, n.Y]):
                        self.numberWaypoints = 0
                        self.waypoints = []
                        self.retraceRRTPath(n)
                        cout_actuel = n.cost
                        if cout_actuel < meilleur_cout:
                            
                            meilleur_cout = cout_actuel
                            meilleur_chemin = [(self.randomTree.X, self.randomTree.Y)]
                            for p in self.waypoints:
                                meilleur_chemin.append((p[0], p[1]))

        
        return meilleur_chemin, self.toutes_les_branches