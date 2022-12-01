import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
pygame.init()
zoneText= pygame.font.SysFont("arial",25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# reset
# reward
#play(action) -> direction
#game iteration
#is collision
Point= namedtuple("Point","x, y")

#RGB COLORS

Blanc=(255,255,255)
Rouge=(200,0,0)
Bleu1=(0,0,255)
Bleu2=(0,100,255)
Noir=(0,0,0)

Taille= 20
Speed=50
class SnakeAI:

    def __init__(self,longueur=640,largeur=480):
        self.longueur=longueur
        self.largeur=largeur

        #init display
        self.display=pygame.display.set_mode((self.longueur,self.largeur))
        pygame.display.set_caption("Snake leandro")
        self.clock=pygame.time.Clock()
        self.reset()
        #init game state
    def reset(self):
        self.direction= Direction.RIGHT
        self.tete= Point(self.longueur/2, self.largeur/2)
        self.snake= [self.tete,Point(self.tete.x-Taille,self.tete.y), Point(self.tete.x-(2*Taille),self.tete.y)]
        self.score=0
        self.food=None
        self._spawn_food()
        self.iteration=0

    def _spawn_food(self):
        x = random.randint(0,(self.longueur-Taille)//Taille)* Taille
        y = random.randint(0,(self.largeur-Taille)//Taille)*Taille
        self.food= Point(x,y)
        if self.food in self.snake:
            self._spawn_food()

    def etapeDuJeu(self,action):
        #Collect user input
        self.iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


        #move
        self.move(action) #met a jour la tete
        self.snake.insert(0,self.tete)
        recompense= 0
        gameOver=False
        if self.isCollision() or self.iteration > 100*len(self.snake):
            gameOver=True
            recompense= -10
            return recompense,gameOver,self.score

        #place new food or just move

        if self.tete == self.food:
            self.score += 1
            recompense=10
            self._spawn_food()
        else:
            self.snake.pop()

        #update ui and clock

        self.updateUi()
        self.clock.tick(Speed)

        #return game over and score

        gameOver=False
        return recompense,gameOver,self.score
    def isCollision(self,pt=None):

        if pt is None:
            pt=self.tete

        if pt.x > self.longueur - Taille or pt.x < 0 or pt.y > self.largeur -Taille or pt.y < 0:
            return True

        if pt in self.snake[1:]:
            return True

        return False

    def updateUi(self):
        self.display.fill(Noir)
        for pt in self.snake:
            pygame.draw.rect(self.display,Bleu1,pygame.Rect(pt.x,pt.y,Taille,Taille))
            pygame.draw.rect(self.display, Bleu2, pygame.Rect(pt.x+4, pt.y+4, Taille-8, Taille-8))

        pygame.draw.rect(self.display,Rouge,pygame.Rect(self.food.x,self.food.y,Taille,Taille))

        text= zoneText.render("Score: "+ str(self.score),True,Blanc)
        self.display.blit(text,[0,0])
        pygame.display.flip()

    def move(self,action):
        #[Continue,droite,gauche]

        clockWise=[Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx = clockWise.index(self.direction)

        if np.array_equal(action,[1,0,0]):
            newDirection=clockWise[idx]
        elif np.array_equal(action,[0,1,0]):
            nextIdx=(idx+1)%4
            newDirection=clockWise[nextIdx]
        else:
            nextIdx = (idx - 1) % 4
            newDirection = clockWise[nextIdx]
        self.direction = newDirection


        x= self.tete.x
        y=self.tete.y
        if self.direction == Direction.RIGHT:
            x+=Taille
        elif self.direction == Direction.LEFT:
            x -= Taille
        elif self.direction == Direction.DOWN:
            y += Taille
        elif self.direction == Direction.UP:
            y -= Taille

        self.tete= Point(x,y)



