import torch
import random
import numpy as np
from collections import deque
from Snake_Leandro_AI import SnakeAI,Direction,Point
from model import Linear_Qnet,QTrainer
from helper import plot
MaxMemory=100_00
BATCH_SIZE=1000
LearningRate=0.001

class Agent:

    def __init__(self):
        self.n_games=0
        self.epsilon=0 #randomness
        self.gamma=0.9 #discount rate
        self.memoire=deque(maxlen=MaxMemory) #Vire les premiers elements quand depasse memoire
        self.model = Linear_Qnet(11,256,3)
        self.trainer = QTrainer(self.model,lr=LearningRate,gamma=self.gamma)
        #TODO:model,trainer

    def getstate(self,game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.isCollision(point_r)) or
            (dir_l and game.isCollision(point_l)) or
            (dir_u and game.isCollision(point_u)) or
            (dir_d and game.isCollision(point_d)),

            # Danger right
            (dir_u and game.isCollision(point_r)) or
            (dir_d and game.isCollision(point_l)) or
            (dir_l and game.isCollision(point_u)) or
            (dir_r and game.isCollision(point_d)),

            # Danger left
            (dir_d and game.isCollision(point_r)) or
            (dir_u and game.isCollision(point_l)) or
            (dir_r and game.isCollision(point_u)) or
            (dir_l and game.isCollision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.tete.x,  # food left
            game.food.x > game.tete.x,  # food right
            game.food.y < game.tete.y,  # food up
            game.food.y > game.tete.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self,state,action,reward,nextState,done):
        self.memoire.append((state,action,reward,nextState,done))

    def trainLongMemory(self):
        if len(self.memoire) > BATCH_SIZE:
            miniSample=random.sample(self.memoire,BATCH_SIZE)#liste de tuples
        else:
            miniSample=self.memoire
        states,actions,rewards,nextStates,dones=zip(*miniSample)
        self.trainer.trainStep(states, actions, rewards, nextStates, dones)

    def trainShortMemory(self, state, action, reward, next_state, done):
        self.trainer.trainStep(state, action, reward, next_state, done)

    def getAction(self,state):
        self.epsilon=80-self.n_games
        finalMove=[0,0,0]
        if random.randint(0,200)< self.epsilon:
            move=random.randint(0,2)
            finalMove[move]=1
        else:
            state0=torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
            move=torch.argmax(prediction).item()
            finalMove[move]=1
        return finalMove



def train():
    plotScore=[]
    plotMeanScore1=[]
    totalScore=0
    best=0
    agent=Agent()
    game=SnakeAI()
    while True:
        #Avoir le dernier etat
        stateOld = agent.getstate(game)
        #getMove
        final_move=agent.getAction(stateOld)
        #ameliore le move
        reward,done,score=game.etapeDuJeu(final_move)
        stateNew=agent.getstate(game)
        #entraine la short memory
        agent.trainShortMemory(stateOld,final_move,reward,stateNew,done)
        #retient
        agent.remember(stateOld,final_move,reward,stateNew,done)
        if done:
            #entraine long memory
            game.reset()
            agent.n_games += 1
            agent.trainLongMemory()

            if score > best:
                best=score
                agent.model.save()

            print('Partie jouée :',agent.n_games,'Score :',score,'Record :',best)
            plotScore.append(score)
            totalScore += score
            plotMeanScore=totalScore/agent.n_games
            plotMeanScore1.append(plotMeanScore)
            plot(plotScore,plotMeanScore1)


    pass

if __name__ =='__main__':
    train()