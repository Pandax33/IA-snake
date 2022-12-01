import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_Qnet(nn.Module):
    def __init__(self,inputSize,hiddenSize,outputSize):
        super().__init__()
        self.linear1=nn.Linear(inputSize,hiddenSize)
        self.linear2=nn.Linear(hiddenSize,outputSize)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x= self.linear2(x)
        return x

    def save(self,fileName='model.pth'):
        modelFolder='./model'
        if not os.path.exists(modelFolder):
            os.makedirs(modelFolder)
        fileName=os.path.join(modelFolder,fileName)
        torch.save(self.state_dict(),fileName)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr=lr
        self.gamma=gamma
        self.model=model
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        self.criterion= nn.MSELoss()

    def trainStep(self,state,move,reward,nextState,done):
        state=torch.tensor(state,dtype=torch.float)
        nextState=torch.tensor(nextState,dtype=torch.float)
        move=torch.tensor(move ,dtype=torch.long)
        reward=torch.tensor(reward,dtype=torch.float)
        if len(state.shape)==1:
            state=torch.unsqueeze(state,0)
            nextState=torch.unsqueeze(nextState,0)
            move=torch.unsqueeze(move,0)
            reward=torch.unsqueeze(reward,0)
            done= (done,)
        pred=self.model(state)
        target= pred.clone()
        for idx in range(len(done)):
            Q_new=reward[idx]
            if not done[idx]:
                Q_new=reward[idx]+self.gamma * torch.max(self.model(nextState[idx]))
            target[idx][torch.argmax(move[idx]).item()] = Q_new
        self.optimizer.zero_grad()
        loss=self.criterion(target,pred)
        loss.backward()
        self.optimizer.step()




