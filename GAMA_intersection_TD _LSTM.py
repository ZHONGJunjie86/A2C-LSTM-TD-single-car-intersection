from utils import compute_returns, cross_loss_curve, GAMA_connect,reset,send_to_GAMA
import os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from shared_adam import SharedAdam
#from torch.distributions import Categorical from pandas import DataFrame
from torch.distributions import MultivariateNormal
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

state_size = 4 
action_size = 1 
lr = 0.0007
C_cx = torch.zeros(8).reshape(2,1,4).to(device)
from_python_1 = 'D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/python_AC_1.csv'
from_python_2 = 'D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/python_AC_2.csv'

class LSTMtrigger(nn.Module):
    def __init__(self,):
        super(LSTMtrigger,self).__init__()
        self.LSTM_layer = nn.LSTM(4,4,2)
    
    def forward(self,state,output = (C_cx,C_cx)):
        state , output = self.LSTM_layer(state,output)
        return state, output


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 64)
        self.linear2 = nn.Linear(64, 32)
        self.mu = nn.Linear(32,self.action_size)  #256 linear2
        self.sigma = nn.Linear(32,self.action_size)
        #self.lstm = LSTMtrigger()

    def forward(self, state):
        a,b,c = state.shape
        output_1 = F.relu(self.linear1(state.view(-1,c)))
        output_2 = F.relu(self.linear2(output_1))
        mu = 2 * torch.tanh(self.mu(output_2))   #有正有负 sigmoid 0-1
        sigma = F.relu(self.sigma(output_2)) + 0.001 
        mu = torch.diag_embed(mu).to(device)
        sigma = torch.diag_embed(sigma).to(device)  # change to 2D
        dist = MultivariateNormal(mu,sigma)  #N(μ，σ^2)
        entropy = dist.entropy().mean()
        action = dist.sample()
        action_logprob = dist.log_prob(action)     
        return action.detach(),action_logprob,entropy  #distribution .detach()

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 64)
        self.linear2 = nn.Linear(64,32) #
        self.linear3 = nn.Linear(32, 1)

    def forward(self, state):
        a,b,c = state.shape
        output_1 = F.relu(self.linear1(state.view(-1,c)))
        output_2 = F.relu(self.linear2(output_1))
        value = torch.tanh(self.linear3(output_2)) #有正有负
        return value

def main():
    ################ load ###################
    if os.path.exists('D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/weight/lstm.pkl'):
        lstm =  LSTMtrigger().to(device)
        lstm.load_state_dict(torch.load('D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/weight/lstm.pkl'))
        print('LSTM Model loaded')
    else:
        lstm =  LSTMtrigger().to(device)
    if os.path.exists('D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/weight/actor.pkl'):
        actor =  Actor(state_size, action_size).to(device)
        actor.load_state_dict(torch.load('D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/weight/actor.pkl'))
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)
    if os.path.exists('D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/weight/critic.pkl'):
        critic = Critic(state_size, action_size).to(device)
        critic.load_state_dict(torch.load('D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/weight/critic.pkl'))
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)
    print("Waiting for GAMA...")
    ################### initialization ########################
    reset()

    optimizerA = optim.Adam(actor.parameters(), lr, betas=(0.95, 0.999))
    optimizerC = optim.Adam(critic.parameters(), lr, betas=(0.95, 0.999))
    optimizerT = optim.Adam(lstm.parameters(), lr, betas=(0.95, 0.999))

    episode = 0
    test = "GAMA"
    state,reward,done,time_pass,over = GAMA_connect(test) #connect
    print("done:",done,"timepass:",time_pass)
    log_probs = [] #log probability
    values = []
    rewards = []
    masks = []
    total_loss = []
    total_rewards = []
    entropy = 0
    loss = []
    value = 0
    log_prob = 0
    C_cx = torch.zeros(8).reshape(2,1,4).to(device)
    lstm_input = 0
    lstm_state = 0

    ##################  start  #########################
    while over!= 1:
        #普通の場合
        if(done == 0 and time_pass != 0):  
            #前回の報酬
            reward = torch.tensor([reward], dtype=torch.float, device=device)
            state_next = np.reshape(state,(1,len(state)))
            state_next,lstm_input_next = lstm(torch.FloatTensor(state_next).unsqueeze(0).to(device),lstm_input) #2d - 3d
            lstm_input_next[0].detach()
            rewards.append(reward)   
            with torch.autograd.set_detect_anomaly(True):
                # TD:r(s) + gamma * v(s+1) - v(s)
                value_next = critic(state_next.detach())
                C_cx[0].detach()
                advantage = reward.detach() + value_next- value #values[len(values)-1].detach()
                actor_loss = -(log_prob * advantage.detach())     
                critic_loss = (reward.detach() + value_next - value).pow(2)
                lstm_loss = critic_loss
                optimizerA.zero_grad()
                optimizerC.zero_grad()
                optimizerT.zero_grad()
                lstm_loss.backward(retain_graph=True) 
                lstm_state.detach()
                critic_loss.backward() 
                actor_loss.backward()
                loss.append(critic_loss)
                optimizerT.step()
                optimizerA.step()
                optimizerC.step()

            state = np.reshape(state,(1,len(state)))
            state,lstm_input = lstm(torch.FloatTensor(state).unsqueeze(0).to(device),lstm_input) #2d - 3d
            value =  critic(state.detach())  #dist,  # now is a tensoraction = dist.sample() 
            action,log_prob,entropy= actor(state.detach())
            log_prob = log_prob.unsqueeze(0) #log_prob = dist.log_prob(action).unsqueeze(0)                  
            entropy += entropy

            #print("acceleration: ",action.cpu().numpy())#,"action.cpu().numpy()",type(float(action.cpu().numpy()))
            send_to_GAMA([[1,float(action.cpu().numpy()*10)]])#行
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))   #over-0; otherwise-1 contains the last
            values.append(value)
            log_probs.append(log_prob)

        # 終わり 
        elif done == 1:
            #print("restart acceleration: 0")
            send_to_GAMA([[1,0]])
            #先传后计算
            rewards.append(reward) #contains the last
            reward = torch.tensor([reward], dtype=torch.float, device=device)
            rewards.append(reward) #contains the last
            total_reward = sum(rewards)
            total_rewards.append(total_reward)

            state = np.reshape(state,(1,len(state)))
            last_state,_ = lstm(torch.FloatTensor(state).unsqueeze(0).to(device),lstm_input)
            last_value= critic(last_state.detach())

            with torch.autograd.set_detect_anomaly(True):
                advantage = reward.detach() + last_value - value
                actor_loss = -( log_prob * advantage.detach())    
                print("actor_loss, ",actor_loss ," size",actor_loss.dim())
                critic_loss = (reward.detach() + last_value - value).pow(2) 
                lstm_loss = critic_loss

                optimizerA.zero_grad()
                optimizerC.zero_grad()
                optimizerT.zero_grad()

                lstm_loss.backward(retain_graph=True)
                last_state.detach()
                critic_loss.backward() 
                actor_loss.backward()
                loss.append(critic_loss)
                
                optimizerA.step()
                optimizerC.step()
                optimizerT.step()


            print("----------------------------------Net_Trained---------------------------------------")
            print('--------------------------Iteration:',episode,'over--------------------------------')
            episode += 1
            log_probs = []
            values = []
            rewards = []
            masks = []
            torch.save(actor.state_dict(), 'D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/weight/actor.pkl')
            torch.save(critic.state_dict(), 'D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/weight/critic.pkl')
            torch.save(lstm.state_dict(), 'D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/weight/lstm.pkl')
            loss_sum = sum(loss)
            total_loss.append(loss_sum)
            #print("total_loss: ",total_loss)
            cross_loss_curve(total_loss,total_rewards)
            loss = []
            if episode > 50 : #50
                new_lr = lr * (0.94 ** ((episode-40) // 10)) #40
                optimizerA = optim.Adam(actor.parameters(), new_lr, betas=(0.95, 0.999))
                optimizerC = optim.Adam(critic.parameters(), new_lr, betas=(0.95, 0.999))
                optimizerT = optim.Adam(critic.parameters(), new_lr, betas=(0.95, 0.999))

        #最初の時
        else:
            print('Iteration:',episode)
            state = np.reshape(state,(1,len(state))) #xxx
            print("state: ",state)
            lstm_state,lstm_input = lstm(torch.FloatTensor(state).unsqueeze(0).to(device)) #默认参数
            value =  critic(lstm_state.detach())  #dist,  # now is a tensoraction = dist.sample() 
            action,log_prob,entropy = actor(lstm_state.detach())
            print("acceleration: ",action.cpu().numpy())
            send_to_GAMA([[1,float(action.cpu().numpy()*10)]])
            log_prob = log_prob.unsqueeze(0)
            print(log_prob)
            entropy += entropy

        state,reward,done,time_pass,over = GAMA_connect(test)
    return None 

if __name__ == '__main__':
    main()
