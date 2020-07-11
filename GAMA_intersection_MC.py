from utils import compute_returns, cross_loss_curve, GAMA_connect,reset
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
lr = 0.002
from_python_1 = 'D:/Software/GamaWorkspace/Python/python_AC_1.csv'
from_python_2 = 'D:/Software/GamaWorkspace/Python/python_AC_2.csv'

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 64)
        self.mu = nn.Linear(64,self.action_size)  #256 linear2
        self.sigma = nn.Linear(64,self.action_size)
        #self.linear3 = nn.Linear(256, self.action_size)
        #self.action_var = torch.full((self.action_size,), action_std*action_std).to(device)

    def forward(self, state):
        output_1 = F.relu(self.linear1(state))
        output_2 = F.relu(self.linear2(output_1))
        mu = 2 * torch.sigmoid(self.mu(output_2))   #有正有负
        sigma = F.relu(self.sigma(output_2)) + 0.001   # avoid 0 softplus    output = F.softmax(output, dim=-1)         action_mean = self.linear3(output)
        #cov_mat = torch.diag(self.action_var).to(device)
        mu = torch.diag_embed(mu).to(device)
        sigma = torch.diag_embed(sigma).to(device)  # change to 2D
        dist = MultivariateNormal(mu,sigma)  #N(μ，σ^2)  σ超参不用训练 MultivariateNormal(action_mean, cov_mat) 
        #distribution = Categorical(F.softmax(output, dim=-1))
        entropy = dist.entropy().mean()
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(),action_logprob,entropy   #distribution .detach()

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128,64) #
        self.linear3 = nn.Linear(64, 1)

    def forward(self, state):
        output_1 = F.relu(self.linear1(state))
        output_2 = F.relu(self.linear2(output_1))
        value = torch.sigmoid(self.linear3(output_2)) #有正有负
        return value


def main():
    ################ load ###################
    if os.path.exists('D:/Software/GamaWorkspace/Python/weight/actor.pkl'):
        actor =  Actor(state_size, action_size).to(device)
        actor.load_state_dict(torch.load('D:/Software/GamaWorkspace/Python/weight/actor.pkl'))
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)
    if os.path.exists('D:/Software/GamaWorkspace/Python/weight/critic.pkl'):
        critic = Critic(state_size, action_size).to(device)
        critic.load_state_dict(torch.load('D:/Software/GamaWorkspace/Python/weight/critic.pkl'))
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)
    print("Waiting for GAMA...")
    ################### initialization ########################
    reset()

    optimizerA = optim.Adam(actor.parameters(), lr, betas=(0.95, 0.999))#optim.Adam(actor.parameters())  
    optimizerC = optim.Adam(critic.parameters(), lr, betas=(0.95, 0.999))#optim.Adam(critic.parameters())

    episode = 0
    test = "GAMA"
    state,reward,done,time_pass,over = GAMA_connect(test)
    print("done:",done,"timepass:",time_pass)
    log_probs = [] #log probability
    values = []
    rewards = []
    masks = []
    entropys = []
    total_rewards = []
    total_loss = []
    loss = 0
    entropy = 0

    ##################  start  #########################
    while over!= 1:
        #普通の場合
        if(done == 0 and time_pass != 0):  
            state = torch.FloatTensor(state).to(device)
            value =  critic(state)  #dist,  # now is a tensoraction = dist.sample() 
            action,log_prob,entropy = actor(state) #action = dist.sample()  # now is a tensor
            #pass parameter
            print("acceleration: ",action.cpu().numpy())#,"action.cpu().numpy()",type(float(action.cpu().numpy()))
            to_GAMA = [[1,float(action.cpu().numpy()*10)]] #行
            np.savetxt(from_python_1,to_GAMA,delimiter=',')
            np.savetxt(from_python_2,to_GAMA,delimiter=',')
            #前回の報酬
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device)) #contains the last
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))   #over-0; otherwise-1 contains the last
            log_prob = log_prob.unsqueeze(0) #log_prob = dist.log_prob(action).unsqueeze(0)       # entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            entropy += entropy
        # 終わり 
        elif done == 1:
            print("restart acceleration: 0")
            to_GAMA = [[1,0]]
            np.savetxt(from_python_1,to_GAMA,delimiter=',')
            np.savetxt(from_python_2,to_GAMA,delimiter=',')
            #先传后计算
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device)) #contains the last
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))   #over-0; otherwise-1 contains the last
            
            total_reward = sum(rewards)
            total_rewards.append(total_reward)

            last_state = torch.FloatTensor(state).to(device)
            last_value = critic(last_state)
            returns = compute_returns(last_value, rewards, masks) 
            values_next = returns[1:]#values[1:]
            values_next.append(torch.tensor([0], dtype=torch.float, device=device))
            
            log_probs = torch.cat(log_probs,1).squeeze()  #Concatenates the given sequence of seq tensors in the given dimension.
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            values_next = torch.cat(values_next)
            rewards = torch.cat(rewards)

            # TD:r(s) + v(s+1) - v(s)      #rewards.detach() + values_next - values  r(s) MC: returns.detach() - values???
            advantage = returns.detach() - values
            actor_loss = -(log_probs * advantage.detach()).mean()
            loss = advantage.pow(2).sum()
            loss.detach()
            critic_loss = (returns.detach() - values).pow(2).mean()

            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizerA.step()
            optimizerC.step()

            print("--------------------------Net_Trained-------------------------------")
            print('--------------------------Iteration:',episode,'over--------------------------------')
            episode += 1
            log_probs = []
            values = []
            rewards = []
            masks = []
            torch.save(actor.state_dict(), 'D:/Software/GamaWorkspace/Python/weight/actor.pkl')
            torch.save(critic.state_dict(), 'D:/Software/GamaWorkspace/Python/weight/critic.pkl')
            #print("entropy: ",entropy,"total_rewards:",total_rewards)
            entropys.append(entropy)
            total_loss.append(loss)
            if(episode!=0):
                cross_loss_curve(total_loss,total_rewards)
            loss = 0

            if episode > 90  :
                new_lr = lr * (0.92 ** ((episode-80) // 10))
                optimizerA = optim.Adam(actor.parameters(), new_lr, betas=(0.95, 0.999))
                optimizerC = optim.Adam(critic.parameters(), new_lr, betas=(0.95, 0.999))

        #最初の時
        else:
            print('Iteration:',episode)
            state = torch.FloatTensor(state).to(device)
            value =  critic(state)  #dist,  # now is a tensoraction = dist.sample() 
            action,log_prob,entropy = actor(state)
            print("acceleration: ",float(action.cpu().numpy()*10))
            to_GAMA = [[1,action.cpu().numpy()]]
            np.savetxt(from_python_1,to_GAMA,delimiter=',')
            np.savetxt(from_python_1,to_GAMA,delimiter=',')
            log_prob = log_prob.unsqueeze(0) #log_prob = dist.log_prob(action).unsqueeze(0) #entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            entropy += entropy

        state,reward,done,time_pass,over = GAMA_connect(test)
    return None #[action,log_prob_return,value]

if __name__ == '__main__':
    main()
