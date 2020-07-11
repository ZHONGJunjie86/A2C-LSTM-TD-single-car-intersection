import matplotlib.pyplot as plt
import numpy as np
import time,random
import os 

from_GAMA_1 = 'D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/GAMA_intersection_data_1.csv'
from_GAMA_2 = 'D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/GAMA_intersection_data_2.csv'
from_python_1 = 'D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/python_AC_1.csv'
from_python_2 = 'D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/python_AC_2.csv'
save_curve_pic = 'D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/result/loss_curve.png'

def reset():
    f=open(from_GAMA_1, "r+")
    f.truncate()
    f=open(from_GAMA_2, "r+")
    f.truncate()
    f=open(from_python_1, "r+")
    f.truncate()
    f=open(from_python_2, "r+")
    f.truncate()
    return_ = [0]
    np.savetxt(from_python_1,return_,delimiter=',')
    np.savetxt(from_python_2,return_,delimiter=',')

def compute_returns(last_value, rewards, masks, gamma=0.99):
    R = last_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step] #masks 最后是0
        returns.insert(0, R) #index = 0
    return returns

def cross_loss_curve(entropys,total_rewards):
    plt.plot(np.array(entropys), c='b', label='critic_loss') #cross_entropy
    plt.plot(np.array(total_rewards), c='r', label='total_rewards')
    plt.legend(loc='best')
    plt.ylim(-3,3)#(-6,15)
    plt.ylabel('critic_loss') #cross_entroy
    plt.xlabel('training steps')
    plt.grid()
    #plt.show()
    plt.savefig(save_curve_pic)
    plt.close()

def send_to_GAMA(to_GAMA):
    error = True
    while error == True:
        try:
            np.savetxt(from_python_1,to_GAMA,delimiter=',')
            np.savetxt(from_python_2,to_GAMA,delimiter=',')
            error = False
        except(IndexError,FileNotFoundError,ValueError,OSError,PermissionError):  
            error = True 
            
#[real_speed, target_speed, elapsed_time_ratio, distance_left,reward,done,over]
def GAMA_connect(test):
    error = True
    while error == True:
        try:
            time.sleep(0.003)
            if(random.random()>0.3):
                state = np.loadtxt(from_GAMA_1, delimiter=",")
            else:
                state = np.loadtxt(from_GAMA_2, delimiter=",")
            time_pass = state[2]
            error = False
        except (IndexError,FileNotFoundError,ValueError,OSError):
            time.sleep(0.003)
            error = True
    reward = state[4]
    done = state[5]  # time_pass = state[6]
    over = state [6] 
    print("Recived:",state," done:",done)
    state = np.delete(state, [4,5,6], axis = 0)
    error = True
    while error == True:
        try:
            f1=open(from_GAMA_1, "r+")
            f1.truncate()
            f2=open(from_GAMA_2, "r+")
            f2.truncate()
            error = False
        except (IndexError,FileNotFoundError,ValueError,OSError):
            time.sleep(0.003)
            error = True

    return state,reward,done,time_pass,over,