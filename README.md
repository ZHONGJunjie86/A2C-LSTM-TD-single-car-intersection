# A2C-LSTM-TD-single-car-intersection
　A model used to identify the usefulness of LSTM with sequential data.

　When I noticed neither the A2C-TD nor the A2C-MC can learn sequent data very well, I turned to add the LSTM (a Recurrent Neural Network, RNN）into A2C_TD algorithm.
　I desire the LSTM can make the algorithm know the relation of the data. 
　Go to see Actor-critic-TD-model as a contrast from [my previous work](https://github.com/ZHONGJunjie86/A2C-TD-single-car-intersection).    
     　　
   ![image](https://github.com/ZHONGJunjie86/A2C-TD-single-car-intersection/blob/master/illustrate/illustration_1.gif )   
# Structure
 ![image](https://github.com/ZHONGJunjie86/A2C-LSTM-TD-single-car-intersection/blob/master/result/structure.png)
# Sequential data
　Input [real_speed/10, target_speed/10, elapsed_time_ratio, distance_to_goal/100,reward,done,time_pass,over]  
　Station representation: [real_speed/10, target_speed/10, elapsed_time_ratio, distance_to_goal/100]  
 　It's notable that the data elements have some relation rather than random distribute.   
 　The　target_speed is a constant value and the  elapsed_time_ratio and distance_to_goal are monotonically increasing or monotonically decreasing data.
  # Reward shaping
　Output accelerate.
　Action representation [acceleration].
  
　The car will learn to control its accelerate with the restructions shown below:  
　Reward shaping:  
* rt = r terminal + r danger + r speed  
* r terminal： -0.013(target_speed > real_speed) or  -0.01(target_speed < real_speed)：crash / time expires 
                 0.005:non-terminal state  
* r speed： related to the target speed  
* if sa ≤st: 0.05 - 0.036*(target_speed / real_speed) 
* if sa > st: 0.05 - 0.033*(real_speed / target_speed ) 

　In my experiment it's obviously I desire the agent to learn controling its speed around the target-speed.   
 # Result
　It's obvoiusly that the LSTM can be trained much faster than models without LSTM. 
## Actor-Ctitic + LSTM (acceleration > 0 OR acceleration < 0; tanh())
 ![image](https://github.com/ZHONGJunjie86/A2C-LSTM-TD-single-car-intersection/blob/master/result/loss_curve_TD_LSTM_lr000700020001.png)
## (acceleration > 0 OR acceleration < 0; tanh())
![image](https://github.com/ZHONGJunjie86/A2C-TD-single-car-intersection/blob/master/illustrate/loss_curve_TD_tanh.png)
## Only speed up (acceleration > 0; sigmoid())
![image](https://github.com/ZHONGJunjie86/A2C-TD-single-car-intersection/blob/master/illustrate/loss_curve_TD_21.png)

  # Experiment
  ###### gama:
           time_target <- int((distance_left/100)*5)+ rnd(3); 
           target_speed<- distance_left/time_target;
           random_node <- int(rnd(12));
           target<- node_agt(random_node);
           true_target <- node_agt(random_node);
           final_target <- node_agt(random_node).location;	
           location <- any_location_in(node_agt(5)); 
　There are 12 nodes in the intersection map and the start point is fixed at the 5th point. Every time before a cycle there will be a random number between 0 and 12 used to choose destination node. And the target-time and target speed will also be changed.   
#### In other words, I let the agent to learn 3*11=33 situations.  　The rewards depend on the situation, so it will change every cycle.  
#### The model will be trained every step(TD). 

　<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangledown&space;R&space;=&space;\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(r_{t}&plus;V_{s&plus;1}^{n}-V_{s}^{n})\bigtriangledown&space;log&space;P_{\Theta&space;}(a_{t}^{n}|s_{t}^{n})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangledown&space;R&space;=&space;\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(r_{t}&plus;V_{s&plus;1}^{n}-V_{s}^{n})\bigtriangledown&space;log&space;P_{\Theta&space;}(a_{t}^{n}|s_{t}^{n})" title="\bigtriangledown R = \frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T}(r_{t}+V_{s+1}^{n}-V_{s}^{n})\bigtriangledown log P_{\Theta }(a_{t}^{n}|s_{t}^{n})" /></a>
 ## Learning rate weakened
            lr = 0.0007  
            if episode > 50 : 
                if episode > 135:
                    lr = 0.0002
                new_lr = lr * (0.94 ** ((episode-40) // 10)) 
                optimizerA = optim.Adam(actor.parameters(), new_lr, betas=(0.95, 0.999))
                optimizerC = optim.Adam(critic.parameters(), new_lr, betas=(0.95, 0.999))

# Hyperparameter optimization
　Station representation: [real_speed, target_speed, elapsed_time_ratio, distance_to_goal]  
　Action representation [accelerate].
## representations' values are very small
### tanh funtion --acceleration can be positive or negative values
### sigmoid funtion --acceleration can only be positive values (learning faster)
　I use tanh funtion as the actor/critic network's output layer which can output positive or negative values, while  I use sigmoid funtion as the actor/critic network's output layer which can only output positive values(speed up only).
　To prevent vanishing gradient problem the value used for backpropagate should be close to 0.
## The agent was trying to reach a very fast speed to reduce steps and thus penalties
* r speed： related to the target speed  
* if sa ≤st: 0.05 - 0.033*(target_speed / real_speed) 
* if sa > st: 0.05 - 0.036*(real_speed / target_speed )   
　So the faster the speed, the greater the penalty. The same goes for the very low speed. 
　The value of reward-speed can be positive or negative.
　Over or under speeding is be balanced.
## Over time penalty
　-0.013(target_speed > real_speed) or  -0.01(target_speed < real_speed)：crash / time expires 
　The formost issue is acceleration, so the value of over time penalty should much smaller than the value of reward-speed.
　The reason 0.013>0.01 is that lower speeds lead to  over time more likely.
## Input's and output's values should not be too different 
　In fact the values sent from the GAMA side is [real_speed/10, target_speed/10, elapsed_time_ratio, distance_left/100,reward,done,over].
　SO the station representation is [real_speed/10, target_speed/10, elapsed_time_ratio, distance_left/100].
　And the action representation is [accelerate*10].
　In this way, the loss will not violently oscillate and the image of learning curve will be more cognizable.


# About GAMA
　The GAMA is a platefrom to do simulations.      
　I have a GAMA-modle "simple_intersection.gaml", which is assigned a car and some traffic lights. The model will sent some data  
　[real_speed, target_speed, elapsed_time_ratio, distance_to_goal,reward,done,time_pass,over]  
　as a matrix to python environment, calculating the car's accelerate by A2C. And applying to the Markov Decision Process framework, the car in the GAMA will take up the accelerate and send the latest data to python again and over again until  reaching the destination.
# Architecture
　The interaction between the GAMA platform and python environment is built on csv files I/O. So GAMA model needs to use R-plugin and the R environment needs package "reticulate" to connect with python (I use python more usually).
 
  A2C-architecture
  --------------
  ![image](https://github.com/ZHONGJunjie86/A3C-single-car-intersection/blob/master/illustrate/A2C-Architecture.JPG) 
  A3C-architecture
  ------------
  ![image](https://github.com/ZHONGJunjie86/A3C-single-car-intersection/blob/master/illustrate/A3C-Architecture.JPG) 
 # Reference
 [Synchronous A3C (A2C) ](https://arxiv.org/abs/1602.01783).    
 
# 2020.07.11
It semms some trouble with this model because it wasn't convergent.

# 2020.07.12
Got final result.
