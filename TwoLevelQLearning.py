#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 15 15:48:13 2022

@author: willnehrboss
"""

########
# The action space is defined by a size * size matrix. Each agent maintains a q-matrix of the same size
# as the action space index, with each entry containing a three elment vector of the action values for 
# 'up', 'down' , 'flat'.  A given agent's own bid is represented by the row index; the column index is 
# the opponents bid in the last round. Each agent maintains a second level of Q-values for the learning
# of the stability factor, theta. 
#########

import random
import copy
import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

#defining action space
num_auctions = 1000
size = 6
actions = [ 'up', 'down' , 'flat' ]
iters= 80000


#defining agent charectaristics 
class agent:

    def __init__(self):
        self.stab_size = 2
        self.stab_q_values = np.zeros((self.stab_size, self.stab_size, 3)) 
        self.stab_column = 0 
        self.stab_rewards = 0
        self.old_stab_column =0
        self.stability_factor = 0
        self.stability_list = [0] * iters
        self.end_bids = [0] * num_auctions
        self.environment_rows = 21
        self.environment_columns = 21
        self.q_values = np.zeros((size, size, 3))  
        self.current_row_index = 0  
        self.current_column_index = 0
        self.bids  = [0] * iters
        self.profit  = [0] * iters
        self.tiebreak_condition = 3
        self.stability = 1
        self.end_rewards = [0]*num_auctions
        self.rewards = np.zeros((size, size))
        self.environment  = ''
        self.shock = 0
        self.stability_factor_list = [0] * iters
        self.win = False
        #creating reward matrix for FPA
        for i in range(0, size):
            for j in range(0,size):
                if i >= j:
                    self.rewards[i][j] = size - 1  +.1- i 
                
                else:
                    self.rewards[i][j] = 0 


#getting starting bid (index or q-matrix)       
        
    def get_starting_location(self):
      self.current_row_index = 0
      return self.current_row_index, self.current_column_index
#Return next action according to E greedy policy. 
def get_next_action(q_values, current_row_index, current_column_index, epsilon):

  if np.random.random() > epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
    return np.random.randint(3)
#get next bid (index in Q-matrix)
def get_next_location(current_row_index, current_column_index, action_index, op_bidder_bid):
  new_row_index = current_row_index
  new_column_index = op_bidder_bid
  if actions[action_index] == 'down' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'up' and current_row_index < size -1:
    new_row_index += 1
  elif actions[action_index] == 'flat' :
    new_row_index = new_row_index

      
  return new_row_index, new_column_index
# Similar function, moving to new Q-matrix row and column index for second level Q learning
def get_next_stab_factor(current_stab, current_op_stab, action_index):
  new_stab = current_stab
  
  new_op_stab = current_op_stab
  if actions[action_index] == 'down' and current_stab > 0:
    new_stab -= 1
  elif actions[action_index] == 'up' and current_stab < 1:
    new_stab += 1
  elif actions[action_index] == 'flat' :
    new_stab = new_stab

      
  return new_stab, new_op_stab

#environment parameters
epsilon = 0.99 
discount_factor = 0.9
learning_rate = 0.4


bidder1  =  agent()
bidder2  =  agent()
bidder1.tiebreak_condition = 0
bidder2.tiebreak_condition = 0
bidder1.row_index, bidder1.column_index = bidder1.get_starting_location()
bidder2.row_index, bidder2.column_index = bidder2.get_starting_location()
bs1list = []
bs2list =[]
#noise parameter
gamma =.05
shocks1 = []
shocks2 =[]
expec_bidder1_prof = []
expec_bidder2_prof = []
all_auctions = pd.DataFrame(index=range(iters))
all_profit   = pd.DataFrame(index=range(iters))
all_shocks   = pd.DataFrame(index=range(iters))
all_stability= pd.DataFrame(index=range(iters))
all_stability_factor= pd.DataFrame(index=range(iters))

# The function auction() runs for iters number of rounds. This represents a single sub-experiment. 
def auction(num_runs):
    expec_bidder1_prof = []
    expec_bidder2_prof = []

    bs1list = []
    bs2list =[]
    iters = num_runs
    bidder1.q_values = np.zeros((size, size, 3)) 
    bidder2.q_values = np.zeros((size, size, 3)) 
    bidder1.row_index, bidder1.column_index = bidder1.get_starting_location()
    bidder2.row_index, bidder2.column_index = bidder2.get_starting_location()

    environment = 'FPA'
    bidder1.environment = environment
    print(environment)
    bs1list = []
    bs2list =[]
    gamma =0.05
    shocks1 = []
    shocks2 =[]
    eps= [0] * iters
    
    for episode in range(iters):
           
            #each agent has a 'shock' value which changes each agent's valuation of the good between auctions. 
            # sum of the prior shock value plus a random values weighted by the noise parameter gamma
            bidder1.shock  =   gamma * np.random.normal(0,1,1)[0] +  gamma * bidder1.shock 
            bidder2.shock  =   gamma * np.random.normal(0,1,1)[0] +  gamma * bidder2.shock
            shocks1.append(bidder1.shock)
            shocks2.append(bidder2.shock)
            
           # bidder1.stability_factor =0
           # bidder2.stability_factor =0
            #E decays to zero
            epsilon =    np.power(np.e, -(0.0001* episode) ) 
    
    
            eps[episode] = epsilon
            
            #determines the next action to take based on the q values at a the current state and the realization of the E-greedy function
            bidder2.action_index = get_next_action(bidder2.q_values, bidder2.row_index, bidder2.column_index, epsilon)
            bidder1.action_index = get_next_action(bidder1.q_values, bidder1.row_index, bidder1.column_index, epsilon)

            bidder1.old_row_index, bidder1.old_column_index = bidder1.row_index, bidder1.column_index #store the old row and column indexes
    
            bidder2.old_row_index, bidder2.old_column_index = bidder2.row_index, bidder2.column_index #store the old row and column indexes
            #updates row and column index based on the chosen function
            bidder2.row_index, bidder2.column_index = get_next_location(bidder2.row_index, bidder2.column_index, bidder2.action_index, bidder1.old_row_index)
            bidder1.row_index, bidder1.column_index = get_next_location(bidder1.row_index, bidder1.column_index, bidder1.action_index, bidder2.old_row_index)
            bidder1.effective_bid =  bidder1.row_index # + random.randrange(-1,2)/1000
            
            bidder1.reward= 0
            bidder2.reward= 0
            #simulates tiebreaks in a round of fifty repeated auctions.
            bidder1.tiebreaks = sum(bernoulli.rvs(.5, size=50))/50
            bidder2.tiebreaks = 1 - bidder1.tiebreaks
            #determines the outcome of the round and associated rewards
            if environment == 'SPA':
    

                
                if bidder1.effective_bid > bidder2.row_index:
                     bidder1.shock  =   gamma * np.random.normal(0,1,1)[0] +  gamma * bidder1.shock                  
                     bidder1.reward = size-1 - bidder2.row_index   +.1 + bidder1.shock
                     expec_bidder1_prof.append(size +bidder2.shock -1 - bidder2.row_index   +.1 )
                     expec_bidder2_prof.append(0)
                elif  bidder2.row_index > bidder1.effective_bid:
                     bidder2.shock  =   gamma * np.random.normal(0,1,1)[0] +  gamma * bidder2.shock
                     bidder2.reward = (size + bidder2.shock -1 - bidder1.row_index   +.1)
                     expec_bidder2_prof.append(size  + bidder1.shock-1 - bidder1.row_index   +.1 )
                     expec_bidder1_prof.append(0)

                else:
                    bidder2.shock  =   gamma * np.random.normal(0,1,1)[0] +  gamma * bidder2.shock
                    bidder1.shock  =   gamma * np.random.normal(0,1,1)[0] +  gamma * bidder1.shock                  
                    bidder1.reward = (size +bidder1.shock- 1 - bidder1.row_index  +.1)  * bidder1.tiebreaks
                    bidder2.reward = (size +bidder2.shock-1 - bidder2.row_index   +.1)  * bidder2.tiebreaks
                    expec_bidder2_prof.append((size+bidder1.shock- 1 - bidder1.row_index  +.1)  * bidder2.tiebreaks) 
                    expec_bidder1_prof.append( (size+bidder2.shock-1 - bidder2.row_index   +.1)  * bidder1.tiebreaks)
            else:
                if bidder1.effective_bid > bidder2.row_index:
                     bidder1.shock  =   gamma * np.random.normal(0,1,1)[0] +  gamma * bidder1.shock                 
                     bidder1.reward = bidder1.rewards[bidder1.row_index,bidder2.row_index] 
                     expec_bidder1_prof.append(bidder1.reward +bidder2.shock)
                     bidder1.reward= bidder1.reward + bidder1.shock
                     expec_bidder2_prof.append(0)
                    
                elif bidder2.row_index > bidder1.effective_bid:
                     bidder2.shock  =   gamma * np.random.normal(0,1,1)[0] +  gamma * bidder2.shock                 
                     expec_bidder2_prof.append(bidder2.rewards[bidder2.row_index,bidder1.row_index]  +bidder1.shock)
                     bidder2.reward= bidder1.rewards[bidder2.row_index,bidder1.row_index]  + bidder2.shock
                     expec_bidder1_prof.append(0)
                else:
                    bidder2.shock  =   gamma * np.random.normal(0,1,1)[0] +  gamma * bidder2.shock
                    bidder1.shock  =   gamma * np.random.normal(0,1,1)[0] +  gamma * bidder1.shock                  
                    bidder1.reward = ( bidder1.rewards[bidder1.row_index,bidder2.row_index]+ bidder1.shock) *bidder1.tiebreaks
                    bidder2.reward = ((bidder2.rewards[bidder2.row_index,bidder1.row_index]) + bidder2.shock) *bidder2.tiebreaks
                    expec_bidder1_prof.append(( bidder1.rewards[bidder1.row_index,bidder2.row_index]+ bidder2.shock)*bidder2.tiebreaks ) 
                    expec_bidder2_prof.append( ( bidder2.rewards[bidder2.row_index,bidder1.row_index]+ bidder1.shock)*bidder1.tiebreaks  )

                 
        
            bidder1.profit[episode] = bidder1.reward
    
            bidder2.profit[episode] = bidder2.reward
        
            bs1 = 1
            bs2 = 1
            
            #determines G(theta)
            if episode >10:
                bidder1.stability =   min(1, ((np.mean(expec_bidder2_prof[episode-3:episode +1]) + .1)/(np.mean(bidder1.profit[episode-9:episode + 1] ) + .1)))
                bidder2.stability =   min(1, ((np.mean(expec_bidder1_prof[episode-3:episode +1]) + .1)/(np.mean(bidder2.profit[episode-9:episode + 1] ) + .1)))
                bs1 =  ((1 + bidder1.stability * 1* bidder1.stability_factor)/(1 + 1*bidder1.stability_factor))
                bs2 =  ((1 + bidder2.stability * 1* bidder2.stability_factor)/(1 + 1*bidder2.stability_factor))
            bs1list.append(bs1)
            bs2list.append(bs2)
            
            
            
            #updates the Q values based on the outcome of the auction, G(theta), learning rate, and discount factor
            bidder1.old_q_value = bidder1.q_values[bidder1.old_row_index, bidder1.old_column_index, bidder1.action_index]
            bidder1.temporal_difference = bidder1.reward  + (discount_factor *bs1 * np.max(bidder1.q_values[bidder1.row_index, bidder1.column_index])  )  -  bidder1.old_q_value
            bidder1.new_q_value = ((1 ) * (bidder1.old_q_value) + (learning_rate * bidder1.temporal_difference  ))  
            bidder1.q_values[bidder1.old_row_index, bidder1.old_column_index, bidder1.action_index] = bidder1.new_q_value  # * bs1
            bidder1.bids[episode] = bidder1.row_index
            
        
            
            bidder2.old_q_value = bidder2.q_values[bidder2.old_row_index, bidder2.old_column_index, bidder2.action_index]
            bidder2.temporal_difference = bidder2.reward  + (discount_factor *bs2  * np.max(bidder2.q_values[bidder2.row_index, bidder2.column_index]) )  - bidder2.old_q_value
            bidder2.new_q_value =  ((1) * (bidder2.old_q_value) + (learning_rate * bidder2.temporal_difference     ) )  
            bidder2.q_values[bidder2.old_row_index, bidder2.old_column_index, bidder2.action_index] = bidder2.new_q_value # * bs2
            bidder2.bids[episode] = bidder2.row_index
            bidder1.stability_list[episode] = bs1
            bidder2.stability_list[episode] = bs2
#save and plot certain simulation values    
    wave = int(iters/500)      
                                              
    print('Training complete!')
    bids1 = bidder1.bids
    bids2  = bidder2.bids
    endbids1 = np.mean(bids1[-1000:])
    endbids2 = np.mean(bids2[-1000:])
    endprof1 = np.mean(bidder1.profit)
    endprof2 = np.mean(bidder2.profit)
    r1 = bidder1.q_values
    r2 = bidder2.q_values
    ser1 = pd.Series(bids1)
    prof1 = pd.Series(bidder1.profit)
    prof2 = pd.Series(bidder2.profit)
    p1 = prof1.rolling( wave).mean()
    p2 = prof2.rolling( (wave)).mean()
    b1 = ser1.rolling( (wave)).mean()
    ser2 = pd.Series(bids2)
    
    b2 = ser2.rolling( (wave)).mean()
    
    b1 = ser1.rolling( (wave)).mean()
    ser2 = pd.Series(bids2)
    
    b2 = ser2.rolling( (wave)).mean()
    ax = plt.subplots()
    
    plt.plot(range(iters), b1)
    plt.plot(range(iters), b2)
    decay = pd.Series(eps)
    
    plt.plot(range(iters), p1)
    plt.plot(range(iters), p2)
    plt.plot(range(iters), decay)
   



    plt.show()
    p2.plot()
    
    plt.close()
    
    plt.errorbar(range(iters), pd.Series(bidder1.stability_list).rolling( (wave)).mean())
    plt.errorbar(range(iters), pd.Series(bidder2.stability_list).rolling( (wave)).mean())
    plt.errorbar(range(iters), pd.Series(bidder1.stability_list).rolling( (wave)).mean())
    plt.errorbar(range(iters), pd.Series(bidder2.stability_list).rolling( (wave)).mean())
    plt.show()
    plt.close
    
    plt.plot(range(iters), pd.Series(shocks1).rolling( (wave)).mean())
    plt.plot(range(iters), pd.Series(shocks2).rolling( (wave)).mean())
    plt.show()
    plt.close 
    return endprof1, endprof2, endbids1, endbids2, shocks1, shocks2, expec_bidder1_prof, expec_bidder2_prof
stab_learning_rate = .5

#higher level of Q-learning. Exact same process of Q-learning. 
def stab_sim(runs):
    
    stab_ep = .99

    for i in range(1, runs):
        print(i)

        stab_ep =   1* np.power(np.e, -(0.02* i) ) 
        stab_ep =   1* np.power(np.e, -(0.02* i) ) 

        bidder2.stab_action_index = get_next_action(bidder2.stab_q_values, bidder2.stability_factor, bidder2.stab_column, stab_ep)
        bidder1.stab_action_index = get_next_action(bidder1.stab_q_values, bidder1.stability_factor, bidder1.stab_column, stab_ep)

        bidder1.old_stability_factor, bidder1.old_stab_column = bidder1.stability_factor, bidder1.old_stab_column #store the old row and column indexes
  
        bidder2.old_stability_factor, bidder2.old_stab_column = bidder2.stability_factor, bidder2.old_stab_column #store the old row and column indexes
        
        bidder2.stability_factor, bidder2.stab_column = get_next_stab_factor(bidder2.stability_factor, bidder1.stability_factor, bidder2.stab_action_index)
        bidder1.stability_factor, bidder1.stab_column = get_next_stab_factor(bidder1.stability_factor, bidder2.stability_factor, bidder1.stab_action_index )
       
        bidder1.stab_rewards, bidder2.stab_rewards , bidder1.end_bids[i-1] ,bidder2.end_bids[i-1], shocks1, shocks2, expec_bidder1_prof, expec_bidder2_prof = auction(iters)

        print( bidder1.stab_rewards)
        print( bidder2.stab_rewards)
        print( bidder1.stability_factor)
        print( bidder1.stability_factor)

        bidder1.old_q_value = bidder1.stab_q_values[bidder1.old_stability_factor, bidder1.old_stab_column, bidder1.stab_action_index]
        bidder1.temporal_difference = bidder1.stab_rewards  + (discount_factor * np.max(bidder1.stab_q_values[bidder1.stability_factor, bidder1.stab_column])  )  #-  bidder1.old_q_value

        bidder1.new_q_value = (( 1- stab_learning_rate) * (bidder1.old_q_value) + (stab_learning_rate * bidder1.temporal_difference  ))  
        bidder1.stab_q_values[bidder1.old_stability_factor, bidder1.old_stab_column, bidder1.stab_action_index] = bidder1.new_q_value  
        
    
        
        bidder2.old_q_value = bidder2.stab_q_values[bidder2.old_stability_factor, bidder2.old_stab_column, bidder2.stab_action_index]
        bidder2.temporal_difference = bidder2.stab_rewards  + (discount_factor * np.max(bidder2.stab_q_values[bidder2.stability_factor, bidder2.stab_column]) )  #- bidder2.old_q_value
        bidder2.new_q_value =  ((1- stab_learning_rate) * (bidder2.old_q_value) + (stab_learning_rate * bidder2.temporal_difference    ) )  
        bidder2.stab_q_values[bidder2.old_stability_factor, bidder2.old_stab_column, bidder2.stab_action_index] = bidder2.new_q_value 


#saving auction data

        name1 = 'auction_' + str(i) + '_bid1'
        name2 = 'auction_' + str(i) + '_bid2'

        all_auctions[name1] = bidder1.bids
        all_auctions[name2] = bidder2.bids
        name1 = 'auction_' + str(i) + '_profit1'
        name2 = 'auction_' + str(i) + '_profit2'
        all_profit[name1] = bidder1.profit
        all_profit[name2] = bidder2.profit
        
        name1 = 'auction_' + str(i) + '_shock1'
        name2 = 'auction_' + str(i) + '_shock2'
        all_shocks[name1] = shocks1
        all_shocks[name2] = shocks2
        name1 = 'auction_' + str(i) + '_stability1'
        name2 = 'auction_' + str(i) + '_stability2'
        all_stability[name1] = bidder1.stability_list
        all_stability[name2] = bidder2.stability_list
        name1 = 'auction_' + str(i) + '_stability_factor1'
        name2 = 'auction_' + str(i) + '_stability_factor2'        
        all_stability_factor[name1] = bidder1.stability_factor_list
        all_stability_factor[name2] = bidder2.stability_factor_list
        if i %100 == 0:
       
            all_auctions.to_csv( bidder1.environment + "all_bids.csv")  
            all_profit.to_csv(bidder1.environment + "all_profit.csv")  

            all_shocks.to_csv( bidder1.environment +"all_shocks.csv")  
            all_stability.to_csv( bidder1.environment + "all_stability.csv")
            
#run the two-level Q-learning algorithm for a certian number of sub-experiments. 
stab_sim(300) 


all_auctions.to_csv( bidder1.environment + "partial_collusion_bids.csv")  
all_profit.to_csv(bidder1.environment + "partial_collusion_profit.csv")  

all_shocks.to_csv( bidder1.environment +"partial_collusion_shocks.csv")  
all_stability.to_csv( bidder1.environment + "partial_collusion_stability.csv")             # set figure size


plt.plot(all_auctions.index, all_auctions.filter(regex = '1$', axis =1 ).mean(axis = 1).rolling( (10)).mean())
plt.plot(all_auctions.index, all_auctions.filter(regex = '2$', axis =1).mean(axis = 1).rolling( (10)).mean())

plt.xlabel('Rounds')
plt.ylabel('Values')
plt.title('SPA')


plt.plot(all_profit.index, all_profit.filter(regex = '1$', axis =1 ).mean(axis = 1).rolling( (10)).mean())
plt.plot(all_profit.index, all_profit.filter(regex = '2$', axis =1).mean(axis = 1).rolling( (10)).mean())
plt.xlabel('Rounds')
plt.ylabel('Values')
plt.title('FPA')
plt.savefig('FPA_partial_spa_collusiuon.png', dpi = 1000)


