# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:03:56 2017

@author: Administrator
"""

class Mdp:
    def __init__(self):
        self.states = [1,2,3,4,5,6,7,8]
        self.terminal_states = dict()
        self.terminal_states[6] = 1
        self.terminal_states[7] = 1
        self.terminal_states[8] = 1
        
        self.actions = ['n','e','s','w']
        
        self.rewards = dict()
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0
        
        self.t = dict()
        self.t['1_s'] = 6
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['2_e'] = 3
        self.t['3_s'] = 7
        self.t['3_w'] = 2
        self.t['3_e'] = 4
        self.t['4_w'] = 3
        self.t['4_e'] = 5
        self.t['5_s'] = 8
        self.t['5_w'] = 4
        
        self.gamma = 0.8
    
    def transform(self,state,action):
        if state in self.terminal_states:
            return True,state,0
        
        key = '%d_%s' % (state,action)
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        
        is_terminal = False
        if next_state in self.terminal_states:
            is_terminal = True
        
        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]
        
        return is_terminal,next_state,r
    
    
    def get_terminal_states(self):
        return self.terminal_states
    
    
    def get_gamma(self):
        return self.gamma
    
    
    def get_states(self):
        return self.states
    
    
    def get_actions(self):
        return self.actions
    
    
    def get_random_pi_sample(self,num):
        import random
        state_sample = []
        action_sample = []
        reward_sample = []
        for i in xrange(num):
            s_tmp = []
            a_tmp = []
            r_tmp = []
            
            s = self.states[int(random.random() * len(self.states))]
            t = False
            
            while t == False:
                a = self.actions[int(random.random() * len(self.actions))]
                t,s1,r = self.transform(s,a)
                s_tmp.append(s)
                r_tmp.append(r)
                a_tmp.append(a)
                s = s1
            state_sample.append(s_tmp)
            action_sample.append(a_tmp)
            reward_sample.append(r_tmp)
        
        return state_sample,action_sample,reward_sample