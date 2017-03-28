# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:01:50 2017

@author: Administrator
"""

from mdp import Mdp

class PolicyValue:
    def __init__(self,mdp):
        self.v = [0.0 for i in xrange(len(mdp.states) + 1)]
        
        self.pi = dict()
        for state in mdp.states:
            if state in mdp.terminal_states:
                continue
            self.pi[state] = mdp.actions[0]
    
    
    def evaluate_policy(self,mdp):
        for i in xrange(1000):
            delta = 0.0
            for state in mdp.states:
                if state in mdp.terminal_states:
                    continue
                action = self.pi[state]
                t,s,r = mdp.transform(state,action)
                new_v = r + mdp.gamma * self.v[s]
                delta += abs(self.v[state] - new_v)
                self.v[state] = new_v
            
            if delta < 1e-8:
                break


    def imporve_policy(self,mdp):
        for state in mdp.states:
            if state in mdp.terminal_states:
                continue
            a1 = mdp.actions[0]
            t,s,r = mdp.transform(state,a1)
            v1 = r + mdp.gamma * self.v[s]
            
            for action in mdp.actions:
                t,s,r = mdp.transform(state,action)
                if v1 < r + mdp.gamma * self.v[s]:
                    a1 = action
                    v1 = r + mdp.gamma * self.v[s]
            
            self.pi[state] = a1
   
    
    def iterate_policy(self,mdp):
        for i in xrange(100):
            self.evaluate_policy(mdp)
            self.imporve_policy(mdp)
    
    
def main():
    mdp = Mdp()
    policy_value = PolicyValue(mdp)
    policy_value.iterate_policy(mdp)
    print 'value:'
    for i in xrange(1,6):
        print '%d:%f\t' % (i,policy_value.v[i])
        print ''
    for i in xrange(1,6):
        print policy_value.pi[i]


if __name__ == '__main__':
    main()