# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:56:37 2017

@author: Administrator
"""

from mdp import Mdp

mdp = Mdp()
states = mdp.get_states()
actions = mdp.get_actions()
gamma = mdp.get_gamma()

def mc(gamma,state_sample,action_sample,reward_sample):
    vfunc =  dict()
    nfunc = dict()
    for state in states:
        vfunc[state] = 0.0
        nfunc[state] = 0.0
    
    for i in xrange(len(state_sample)):
        G = 0.0
        for step in xrange(len(state_sample[i])-1,-1,-1):
            G *= gamma
            G += reward_sample[i][step]
        for step in xrange(len(state_sample[i])):
            s = state_sample[i][step]
            vfunc[s] += G
            nfunc[s] += 1.0
            G -= reward_sample[i][step]
            G /= gamma
    
    for state in states:
        if nfunc[s] > 0.000001:
            vfunc[s] /= nfunc[s]
    print 'mc'
    print vfunc
    return vfunc


def td(alpha,gamma,state_sample,action_sample,reward_sample):
    vfunc = dict()
    for state in states:
        vfunc[state] = 0.0
    
    for i in xrange(len(state_sample)):
        for step in xrange(len(state_sample[i])):
            s = state_sample[i][step]
            r = reward_sample[i][step]
            
            if len(state_sample[i]) - 1 > step:
                s1 = state_sample[i][step+1]
                next_v = vfunc[s1]
            else:
                next_v = 0.0
            
            vfunc[s] = vfunc[s] + alpha * (r + gamma * next_v - vfunc[s])
    print 'td'
    print vfunc
    return vfunc


def main():
    s,a,r = mdp.get_random_pi_sample(100)
    mc(0.5,s,a,r)
    td(0.15,0.5,s,a,r)


if __name__ == '__main__':
    main()