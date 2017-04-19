# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:38:31 2017

@author: Administrator
"""

from mdp import MdpId
from mdp import MdpWall
from policy import Policy
from evaler import Evaler
import random
import matplotlib.pyplot as plt

random.seed(0)


def update(policy, f, a, tvalue, alpha):
    pvalue        = policy.qfunc(f, a)
    error         = pvalue - tvalue;
    fea           = policy.get_fea_vec(f, a)
    policy.theta -= alpha * error * fea

################ Different model free RL learning algorithms #####

def mc(mdp, policy, evaler, num_iter, alpha):
    gamma   = mdp.gamma
    y = []
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for i in xrange(num_iter):

        y.append(evaler.eval(policy))
        s_sample = []
        f_sample = []
        a_sample = []
        r_sample = []   
        
        f = mdp.start()
        t = False
        count = 0
        while False == t and count < 100:
            a = policy.epsilon_greedy(f)
            s_sample.append(mdp.current)
            t, f1, r  = mdp.receive(a)
            f_sample.append(f)
            r_sample.append(r)
            a_sample.append(a)
            f = f1            
            count += 1


        g = 0.0
        for i in xrange(len(f_sample)-1, -1, -1):
            g *= gamma
            g += r_sample[i]
        
        for i in xrange(len(f_sample)):
            update(policy, f_sample[i], a_sample[i], g, alpha)

            g -= r_sample[i]
            g /= gamma
        
    return policy,y 


def sarsa(mdp, policy, evaler, num_iter, alpha):
    actions = mdp.actions
    gamma   = mdp.gamma
    y = []
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for i in xrange(num_iter):
        y.append(evaler.eval(policy))
        f = mdp.start()
        a = actions[int(random.random() * len(actions))]
        t = False
        count = 0

        while False == t and count < 100:
            t,f1,r      = mdp.receive(a)
            a1          = policy.epsilon_greedy(f1)
            update(policy, f, a, r + gamma * policy.qfunc(f1, a1), alpha)

            f           = f1
            a           = a1
            count      += 1

    return policy, y;

def qlearning(mdp, policy, evaler, num_iter, alpha):
    actions = mdp.actions
    gamma   = mdp.gamma
    y = []
    
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for i in xrange(num_iter):
        y.append(evaler.eval(policy))

        f = mdp.start();    
        a = actions[int(random.random() * len(actions))]
        t = False
        count = 0

        while False == t and count < 100:
            t,f1,r      = mdp.receive(a)
            qmax = -1.0
            for a1 in actions:
                pvalue = policy.qfunc(f1, a1)
                if qmax < pvalue:
                    qmax = pvalue
            update(policy, f, a, r + gamma * qmax, alpha)

            f           = f1
            a           = policy.epsilon_greedy(f)
            count      += 1   
    
    return policy, y

    
def main():
    mdp = MdpId()
    evaler = Evaler(mdp)
    policy = Policy(mdp,0.1)
    policy,y1 = mc(mdp,policy,evaler,1000,0.1)
    policy,y2 = sarsa(mdp,policy,evaler,1000,0.1)
    policy,y3 = qlearning(mdp,policy,evaler,1000,0.1)
    
    fig    = plt.figure(figsize=(12,8))
    ax     = fig.add_subplot(111)
    ax.plot(y1,color='g',linestyle=':',label='mc')
    ax.plot(y2,color='b',linestyle='-',label='sarsa')
    ax.plot(y3,color='r',linestyle='-.',label='qlearning')
    ax.legend()
    fig.show()
    
if __name__ == '__main__':
    main()