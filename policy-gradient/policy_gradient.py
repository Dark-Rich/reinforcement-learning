# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:34:27 2017

@author: Administrator
"""

from mdp import MdpId
from mdp import MdpWall
from policy import SoftmaxPolicy
from policy import ValuePolicy
from evaler import Evaler
import random
import matplotlib.pyplot as plt

random.seed(0)

def update_value_policy(value_policy, f, a, tvalue, alpha):
    pvalue        = value_policy.qfunc(f, a)
    error         = pvalue - tvalue
    fea           = value_policy.get_fea_vec(f, a)
    value_policy.theta -= alpha * error * fea 


def update_softmax_policy(softmax_policy, f, a, qvalue, alpha):

    fea  = softmax_policy.get_fea_vec(f,a)
    prob = softmax_policy.pi(f)
    
    delta_logJ = fea
    
    for i in xrange(len(softmax_policy.actions)):
        a1          = softmax_policy.actions[i]
        fea1        = softmax_policy.get_fea_vec(f,a1)
        delta_logJ -= fea1 * prob[i]

    softmax_policy.theta += alpha * delta_logJ * qvalue


################ Different model free RL learning algorithms #####
def mc(mdp,softmax_policy, num_iter, alpha):
    gamma   = mdp.gamma
    for i in xrange(len(softmax_policy.theta)):
        softmax_policy.theta[i] = 0.1

    for iter1 in xrange(num_iter):

        f_sample = []
        a_sample = []
        r_sample = []   
        
        f = mdp.start()
        t = False
        count = 0
        while False == t and count < 100:
            a = softmax_policy.take_action(f)
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
            update_softmax_policy(softmax_policy, f_sample[i], a_sample[i], g, alpha)

            g -= r_sample[i]
            g /= gamma
        
    return softmax_policy

def sarsa(mdp, evaler, softmax_policy, value_policy, num_iter, alpha):
    gamma   = mdp.gamma
    actions = mdp.actions
    y       = []
    
    for i in xrange(len(value_policy.theta)):
        value_policy.theta[i]  = 0.1
    for i in xrange(len(softmax_policy.theta)): 
        softmax_policy.theta[i] = 0.0
    

    for i in xrange(num_iter):
        y.append(evaler.eval(value_policy))
        
        f = mdp.start()
        a = actions[int(random.random() * len(actions))]
        t = False
        count = 0

        while False == t and count < 100:
            t,f1,r      = mdp.receive(a)
            a1          = softmax_policy.take_action(f1)
            
            update_value_policy(value_policy, f, a, r + gamma * value_policy.qfunc(f1, a1), alpha)
            update_softmax_policy(softmax_policy, f, a, value_policy.qfunc(f,a), alpha)

            f           = f1
            a           = a1
            count      += 1

    return softmax_policy, y

def main():
    mdp = MdpId()
    evaler = Evaler(mdp)
    softmax_policy = SoftmaxPolicy(mdp,0.1)
    value_policy = ValuePolicy(mdp,0.1)
    policy = mc(mdp,softmax_policy,1000,0.1)
    policy,y = sarsa(mdp,evaler,softmax_policy,value_policy,1000,0.1)
    
    fig    = plt.figure(figsize=(12,8))
    ax     = fig.add_subplot(111)
    ax.plot(y,color='r',linestyle=':',label='Actor-Critic')
    ax.legend()
    fig.show()

if __name__ == '__main__':
    main()