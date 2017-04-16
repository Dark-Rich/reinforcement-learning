# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:45:12 2017

@author: Administrator
"""

from mdp import Mdp
import random
import matplotlib.pyplot as plt
random.seed(0)

mdp = Mdp()
states = mdp.get_states()
actions = mdp.get_actions()
gamma = mdp.get_gamma()

best = dict()

def read_best():
    f = open('best_qfunc.txt')
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        eles = line.split(':')
        best[eles[0]] = float(eles[1])


def compute_error(qfunc):
    sum_num = 0.0
    for key in qfunc:
        error = qfunc[key] - best[key]
        sum_num += error *  error
    return sum_num


def epsilon_greedy(qfunc,state,epsilon):
    #max q action
    amax = 0
    key = '%d_%s' % (state,actions[0])
    qmax = qfunc[key]
    for i in xrange(len(actions)):
        key = '%d_%s' % (state,actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i
    
    #probability
    pro = [0.0 for i in xrange(len(actions))]
    pro[amax] += 1 - epsilon
    for i in xrange(len(actions)):
        pro[i] += epsilon / len(actions)
    
    #choose
    r = random.random()
    s = 0.0
    for i in xrange(len(actions)):
        s += pro[i]
        if s >= r:
            return actions[i]
    return actions[len(actions)-1]


"""
on-policy off-policy
区别在于所估计的policy或value-function和生成样本时所采用的policy是否相同
如果一样，那就是on-policy的，否则是off-policy的
"""


#on-policy
#GLIE Monte-Carlo Control
def mc(num_iter,epsilon):
    x = []
    y = []
    n = dict()
    qfunc = dict()
    for s in states:
        for a in actions:
            qfunc['%d_%s' % (s,a)] = 0.0
            n['%d_%s' % (s,a)] = 0.001
    for i in xrange(num_iter):
        x.append(i)
        y.append(compute_error(qfunc))
        
        s_sample = []
        a_sample = []
        r_sample = []
        
        s = states[int(random.random() * len(states))]
        t = False
        count = 0
        #采样
        while t == False and count < 100:
            a = epsilon_greedy(qfunc,s,epsilon)
            t,s1,r = mdp.transform(s,a)
            s_sample.append(s)
            r_sample.append(r)
            a_sample.append(a)
            s = s1
            count += 1
        
        g = 0.0
        for i in xrange(len(s_sample)-1,-1,-1):
            g *= gamma
            g += r_sample[i]
            
        for i in xrange(len(s_sample)):
            key = '%d_%s' % (s_sample[i],a_sample[i])
            n[key] += 1.0
            qfunc[key] = (qfunc[key] * (n[key]-1) + g) / n[key]
            
            g -= r_sample[i]
            g /= gamma
    plt.plot(x,y,'-',label='mc epsilon=%2.1f' % (epsilon))
    plt.legend()
    return qfunc

#on-policy
def sarsa(num_iter,alpha,epsilon):
    x = []
    y = []
    qfunc = dict()
    for s in states:
        for a in actions:
            key = '%d_%s' % (s,a)
            qfunc[key] = 0.0
    
    for i in xrange(num_iter):
        x.append(i)
        y.append(compute_error(qfunc))
        
        s = states[int(random.random() * len(states))]
        a = actions[int(random.random() * len(actions))]
#        a = epsilon_greedy(qfunc,s,epsilon)
        t = False
        count = 0
        while t == False and count < 100:
            key = '%d_%s' % (s,a)
            t,s1,r = mdp.transform(s,a)
            a1 = epsilon_greedy(qfunc,s1,epsilon)
            key1 = '%d_%s' % (s1,a1)
            qfunc[key] = qfunc[key] + alpha * (r + gamma * qfunc[key1] - qfunc[key])
            s = s1
            a = a1
            count += 1
    
    plt.plot(x,y,'--',label='sarsa alpha=%2.1f epsilon=%2.1f' % (alpha,epsilon))
    plt.legend()
    return qfunc


#off-policy
#Q-learning
def qlearning(num_iter,alpha,epsilon):
    x = []
    y = []
    qfunc =dict()
    for s in states:
        for a in actions:
            key = '%d_%s' % (s,a)
            qfunc[key] = 0.0
    
    for i in xrange(num_iter):
        x.append(i)
        y.append(compute_error(qfunc))
        
        s = states[int(random.random() * len(states))]
        a = actions[int(random.random() * len(actions))]
#        a = epsilon_greedy(qfunc,s,epsilon)
        t = False
        count = 0
        while t == False and count < 100:
            key = '%d_%s' % (s,a)
            t,s1,r = mdp.transform(s,a)
            
            key1 = ''
            qmax = -1.0
            #greedy
            for a1 in actions:
                if qmax < qfunc['%d_%s' % (s1,a1)]:
                    qmax = qfunc['%d_%s' % (s1,a1)]
                    key1 = '%d_%s' % (s1,a1)
            qfunc[key] = qfunc[key] + alpha * (r + gamma * qfunc[key1] - qfunc[key])
            
            s = s1
            a = epsilon_greedy(qfunc,s1,epsilon)
            count += 1
    
    plt.plot(x,y,'-.',label='q alpha=%2.1f epsilon=%2.1f' % (alpha,epsilon))
    plt.legend()
    return qfunc
    
    
def main():
    read_best()
    mc(10000,0.1)
    sarsa(10000,0.1,0.1)
    qlearning(10000,0.1,0.1)


if __name__ == '__main__':
    main()