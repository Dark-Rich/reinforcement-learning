# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 21:09:18 2017

@author: Administrator
"""

from mdp import Mdp
import random

random.seed(0)

def random_pi():
    actions = ['n','w','e','s']
    r = int(random.random() * 4)
    return actions[r]

def compute_random_pi_state_value():
    value = [0.0 for r in xrange(9)]
    #大数模拟求均值
    num = 100000
    
    for k in xrange(1,num):
        for i in xrange(1,6):
            mdp = Mdp()
            s = i
            is_terminal = False
            gamma = 1.0
            v = 0.0
            while False == is_terminal:
                a = random_pi()
                is_terminal,s,r = mdp.transform(s,a)
                v += gamma * r
                gamma *= 0.5
            
            value[i] = (value[i] * (k-1) + v) / k
        if k % 10000 == 0:
            print value[1:9]
    print value[1:9]

def main():
    #计算随机策略下的状态值函数
    compute_random_pi_state_value()

if __name__ == '__main__':
    main()