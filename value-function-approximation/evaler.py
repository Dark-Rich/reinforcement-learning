# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:53:51 2017

@author: Administrator
"""


###############  Compute the gaps between current q and the best q ######
class Evaler:
    def __init__(self, mdp):
        self.mdp = mdp
        self.best = dict();
        f = open("./best_qfunc.txt")
        for line in f:
            line = line.strip()
            if len(line) == 0:  
                continue
            eles  = line.split(":")
            self.best[eles[0]] = float(eles[1])
    

    def eval(self,  policy):
        mdp = self.mdp
        cost = 0.0
        for key in self.best:
            keys  = key.split("_")
            s     = int(keys[0])
            if s in mdp.terminal_states:
                continue

            f     = mdp.start(s)
            a     = keys[1]

            error = policy.qfunc(f,a) - self.best[key]
            cost += error * error

        return cost