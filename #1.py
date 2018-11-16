# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 00:44:30 2018

@author: vysan
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize 
from scipy.stats import norm
from plotnine import *
from scipy.stats import chi2
from scipy import stats

#load data
data=pd.read_csv("data.txt", header=0, sep= ",")
d=ggplot(data,aes(x="x",y="y"))+geom_point()+theme_classic()
d

#custom likelihood function
def quadllike(p,obs):
    a=p[0]
    b=p[1]
    c=p[2]
    sigma=p[3]
    
    expected=a+(b*obs.x)+c*((obs.x)**2)
    
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

def linllike(p,obs):
    a=p[0]
    b=p[1]
    sigma=p[2]
    
    expected=a+(b*obs.x)
    
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
 
#estimate parameters by minimizing negative log likelihood
quadGuess=np.array([1,1,1,1])
fitquad=minimize(quadllike,quadGuess,method="Nelder-Mead",options={'disp': True},args=data)
print (fitquad.x)

linGuess=np.array([1,1,1])
fitlin=minimize(linllike,linGuess,method="Nelder-Mead",options={'disp': True},args=data)

#run likelihood ratio test
teststat=2*(fitlin.fun-fitquad.fun)
print teststat
df=len(fitquad.x)-len(fitlin.x)
1-chi2.cdf(teststat,df)

#since p value .899 > .05, cannot reject null hypothesis, there is no significant difference between
#using quadratic or linear model for the data
