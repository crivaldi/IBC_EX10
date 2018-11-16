# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:44:22 2018

@author: Seth
"""

#import packages and data

import numpy
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *
from scipy.stats import chi2
data = pd.read_csv('data.txt', header = 0, sep = ',')

#plot data

ggplot(data,aes(x='x',y='y'))+geom_point()+theme_classic()

#linear likelihood function

def linear(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

#quadratic likelihood function

def quadratic(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    expected=B0+B1*obs.x+B2*obs.x**2
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

#linear parameters
    
linearguess=numpy.array([1,1,1])
linearfit=minimize(linear,linearguess,method="Nelder-Mead",options={'disp': True},args=data)
print(linearfit.x)

#quadratic parameters

quadraticguess=numpy.array([1,1,1,1])
quadraticfit=minimize(quadratic,quadraticguess,method="Nelder-Mead",options={'disp': True},args=data)
print(quadraticfit.x)

#likelihood ratio

test = 2*(linearfit.fun-quadraticfit.fun)
df = len(quadraticfit.x)-len(linearfit.x)
1-chi2.cdf(teststat,df)

#p > 0.05. Fit of models to data is not significantly different.