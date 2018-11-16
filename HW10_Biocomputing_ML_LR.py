# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:07:30 2018

@author: Alexis
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2
from scipy import stats
from plotnine import *

Data_df=pd.read_csv('data.txt') #reads the data frame


def linear(p,obs):#linear function
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x
    nll_Linear=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll_Linear

def quadractic(p,obs):#quadractic function
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    expected=B0+B1*obs.x+B2*((obs.x)**(2))
    nll_quadractic=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll_quadractic

initialGuess_Linear=np.array([1,1,1])#initial linear guess
initialGuess_Quadractic=np.array([1,1,1,1])#initial quadractic guess
#minimizes negative log value
fit_Linear=minimize(linear,initialGuess_Linear,method="Nelder-Mead",options={'disp': True},args=Data_df)
fit_Quadractic=minimize(quadractic,initialGuess_Quadractic,method="Nelder-Mead",options={'disp': True},args=Data_df)

#prints optimized parameters
print(fit_Linear.x)
print(fit_Quadractic.x)

# runs likelihood ratio test
teststat=2*(fit_Linear.fun-fit_Quadractic.fun)
df=len(fit_Quadractic.x)-len(fit_Linear.x)
Result=1-stats.chi2.cdf(teststat,df)
print(Result)
#The p value is .89917. As it is so high it means that there is not statistical
#difference between using one model versus the other
#As such is is more appropriate is use the linear model as it simpler with less parameters