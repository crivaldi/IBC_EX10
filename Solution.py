# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 09:38:00 2018

@author: Annaliese
"""

# import packages
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *
from scipy import stats
import scipy.integrate as spint

# import the data
df = pd.read_csv('data.txt')

# define custom likelihood functions
def nllike1(p, obs):
    a = p[0]
    b = p[1]
    sigma = p[2]
    
    expected = a + b * obs.x
    nll = -1 * norm(expected,sigma).logpdf(obs.y).sum()
    return nll

def nllike2(p, obs):
    a = p[0]
    b = p[1]
    c = p[2]
    sigma = p[3]
    
    expected = a + b*obs.x + c*((obs.x)**2)
    nll = -1 * norm(expected, sigma).logpdf(obs.y).sum()
    return(nll)
    
# set initial guess
initialGuess1 = np.array([1,1,1])
initialGuess2 = np.array([15.8, 4.6, -0.002, 19])


# minimize negative log likelihood
fit1 = minimize(nllike1, initialGuess1, method = "Nelder-Mead", 
                options = {'disp':True}, args = df)
fit2 = minimize(nllike2, initialGuess2, method = "Nelder-Mead", 
                options = {'disp':True}, args = df)

# find test statistic
teststat = 2*(fit1.fun - fit2.fun)

# degrees of freedom
df = len(fit2.x) - len(fit1.x)

1 - stats.chi2.cdf(teststat, df)

teststat > 1 - stats.chi2.cdf(teststat, df)
# so we conclude null, that the simpler model is better

#############################################################

# Modelling Problem

def compSim(y,t0,R1, R2, alpha11, alpha12, alpha21, alpha22):
    N1=y[0]
    N2=y[1]
    
    dN1dt=R1 * (1-N1*alpha11 - N2*alpha12) * N1
    dN2dt = R2 * (1 - N2*alpha22 - N1 * alpha21) * N2
    
    return [dN1dt,dN2dt]

# Case 1: alpha12 < alpha11, alpha21 < alpha22 (Coexistence)
times=range(0,100)
y0=[1,1]
params=(0.5, 2,0.7, 0.5, 0.5, 0.7)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"Population 1":sim[:,0],"Population 2":sim[:,1]})
ggplot(simDF,aes(x="t",y="Population 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Population 2"),color='red')+theme_classic()

# Case 2: Both violated- Population 2 Dies Out
times=range(0,100)
y0=[1, 1]
params=(0.5,2,0.5, 0.7, 0.7, 0.5)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"Population 1":sim[:,0],"Population 2":sim[:,1]})
ggplot(simDF,aes(x="t",y="Population 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Population 2"),color='red')+theme_classic()

# Case 3: alpha12 > alpha11, but alpha 21 < alpha 22: Only Population 1 Establishe
#s, though not fully
times=range(0,100)
y0=[1, 1]
params=(0.5, 2,0.5, 0.7, 0.5, 0.7)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"Population 1":sim[:,0],"Population 2":sim[:,1]})
ggplot(simDF,aes(x="t",y="Population 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Population 2"),color='red')+theme_classic()

