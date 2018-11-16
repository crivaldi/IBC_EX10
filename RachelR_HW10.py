# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 23:40:27 2018

@author: Rachel R
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *
import scipy.integrate as spint
from scipy.stats import chi2

## Question 1
df = pd.read_csv('data.txt')
def nllike(p,obs):
    a=p[0]
    b=p[1]
    sigma=p[2]
    expected=a+b*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

initialGuess=np.array([1,1,1])
fit=minimize(nllike,initialGuess,method="Nelder-Mead",options={'disp': True},args=df)

def nllike1(p, obs):
    a=p[0]
    b=p[1]
    c=p[2]
    sigma=p[3]
    expected = a + b*obs.x + c*((obs.x)**2)
    nll = -1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

initialGuess1=np.array([1,1,1,1])
fit1=minimize(nllike1,initialGuess1,method="Nelder-Mead",options={'disp': True},args=df)

teststat=2*(fit.fun-fit1.fun)
data=len(fit1.x)-len(fit.x)
1-chi2.cdf(teststat,data)
teststat > 1 - chi2.cdf(teststat, data) #This output was false, it can be assumed that the linear model is better

## Question 2
def sims(y,t0,R1, R2, alpha11, alpha12, alpha21, alpha22):
    N1=y[0]
    N2=y[1]    
    dN1dt=R1 * (1-N1*alpha11 - N2*alpha12) * N1
    dN2dt = R2 * (1 - N2*alpha22 - N1 * alpha21) * N2   
    return [dN1dt,dN2dt]

# Simulation 1 : Population 1 collapses, but population 2 establishes
times=range(0,100)
y0=[1,1]
params=(0.5, 2, 0.2, 0.5, 0.5, 0.2)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"Population 1":sim[:,0],"Population 2":sim[:,1]})
ggplot(simDF,aes(x="t",y="Population 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Population 2"),color='red')+theme_classic()

# Simulation 2 : Population 1 establishes, but population 2 immediately collapses
times=range(0,100)
y0=[1,1]
params=(0.5, 2, 0.2, 0.5, 2, 0.2)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"Population 1":sim[:,0],"Population 2":sim[:,1]})
ggplot(simDF,aes(x="t",y="Population 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Population 2"),color='red')+theme_classic()

# Simulation 3 : Coexistence
times=range(0,100)
y0=[1,1]
params=(0.5, 1, 1, 0.5, 0.5, 1)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"Population 1":sim[:,0],"Population 2":sim[:,1]})
ggplot(simDF,aes(x="t",y="Population 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Population 2"),color='red')+theme_classic()
