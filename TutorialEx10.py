# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 19:17:50 2018

@author: Alicia
"""

#Question 1 - Which model is more appropriate for the data in data.txt, quadratic or linear?

import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2
import scipy.integrate as spint
from plotnine import *

df=pandas.read_csv("data.txt",sep=",",header=0)
df.head(5)

ggplot(df,aes(x="x",y="y"))+geom_point()+theme_classic()

def LinearMod(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

def QuadraticMod(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]

    expected=B0+B1*obs.x+B2*obs.x**2
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

#Ask about these !!!
LinearGuess=numpy.array([1,1,1])
QuadraticGuess=numpy.array([1,1,1,1])

fitLinear=minimize(LinearMod,LinearGuess,method="Nelder-Mead",args=df)
fitQuadratic=minimize(QuadraticMod,QuadraticGuess,method="Nelder-Mead",args=df)

teststat=2*(fitLinear.fun-fitQuadratic.fun)
data=len(fitQuadratic.x)-len(fitLinear.x)
1-chi2.cdf(teststat,data)

#Question 2
def LVSim(y,t0,RN,aNN,aNT,RT,aTT,aTN):
    N=y[0]
    T=y[1]
    
    dNdt=RN*(1-N*aNN-T*aNT)*N
    dTdt=RT*(1-T*aTT-N*aTN)*T
    
    return [dNdt,dTdt]

#Case 1
params=(0.5,0.03,0.05,0.5,0.04,0.06)
y0=[0.1,0.1]
times=range(0,100)
sim=spint.odeint(func=LVSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"Species 1":sim[:,0],"Species 2":sim[:,1]})
print(ggplot(simDF,aes(x="t",y="Species 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Species 2"),color='red')+theme_classic())

#Case 2
params=(0.5,0.02,0.004,0.5,0.003,0.01)
y0=[0.1,0.1]
times=range(0,100)
sim=spint.odeint(func=LVSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"Species 1":sim[:,0],"Species 2":sim[:,1]})
print(ggplot(simDF,aes(x="t",y="Species 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Species 2"),color='red')+theme_classic())

#Case 3
params=(0.5,0.05,0.06,0.5,0.07,0.05)
y0=[0.1,0.1]
times=range(0,100)
sim=spint.odeint(func=LVSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"Species 1":sim[:,0],"Species 2":sim[:,1]})
print(ggplot(simDF,aes(x="t",y="Species 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Species 2"),color='red')+theme_classic())
