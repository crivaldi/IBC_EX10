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
1-stats.chi2.cdf(teststat,data)

#Question 2
def ddSim(y,t0,r,K):
    N=y[0]
    dNdt=r*(1-N/K)*N
    return [dNdt]

params=(-.1,1)
y=[0.1]
times=range(0,600)

modelSim=spint.odeint(func=ddSim,y0=y,t=times,args=params)
modelOutput=pandas.DataFrame({"t":times,"N":modelSim[:,0]})
ggplot(modelOutput,aes(x="t",y="N"))+geom_line()+theme_classic()