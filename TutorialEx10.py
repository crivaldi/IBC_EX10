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

#Question - why is it saying stats is not defined? - this should be almost complete

#Question 2 - make your own cases with different params like the tumor cell example (R should be between zero and 1, not negative, K should be like 100?)
#Note for Ex10 #2 - if r is 1 population is double
#Start with r is less than 1 and k is 1/alpha - alpha should be less than .1 - use these for params - email TA to ask if need to
def LVSim(y,t0,RN,aNN,aNT,RT,aTT,aTN):
    N=y[0]
    T=y[1]
    
    dNdt=RN*(1-N*aNN-T*aNT)*N
    dTdt=RT*(1-T*aTT-N*aTN)*T
    
    return [dNdt,dTdt]

params=(0.5,0.5,)
y0=[1.0,1.0]
times=range(0,100)
param[3]<param[4]

sim=spint.odeint(func=LVSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"species1":sim[:,0],"species2":sim[:,1]})
ggplot(simDF,aes(x="t",y="species1"))+geom_line()+geom_line(simDF,aes(x="t",y="species2"),color='red')+theme_classic()

params=(0.1,1)
y0=[0.1,0.1]
times=range(0,100)
sim=spint.odeint(func=LVSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"species1":sim[:,0],"species2":sim[:,1]})
ggplot(simDF,aes(x="t",y="species1"))+geom_line()+geom_line(simDF,aes(x="t",y="species2"),color='red')+theme_classic()

params=(0.1,100,0.05,0.1,100,0.05)
y0=[0.1,0.1]
times=range(0,100)
sim=spint.odeint(func=LVSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"species1":sim[:,0],"species2":sim[:,1]})
ggplot(simDF,aes(x="t",y="species1"))+geom_line()+geom_line(simDF,aes(x="t",y="species2"),color='red')+theme_classic()
#Questions
#simDF why will it not let me define species1/species2? - need to check params for multiple (3?) cases
#What should y0 be set at for each case?
#Check original function def and equations - should be correct and work for all 3 cases 