#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 00:20:39 2018

@author: saurylara
"""

import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from scipy import stats 
import scipy.integrate as spint
from plotnine import *

#Import the data.txt file
Datafile=pandas.read_csv("/Users/saurylara/Desktop/IBC_EX10/data.txt")

#Create a linear function model
def nllike(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    model=B0+B1*obs.x
    nll=-1*norm(model,sigma).logpdf(obs.y).sum()
    return nll

#Estimate parameters by minimizing the negative log likelihood
initialGuess=numpy.array([1,1,1])
fit=minimize(nllike,initialGuess,method="Nelder-Mead",options={'disp': True},args=Datafile)

#Create a quadratic function model
def nllikequadratic(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    model=B0+B1*obs.x+B2*(obs.x)**2
    nllquadratic=-1*norm(model,sigma).logpdf(obs.y).sum()
    return nllquadratic

#Estimate parameters by minimizing the negative log likelihood
initialGuessQuadratic=numpy.array([2,2,2,2])
fitQuadratic=minimize(nllikequadratic,initialGuessQuadratic,method="Nelder-Mead",options={'disp': True},args=Datafile)

#Perform the likelihood ratio test
teststat=2*(fit.fun-fitQuadratic.fun)
Datafile=len(fitQuadratic.x)-len(fit.x)
ratioTest=1-stats.chi2.cdf(teststat,Datafile)
print(ratioTest)

#The resulting p-value is 0.8991753751339633 which is much higher than 0.05 which indicates that there is no statistical significance. 
#Therefore, the models esentially have the same ability in quantifying the data from the file and there is no advantage to using one model over the other. 
#Nevertheless, based on the parsimony principle in biology, the linear model should be used considering it is simpler and involves the use of fewer parameters. 

#Create a function expressing Lotka & Volterra's classic model of competition
def tumorSim(y,t0,RN,a11,a12,a22,a21,RT):
    N=y[0]
    T=y[1]
    
    dNdt=RN*(1-N*a11-T*a12)*N
    dTdt=RT*(1-T*a22-N*a21)*T
    
    return [dNdt,dTdt]

#Case 1: Model simulation with the conditions a11<a12 and a21<a22 so that only species A(black) lasts
times=range(0,100)
y0=[0.1,0.1]
params=(0.5,0.01,0.06,0.07,0.02,0.5)
sim=spint.odeint(func=tumorSim,y0=y0,t=times,args=params)
simDF1=pandas.DataFrame({"t":times,"Species A":sim[:,0],"Species B":sim[:,1]})
print(ggplot(simDF1,aes(x="t",y="Species A"))+geom_line()+geom_line(simDF1,aes(x="t",y="Species B"),color='red')+theme_classic())

#Case 2: Model simulation with the conditions a12<a11 and a22<a21 so that only species B(red) lasts
times=range(0,100)
y0=[0.1,0.1]
params=(0.5,0.02,0.004,0.003,0.01,0.5)
sim=spint.odeint(func=tumorSim,y0=y0,t=times,args=params)
simDF2=pandas.DataFrame({"t":times,"Species A":sim[:,0],"Species B":sim[:,1]})
print(ggplot(simDF2,aes(x="t",y="Species A"))+geom_line()+geom_line(simDF2,aes(x="t",y="Species B"),color='red')+theme_classic())

#Case 3: Model simulation with the conditions a12<a11 and a21<a22 so that both species coexist
times=range(0,100)
y0=[0.1,0.1]
params=(0.5,0.08,0.04,0.07,0.03,0.5)
sim=spint.odeint(func=tumorSim,y0=y0,t=times,args=params)
simDF3=pandas.DataFrame({"t":times,"Species A":sim[:,0],"Species B":sim[:,1]})
print(ggplot(simDF3,aes(x="t",y="Species A"))+geom_line()+geom_line(simDF3,aes(x="t",y="Species B"),color='red')+theme_classic())

