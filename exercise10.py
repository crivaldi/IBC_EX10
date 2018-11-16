#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:14:50 2018

@author: mlpoterek
"""

import pandas as pd
import numpy as np
import scipy
import scipy.integrate as spint
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2
from plotnine import *

location = r'/Users/mlpoterek/Biocomp/IBC_EX10/data.txt'
df = pd.read_csv(location, sep = ',')

##Question 1##

#Linear model function
def linMod(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    pred=B0+B1*obs.x
    nll=-1*norm(pred,sigma).logpdf(obs.y).sum()
    return nll

#Squared model function
def sqMod(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    pred=B0+B1*obs.x+B2*(obs.x ** 2)
    nll=-1*norm(pred,sigma).logpdf(obs.y).sum()
    return nll

#Estimate parameters
linGuess=np.array([1,1,1])
sqGuess=np.array([1,1,1,1])

linFit=minimize(linMod,linGuess,method="Nelder-Mead",options={'disp': True},args=df)
sqFit=minimize(sqMod,sqGuess,method="Nelder-Mead",options={'disp': True},args=df)
#Based on the generated sigma values, the squared model is marginally better


#Likelihood ratio test
test_stat=2*(sqFit.fun-linFit.fun)
df1= len(sqFit.x)-len(linFit.x)
chi_sq=1-chi2.cdf(test_stat, df1)


##Question 2##

def LVmod(y,t0,R1,R2,a11,a12,a21,a22):
    N1=y[0]
    N2=y[1]
    
    dN1dt=R1*(1-(N1*a11)-(N2*a12))*N1
    dN2dt=R2*(1-(N2*a22)-(N1*a21))*N2
    
    return [dN1dt,dN2dt]

#False: a12>a11 & a21<a22
#False: a12<a11 & a21>a22
#False: a12>a11 & a21>a22
    
#True: a12<a11 & a21<a22
times=range(0,100)
y0=[0.1,0.1]
params=(0.5,2,0.5,0.3,0.4,0.5)
sim=spint.odeint(func=LVmod,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"pop":sim[:,0],"pop1":sim[:,1]})
ggplot(simDF,aes(x="t",y="pop"))+geom_line()+geom_line(simDF,aes(x="t",y="pop1"),color='red')+theme_classic()
#Both populations achieve equilibrium and thus can coexist

#False: a12>a11 & a21<a22
times=range(0,400)
y0=[0.2,0.1]
params=(0.5,2,0.3,0.5,0.4,0.5)
sim=spint.odeint(func=LVmod,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"pop":sim[:,0],"pop1":sim[:,1]})
ggplot(simDF,aes(x="t",y="pop"))+geom_line()+geom_line(simDF,aes(x="t",y="pop1"),color='red')+theme_classic()
#One population goes extinct

#False: a12<a11 & a21>a22
times=range(0,100)
y0=[0.1,0.2]
params=(0.5,2,0.3,0.5,0.5,0.4)
sim=spint.odeint(func=LVmod,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"pop":sim[:,0],"pop1":sim[:,1]})
ggplot(simDF,aes(x="t",y="pop"))+geom_line()+geom_line(simDF,aes(x="t",y="pop1"),color='red')+theme_classic()
#One population goes extinct

#False: a12>a11 & a21>a22
times=range(0,100)
y0=[0.2,0.2]
params=(0.5,2,0.3,0.6,0.6,0.2)
sim=spint.odeint(func=LVmod,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"pop":sim[:,0],"pop1":sim[:,1]})
ggplot(simDF,aes(x="t",y="pop"))+geom_line()+geom_line(simDF,aes(x="t",y="pop1"),color='red')+theme_classic()
#One population goes extinct

