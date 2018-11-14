#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:53:26 2018

@author: syli
"""
# task 1

import pandas as pd
import numpy as np
from plotnine import *
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2

# load data
data = pd.read_table("data.txt",sep=",")
data.head()
ggplot(data,aes(x="x",y="y"))+geom_line()+theme_classic()

# create likelihood functions
def complexMod(p,data):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    
    pred=B0+B1*data.x+B2*data.x**2
    
    nll=-1*norm(pred,sigma).logpdf(data.y).sum()
    return nll

def simpleMod(p,data):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    
    pred=B0+B1*data.x
    
    nll=-1*norm(pred,sigma).logpdf(data.y).sum()
    return nll

# estimate parameters
complexGuess=np.array([1,5,1,1])
simpleGuess=np.array([1,5,1])

fitComplex=minimize(complexMod,complexGuess,method="Nelder-Mead",args=data)
fitSimple=minimize(simpleMod,simpleGuess,method="Nelder-Mead",args=data)

# run likelihood ratio test
teststat=2*(fitSimple.fun-fitComplex.fun)

df=len(fitComplex.x)-len(fitSimple.x)

chi2.cdf(teststat,df)


# task 2

import scipy.integrate as spint

def Sim(y,t0,R1,R2,a11,a12,a22,a21):
    N1=y[0]
    N2=y[1]
    dN1dt=R1*(1-N1*a11-N2*a12)*N1
    dN2dt=R2*(1-N2*a22-N1*a21)*N2
    return [dN1dt,dN2dt]

times=range(0,100)
Nint=[0.01,0.01]

# a12<a11, a21<a22
params1=(0.5,0.5,10,5,8,4)

modelSim1=spint.odeint(func=Sim,y0=Nint,t=times,args=params1)
Output1=pd.DataFrame({"t":times,"N1":modelSim1[:,0],"N2":modelSim1[:,1]})
ggplot(Output1,aes(x="t",y="N1"))+geom_line()+geom_line(Output1,aes(x="t",y="N2"),color='red')+theme_classic()

# a12<a11, a21>a22
params2=(0.5,0.5,10,5,4,8)

modelSim2=spint.odeint(func=Sim,y0=Nint,t=times,args=params2)
Output2=pd.DataFrame({"t":times,"N1":modelSim2[:,0],"N2":modelSim2[:,1]})
ggplot(Output2,aes(x="t",y="N1"))+geom_line()+geom_line(Output2,aes(x="t",y="N2"),color='red')+theme_classic()

# a12>a11, a21>a22
params3=(0.5,0.5,5,10,4,8)

modelSim3=spint.odeint(func=Sim,y0=Nint,t=times,args=params3)
Output3=pd.DataFrame({"t":times,"N1":modelSim3[:,0],"N2":modelSim3[:,1]})
ggplot(Output3,aes(x="t",y="N1"))+geom_line()+geom_line(Output3,aes(x="t",y="N2"),color='red')+theme_classic()

# a12>a11, a21<a22
params4=(0.5,0.5,5,10,8,4)

modelSim4=spint.odeint(func=Sim,y0=Nint,t=times,args=params4)
Output4=pd.DataFrame({"t":times,"N1":modelSim4[:,0],"N2":modelSim4[:,1]})
ggplot(Output4,aes(x="t",y="N1"))+geom_line()+geom_line(Output4,aes(x="t",y="N2"),color='red')+theme_classic()
