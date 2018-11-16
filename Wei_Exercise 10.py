# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

myList = ["Alyssa", "Laura", "Tracie", "Alicia", "Connor"]
myList[0]
myList[1]

import numpy as np
import pandas as pd
#import scipy
import scipy.integrate as spint
#import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import norm
#from plotnine import *
from scipy import stats

#from scipy.stats import chi2
#from sklearn.linear_model import SGDClassifier
#from sklearn.metrics import log_loss

####Problem 1####
data = pd.read_csv("data.txt",sep = ',')
def nllike(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    
    expected=B0+B1*obs.x
    nll1=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll1

def nllike(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[2]
    
    expected=B0+B1*obs.x+B2*obs.x**2
    nll2=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll2

def stat():
    initialGuess1=np.array([1,1,1])
    fit1=minimize(nllike,initialGuess1,method="Nelder-Mead",args=data)
    initialGuess2=np.array([1,1,1,1])
    fit2=minimize(nllike,initialGuess2,method="Nelder-Mead",args=data)
    
    teststat = 2*(fit1.fun-fit2.fun)
    why=len(fit2.x)-len(fit1.x)

    if 1-stats.chi2.cdf(teststat,why)>0.05:
        return 'Linear fits better fits better'
    else:
        print 'Quadratic fits better'
        
print stat()


####Problem 2####
def ddSim(y,t0,r,K):
    N=y[0]
    dY = N*r*(1-N/K)
    return [dY]

params=(0.1,10)
y=[0.01]
times=range(0,600)

log_growth = spint.odeint(func=ddSim,y0=y,t=times,args=params)

modelOutput=pd.DataFrame({"t":times,"N":log_growth[:,0]})

ggplot(modelOutput,aes(x="t",y="N"))+geom_line()

# Challenge - Gatenby and Vincent
#matrix format [a11 a12;a21 a22] = [a b;c d]
def tumorSim(y,t0,RN,a,b,RT,c,d):
    N=y[0]
    T=y[1]
    
    dNdt=RN*(1-N*a-T*b)
    dTdt=RT*(1-T*d+N*c)
    return [dNdt,dTdt]

# Challenge - Gatenby and Vincent
#matrix format [a11 a12;a21 a22] = [a b;c d]
def tumorSim(y,t0,RN,a,b,RT,c,d):
    N=y[0]
    T=y[1]
    
    dNdt=RN*(1-N*a-T*b)
    dTdt=RT*(1-T*d-N*c)
    return [dNdt,dTdt]

# case b<a and c>d
times=range(0,100)
y0=[0.1,0.1]
params=(0.5,10,0.5,0.5,10,0.5)
sim=spint.odeint(func=tumorSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"normal":sim[:,0],"tumor":sim[:,1]})
p = ggplot(simDF,aes(x="t",y="normal"))+geom_line()+geom_line(simDF,aes(x="t",y="tumor"),color='red')+theme_classic()
p.draw()

# case b>a and c<d
times=range(0,100)
y0=[0.05,0.2]
params=(0.5,0.5,10,0.5,0.5,10)
sim=spint.odeint(func=tumorSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"normal":sim[:,0],"tumor":sim[:,1]})
g=ggplot(simDF,aes(x="t",y="normal"))+geom_line()+geom_line(simDF,aes(x="t",y="tumor"),color='red')+theme_classic()
g.draw()

# case b>a and c>d
times=range(0,100)
y0=[0.1,0.1]
params=(0.5,0.5,10,0.5,10,0.5)
sim=spint.odeint(func=tumorSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"normal":sim[:,0],"tumor":sim[:,1]})
m=ggplot(simDF,aes(x="t",y="normal"))+geom_line()+geom_line(simDF,aes(x="t",y="tumor"),color='red')+theme_classic()
m.draw()