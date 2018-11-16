# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:57:19 2018

Exercise10 part 2: Use Lotka and Volterra competition model to demonstrate
the validity of alpha12 < alpha21 and alpha21 < alpha22 in three different 
model simulations

@author: Patricia
"""

import pandas
import scipy
import scipy.integrate as spint
from plotnine import *

# Challenge - Lotka Volterra
def LVSim(y,t0,R1,R2,a11,a12,a22,a21):
    N1=y[0]
    N2=y[1]
    # R1 rate of growth of prey population, R2 rate of growth for predator
    # N1 initial prey population, N2 initial predator population
    dN1dt=R1*(1-N1*a11-N2*a12)*N1
    dN2dt=R2*(1-N2*a22-N1*a21)*N2
    
    return [dN1dt,dN2dt]


# case 1
# All the criteria are met
times=range(1,100)
y0=[0.5,0.1]
parameters=(0.5,0.5,0.6,0.4,0.5,0.3)
sim=spint.odeint(func=LVSim,y0=y0,t=times,args=parameters)
simDF=pandas.DataFrame({"t":times,"prey":sim[:,0],"predator":sim[:,1]})
print(ggplot(simDF,aes(x="t",y="prey"))+geom_line()+geom_line(simDF,aes(x="t",y="predator"),color="red")+theme_classic())
