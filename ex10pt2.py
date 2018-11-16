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
    # N1 initial prey population, N2 initial predator population
    dN1dt=R1*(1-N1*a11-N2*a12)*N1
    dN2dt=R2*(1-N2*a22-N1*a21)*N2
    
    return [dN1dt,dN2dt]


# case 1
# All the criteria are met
times=range(1,50)
y0=[0.1,0.1]
parameters=(0.5,0.5,0.6,0.4,0.5,0.3)
sim=spint.odeint(func=LVSim,y0=y0,t=times,args=parameters)
simDF=pandas.DataFrame({"t":times,"prey":sim[:,0],"predator":sim[:,1]})
print(ggplot(simDF,aes(x="t",y="prey"))+geom_line()+geom_line(simDF,aes(x="t",y="predator"),color="red")+theme_classic())
# graph shows both predator and prey with healthy populations

# case 2
# a12 > a11 -> the criteria are not met
times=range(1,100)
y0=[0.1,0.1]
parameters=(0.5,0.5,0.6,0.8,0.5,0.3)
sim=spint.odeint(func=LVSim,y0=y0,t=times,args=parameters)
simDF=pandas.DataFrame({"t":times,"prey":sim[:,0],"predator":sim[:,1]})
print(ggplot(simDF,aes(x="t",y="prey"))+geom_line()+geom_line(simDF,aes(x="t",y="predator"),color="red")+theme_classic())
# predator population outcompetes prey when a12 > a11 but all other criteria are met

# case 3
# a21 > a22 -> the criteria are not met
times=range(1,100)
y0=[0.2,0.1]
parameters=(0.5,0.5,0.6,0.4,0.5,0.6)
sim=spint.odeint(func=LVSim,y0=y0,t=times,args=parameters)
simDF=pandas.DataFrame({"t":times,"prey":sim[:,0],"predator":sim[:,1]})
print(ggplot(simDF,aes(x="t",y="prey"))+geom_line()+geom_line(simDF,aes(x="t",y="predator"),color="red")+theme_classic())
# prey population outcompetes predator when a21 > a22 but all other criteria are met

# case 4
# a21 > a22 -> the criteria are not met AND
# a12 > a11 -> the criteria are not met
times=range(1,100)
y0=[0.1,0.1]
parameters=(0.5,0.5,0.6,0.8,0.5,0.8)
sim=spint.odeint(func=LVSim,y0=y0,t=times,args=parameters)
simDF=pandas.DataFrame({"t":times,"prey":sim[:,0],"predator":sim[:,1]})
print(ggplot(simDF,aes(x="t",y="prey"))+geom_line()+geom_line(simDF,aes(x="t",y="predator"),color="red")+theme_classic())
# with these particular parameters, the predator outcompetes the prey
'''
In conclusion, it seems that Lotka and Volterra were not lying when they stated
their criteria for coexistence of the predator and prey as case 1 was the only
case that allowed for coexistence.
'''