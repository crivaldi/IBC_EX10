# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 01:26:41 2018

@author: Alexis
"""
#imports library
import pandas
import scipy
import numpy as np
import scipy.integrate as spint
from plotnine import *
#function to call to plot the differential equations
def Scenerios(alpha11,alpha12,alpha21,alpha22):
    def ddSim(y,t0,alpha11,alpha12,alpha21,alpha22,R1,R2):#function inside the function

        N1=y[0]
        N2=y[1]
        dN1dt=(R1*(1-N1*alpha11-N2*alpha12)*N1)#Differential equations
        dN2dt=(R2*(1-N2*alpha22-N1*alpha21)*N2)
    

        return np.array([dN1dt, dN2dt])

    params=(alpha11,alpha12,alpha21,alpha22,.1,.1)#parameter needed for function
    N0=[1,1]#initial conditions
    times=range(0,5000)#time paramater

    modelSim=spint.odeint(func=ddSim,y0=N0,t=times,args=params)#intergrates the equations
### put model output in a dataframe for plotting purposes
    modelOutput=pandas.DataFrame({"t":times,"N1":modelSim[:,0],"N2":modelSim[:,1]})#Puts in df
### plot simulation output
    a=ggplot(modelOutput,aes(x="t",y="N1"))+geom_line(color='blue')+theme_classic()#plots
    a=a+geom_line(aes(x="t",y="N2"),color='red')
    a=a+xlab("Time")+ylab("Population")+ggtitle('alpha11=%.2f alpha12=%.2f alpha21=%.2f alpha22=%.2f'%(alpha11,alpha12,alpha21,alpha22))
    print(a)
    return 0
#Scenerios
Scenerios(.07,.02,.01,.09)#Scenerio where criteria for coexistance is met
Scenerios(.01,.07,.01,.09)#Criteria not met, alpha11 us less than alpha12 (N1 dominates)
Scenerios(.07,.02,.05,.01)#Criteria not met,alpha22 is less than alpha21(N2 dominates)
#The model simulation shows that the coexistence criteria is valid as when it is not met 
#one of the species goes extinct.