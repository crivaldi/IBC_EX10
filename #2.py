# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:59:45 2018

@author: vysan
"""

import pandas as pd
import scipy
import scipy.integrate as spint
from plotnine import *

def comSim(y,t0,R1,R2,a11,a12,a21,a22):
    N1=y[0]
    N2=y[1]
    dN1dt=R1*(1-N1*a11-N2*a12)*N1
    dN2dt=R2*(1-N2*a22-N1*a21)*N2
    return ([dN1dt,dN2dt])
times=range(0,100)
N0=[0.01,0.01]

# Case A a12<a11 and a21<a22
paramsA=(.5,.5,.01,.005,.003,.01)
modelSimA=spint.odeint(func=comSim,y0=N0,t=times,args=paramsA)
modelOutputA=pd.DataFrame({"t":times,"N1":modelSimA[:,0],"N2":modelSimA[:,1]})
ggplot(modelOutputA,aes(x="t",y="N1"))+geom_line()+geom_line(modelOutputA,aes(x="t",y="N2"),color="red")+theme_classic()

#Case B a12>a11 and a21<a22
paramsB=(.5,.5,.01,.015,.003,.01)
modelSimB=spint.odeint(func=comSim,y0=N0,t=times,args=paramsB)
modelOutputB=pd.DataFrame({"t":times,"N1":modelSimB[:,0],"N2":modelSimB[:,1]})
ggplot(modelOutputB,aes(x="t",y="N1"))+geom_line()+geom_line(modelOutputB,aes(x="t",y="N2"),color="red")+theme_classic() 

#Case C a12<a11 and a21>a22
paramsC=(.5,.5,.01,.005,.015,.003)
modelSimC=spint.odeint(func=comSim,y0=N0,t=times,args=paramsC)
modelOutputC=pd.DataFrame({"t":times,"N1":modelSimC[:,0],"N2":modelSimC[:,1]})
ggplot(modelOutputC,aes(x="t",y="N1"))+geom_line()+geom_line(modelOutputC,aes(x="t",y="N2"),color="red")+theme_classic() 

#when a12<a11 and a21<a22 criteria is met, the plot show that there is coexistence between the two species 
#when one portion of the criteria is not met, one of the species does not survive and dies off 
