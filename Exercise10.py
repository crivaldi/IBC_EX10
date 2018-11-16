#Number 1
pwd
cd IBC_EX10

import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
import scipy.integrate as spint
from scipy.stats import norm
from scipy.stats import chi2
from scipy import stats
from plotnine import *
! pip install scikit-misc

data = pd.read_table("data.txt", sep=",")

def quadSim(p,data):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    expected=B0+B1*data.x+B2*data.x**2
    nll=-1*norm(expected,sigma).logpdf(data.y).sum()
    return nll

def linSim(p,data):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*data.x
    nll=-1*norm(expected,sigma).logpdf(data.y).sum()
    return nll

quadGuess=np.array([1,1,1,1])
linGuess=np.array([1,1,1])

fitquad= minimize (quadSim,quadGuess, method="Nelder-Mead",options={'disp': True},args=data)
fitlin= minimize (linSim,linGuess,method="Nelder-Mead",options={'disp': True},args=data)

teststat=2*(fitlin.fun-fitquad.fun)
df=len(fitquad.x)-len(fitlin.x)
1-stats.chi2.cdf(teststat,df)



#Number 2
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
import scipy.integrate as spint
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import stats
from plotnine import *
! pip install scikit-misc

def compSim(y,t0,R1,R2,a11,a22,a21,a12):
# "unpack" lists containing state variables (y)
    N1=y[0]
    N2=y[1]
# calculate change in state variables with time, give parameter values and current value of state variables
    dN1dt=R1*(1-N1*a11-N2*a12)*N1
    dN2dt=R2*(1-N2*a22-N1*a21)*N2
# return list containing change in state variables with time
    return [dN1dt, dN2dt]

#Define parameters, initial values for state variables, and time steps
#case 1 all a's are equal
params=(0.5,0.5,2,2,2,2)
y0=[0.1,0.1]
times=range(0,100)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"eq1":sim[:,0],"eq2":sim[:,1]})
ggplot(simDF,aes(x="t",y="eq1"))+geom_line()+geom_line(simDF,aes(x="t",y="eq2"),color='red')+theme_classic()

#case 2 a22 and a11 are greater than a12 and a21 
params=(0.5,0.5,2,1,1,0.5)
y0=[0.1,0.1]
times=range(0,100)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"eq1":sim[:,0],"eq2":sim[:,1]})
ggplot(simDF,aes(x="t",y="eq1"))+geom_line()+geom_line(simDF,aes(x="t",y="eq2"),color='red')+theme_classic()

#case 3 a22 and a11 are less than a12 and a21 
params=(0.5,0.5,0.5,1,1,2)
y0=[0.05,0.1]
times=range(0,100)
sim=spint.odeint(func=compSim,y0=y0,t=times,args=params)
simDF=pd.DataFrame({"t":times,"eq1":sim[:,0],"eq2":sim[:,1]})
ggplot(simDF,aes(x="t",y="eq1"))+geom_line()+geom_line(simDF,aes(x="t",y="eq2"),color='red')+theme_classic()


