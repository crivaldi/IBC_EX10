# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:40:31 2018

@author: Seth
"""

#import packages

import pandas
import scipy
import scipy.integrate as spint
from plotnine import *

#define model of competitive species interaction

def popsim(y,t0,RN,KN,aTN,RT,KT,aNT):
    N=y[0]
    T=y[1]
    dNdt=RN*(1-(N+aNT*T)/KN)*N
    dTdt=RT*(1-(T+aTN*N)/KT)*T
    return [dNdt,dTdt]

#a superior competitor will extirpate a more abundant competitor over time

times=range(0,100)
y0 = [1, 9]
params = (0.5, 10, 2, 0.5, 10, 0.5)
sim = spint.odeint(func = popsim, y0 = y0, t = times, args = params)
simdata = pandas.DataFrame({"t":times, "1":sim[:,0], "2":sim[:,1]})
ggplot(simdata, aes(x = "t", y = "1")) + geom_line() + geom_line(color = 'blue') + geom_line(simdata, aes(x = "t", y = "2")) + geom_line(color = 'salmon')

#If competition is not an effect, then populations remain constant

times=range(0,100)
y0 = [5, 5]
params = (0.5, 10, 1, 0.5, 10, 1)
sim = spint.odeint(func = popsim, y0 = y0, t = times, args = params)
simdata = pandas.DataFrame({"t":times, "1":sim[:,0], "2":sim[:,1]})
ggplot(simdata, aes(x = "t", y = "1")) + geom_line() + geom_line(color = 'blue') + geom_line(simdata, aes(x = "t", y = "2")) + geom_line(color = 'salmon')

#If neither species is more competitive or fecund, but one has a higher carrying capacity,
#the species with the higher carrying capacity will extirpate the other

times=range(0,100)
y0 = [1, 9]
params = (0.5, 10, 0.5, 0.5, 5, 0.5)
sim = spint.odeint(func = popsim, y0 = y0, t = times, args = params)
simdata = pandas.DataFrame({"t":times, "1":sim[:,0], "2":sim[:,1]})
ggplot(simdata, aes(x = "t", y = "1")) + geom_line() + geom_line(color = 'blue') + geom_line(simdata, aes(x = "t", y = "2")) + geom_line(color = 'salmon')
