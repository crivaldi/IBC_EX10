# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:12:58 2018

Exercise10 part 1: Use a likelihood ratio test to compare two models where one
model is a subset of the other model. 

This code specifically evaluates whether a quadratic or a linear model is more 
appropriate for the data in data.txt

@author: Patricia
"""

import numpy as np
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

def linear(p, obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

def quadratic(p, obs):
    B0=p[0]
    B1=p[1]
    B2=p[3]
    sigma=p[4]
    expected=B0+B1*obs.x+B2*(np.exp(obs.x,2))
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

data=pandas.read_csv("data.txt", sep=",")

initialGuess=numpy.array([1,1,1])
fitlin=minimize(linear,initialGuess,method="Nelder-Mead",options={'disp': True},args=data)
fitquad=minimize(quadratic,initialGuess,method="Nelder-Mead",options={'disp': True},args=data)
print("negative log likelihood for linear model is: ")
print(fitlin.x)
print("negative log likelihood for quadratic model is: ")
print(fitquad.x)
print(ggplot(data,aes(x='x',y='y'))+geom_point()+theme_classic())