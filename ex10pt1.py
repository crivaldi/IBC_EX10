# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 17:12:58 2018

Exercise10 part 1: Use a likelihood ratio test to compare two models where one
model is a subset of the other model. 

This code specifically evaluates whether a quadratic or a linear model is more 
appropriate for the data in data.txt

@author: Patricia
"""

import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

def linear(p, obs):
    B0=p[0] # y-int
    B1=p[1] # slope
    sigma=p[2] # variance
    expected=B0+B1*obs.x # linear equn
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum() # neg log likelihood
    return nll

def quadratic(p, obs):
    B0=p[0] # y-int
    B1=p[1] 
    B2=p[2]
    sigma=p[3]
    expected=B0+B1*obs.x+B2*(obs.x ** 2)
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

data=pandas.read_csv("data.txt", sep=",")

linGuess=numpy.array([12,12,1]) # random parameters
quadGuess=numpy.array([12,12,8,4])
fitlin=minimize(linear,linGuess,method="Nelder-Mead",options={'disp': True},args=data) # call fxns with args
fitquad=minimize(quadratic,quadGuess,method="Nelder-Mead",options={'disp': True},args=data)
print("Most likely parameters for linear model are: ")
print(fitlin.x)
print("Most likely parameters for quadratic are: ")
print(fitquad.x)
print(ggplot(data,aes(x='x',y='y'))+geom_point()+theme_classic())
# quadratic sigma is lower (less variance) and therefore fits the data better
if (fitlin.x[2] > fitquad.x[3]):
    print("The quadratic model suits the data better")
# linear sigma is lower (less variance) and therefore fits the data better
elif (fitlin.x[2] < fitquad.x[3]):
    print("The linear model suits the data better")