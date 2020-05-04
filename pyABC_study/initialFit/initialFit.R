# ref: https://www.r-bloggers.com/learning-r-parameter-fitting-for-models-involving-differential-equations/


library(ggplot2) 
library(reshape2) # library for reshaping data (tall-narrow <-> short-wide)
library(deSolve) # library for solving differential equations
library(minpack.lm) 

setwd("~/OneDrive/PostgraduateProject/dataPreProcess")

# Import data

odeDF = read.csv("../data/iData.csv",header = F)
colnames(odeDF) = c("t","N","M","B","A")

temp = melt(odeDF, id.vars=c("t"))

ggplot(data = temp,aes(x=t, y=value, color=variable))+
  geom_line()+
  geom_point()


# build math model

dynamicSys= function(t,var,para){
  eqns=rep(0,4)
  eqns[1] = para$lambdaN + para$kNB*var["B"] - para$muN*var["N"] - para$vNM*var["N"]*var["M"]
  eqns[2] = para$lambdaM + para$kMB*var["B"] - para$muM*var["M"]
  eqns[3] = (para$sBN*var["N"])/(1+para$iBM)-para$muB*var["B"]
  eqns[4] = para$sAM*var["M"] - para$muA*var["A"]
  
  return(list(eqns))
}


# Test the ODE solver

varInit = c(N=1.8500000,M=0.3333333, B=1.995670, A=0.665976)
time = odeDF$t
para = list(lambdaN = 1, kNB = 2, muN = 1, vNM = 1,
            lambdaM = 1, kMB = 2, muM = 1,
            sBN = 1, iBM = 1, muB = 1,
            sAM = 1, muA = 1)

ODEtest = ode(y = varInit, times = time, func = dynamicSys, parms = para)
ODEtest = as.data.frame(ODEtest)
ODEtemp = melt(ODEtest, id.var = c("time"))

# Build the residual function for lm fitting

ODEssq = function(para) {
  varInit = c(N=1.8500000,M=0.3333333, B=1.995670, A=0.665976)
  time = odeDF$t
  
  ODEout = ode(y = varInit, times = time, func = dynamicSys, parms = para)
  ODEtemp = melt(as.data.frame(ODEout), id.var = c("time"))
  return(ODEtemp$value-temp$value)
}

ODEssq(para)


# Fit the function

para = list(lambdaN = 1, kNB = 1, muN = 1, vNM = 1,
            lambdaM = 1, kMB = 1, muM = 1,
            sBN = 1, iBM = 1, muB = 1,
            sAM = 1, muA = 1)

ODEfit = nls.lm(par = para, fn = ODEssq)
summary(ODEfit)
ODEfit[["par"]]
vcov(ODEfit)


# Plot fit prediction

predPara = ODEfit[["par"]]
varInit = c(N=1.8500000,M=0.3333333, B=1.995670, A=0.665976)
ODEtest = ode(y = varInit, times = time, func = dynamicSys, parms = predPara)
ODEtest = as.data.frame(ODEtest)
ODEtemp = melt(ODEtest, id.var = c("time"))
# test_var = as.character(ODEtemp$variable)
ODEtemp$variable = paste0(as.character(ODEtemp$variable), rep("_EST",36), collapse = NULL)

ggplot(data = ODEtemp[c(1:18),c(1:3)], aes(x=time, y=value, color=variable))+
  # geom_smooth(se= F, method = "gam")+
  geom_point()+
  geom_line()+
  geom_point(data = temp[c(1:18),c(1:3)], aes(x=t, y=value, color=variable))


ggplot(data = ODEtemp[c(19:36),c(1:3)], aes(x=time, y=value, color=variable))+
  # geom_smooth(se= F, method = "gam")+
  geom_point()+
  geom_line()+
  geom_point(data = temp[c(19:36),c(1:3)], aes(x=t, y=value, color=variable))

write.csv(ODEtemp, file = "lmFittingData.csv")
