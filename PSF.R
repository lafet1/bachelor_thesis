
# rm(list=ls())

library(githubinstall) # library for getting the under-development part of mlr package
# gh_install_packages("mlr-org/mlr", ref = "forecasting")
# installation of a package augmentation -- cross-validation for time-series 

library(RCurl) # data download from ftp server
library(dplyr)
library(kernlab) # SVR via ksvm() - caret does not support svm() with RBF kerneù
library(forecast) # ARIMA via auto.arima()
library(tseries)
library(PSF) # PSF via psf()
library(mlr) # train() for optimalization
library(emoa) # required by mlr package
library(FSelector)
library(parallelMap)

set.seed(100)

functionPSF <- function(periods){
  psfPerf <- list(a, b, c)
  psfPred <- list(a, b, c)
  auxPred <- list()
  
  #the loop itself
  for(i in 1:periods){
    psfTrain <- ts(ts3[1:(9264 + (i-1)*168)], frequency = 24) # moves training data by week
    psfTest <- ts(ts3[(9265 + (i-1)*168):(9264 + i*168)], frequency = 24) # moves test set by week
    
    for(j in 1:7){ # for loop to forecast on day-by-day basis
      if(j == 1){
        auxPred[[j]] <- psf(psfTrain, k = seq(2, 10), w = seq(1, 10), cycle = 24, n.ahead = 24)
      }
      else{
        auxPred[[j]] <- psf(c(psfTrain, psfTest[1:24*(j - 1)]), k = seq(2, 10), w = seq(1, 10), cycle = 24, n.ahead = 24)
      }
    }
    
    auxCombPred <- vector()
    auxK <- vector()
    auxW <- vector()
    
    for(k in 1:7){ # saving the predictions and the parameters
      auxCombPred <- c(auxCombPred, unlist(auxPred[[k]]$predictions))
      auxK <- c(auxK, unlist(auxPred[[k]]$k))
      auxW <- c(auxW, unlist(auxPred[[k]]$w))
    }
    
    psfPred[[1]][[i]] <- auxCombPred
    psfPred[[2]][[i]] <- auxK
    psfPred[[3]][[i]] <- auxW
    
    psfPerf[[1]][i] <- sum(abs(psfTest - psfPred[[1]][[i]]))/ length(psfTest) # MAE
    psfPerf[[2]][i] <- sum(abs((psfTest - psfPred[[1]][[i]])/ psfTest))/ length(psfTest)  # MAPE
    psfPerf[[3]][i] <- sqrt(sum((psfTest - psfPred[[1]][[i]])^2)/ length(psfTest)) # RMSE
    print(i)
  }
  
  result <- list(psfPerf, psfPred)
  
  return(result)
}

PSF <- functionPSF(100)


