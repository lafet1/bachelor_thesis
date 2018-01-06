
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

# list with measures
a <- c()
b <- c()
c <- c()

# arimaPerf <- list(a, b, c)
# arimaPred <- list()


# the loop itself - AIC
for(i in 1:100){
  arimaTrain <- ts(ts1[(1 + (i-1)*168):(504 + (i-1)*168)], frequency = 24) # moves training data by week
  arimaTest <- ts(ts1[(505 + (i-1)*168):(672 + (i-1)*168)], frequency = 24) # moves test data by week
  arimaModel <- auto.arima(arimaTrain) # trains model
  arimaModel1 <- Arima(arimaTest, model = arimaModel) # applies train model to test data
  fc1 <- fitted(arimaModel1) # generates the predictions
  arimaPred[[i]] <- fc1
  arimaPerf[[1]][i] <- sum(abs(arimaTest - fc1))/ length(arimaTest) # MAE
  arimaPerf[[2]][i] <- sum(abs((arimaTest - fc1)/ arimaTest))/ length(arimaTest)  # MAPE
  arimaPerf[[3]][i] <- sqrt(sum((arimaTest - fc1)^2)/ length(arimaTest)) # RMSE
  print(i)
}

# arimaPerf1 <- list(a, b, c)
# arimaPred1 <- list()

# the loop itself - BIC
for(i in 1:100){
  arimaTrain1 <- ts(ts1[(1 + (i-1)*168):(504 + (i-1)*168)], frequency = 24) # moves training data by week
  arimaTest1 <- ts(ts1[(505 + (i-1)*168):(672 + (i-1)*168)], frequency = 24) # moves test data by week
  arimaModel <- auto.arima(arimaTrain1, ic = "bic") # trains model
  arimaModel1 <- Arima(arimaTest1, model = arimaModel) # applies train model to test data
  fc1 <- fitted(arimaModel1) # generates the predictions
  arimaPred1[[i]] <- fc1
  arimaPerf1[[1]][i] <- sum(abs(arimaTest1 - fc1))/ length(arimaTest1) # MAE
  arimaPerf1[[2]][i] <- sum(abs((arimaTest1 - fc1)/ arimaTest1))/ length(arimaTest1)  # MAPE
  arimaPerf1[[3]][i] <- sqrt(sum((arimaTest1 - fc1)^2)/ length(arimaTest1)) # RMSE
  print(i)
}

