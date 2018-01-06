
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


functionSVRARIMA <- function(lags, iters, dataset) {
  trainResiduals <- list()
  testResiduals <- list()
  fittedSVRARIMA <- list()
  svrarimaPerfFeat <- list(a, b, c)
  
  for(i in 1:iters){
    # n.of.lags lags used
    # moving training data set
    svmTrainOrigResp <- ts2[(24 + (i-1)*168):(528 + (i-1)*168)]
    svmTrainResp <- diff(ts2[(24 + (i-1)*168):(528 + (i-1)*168)], 1, 1) # response in training
    svmTrainOrigPred <- data.frame()
    svmTrainPred <- data.frame()
    for (j in 0:504){
      auxTrainOrigPred <- c(ts2[((23 - lags + 1) + (i-1)*168 + j):(23 + (i-1)*168 + j)]) # predictors in training
      svmTrainOrigPred <- rbind(svmTrainOrigPred, auxTrainOrigPred) # binding all predictors
    }
    for (j in 0:503){
      auxTrainPred <- c(diff(ts2[((23 - lags) + (i-1)*168 + j):(23 + (i-1)*168 + j)], 1, 1)) # predictors in training
      svmTrainPred <- rbind(svmTrainPred, auxTrainPred) # binding all predictors
    }
    svmTrain <- cbind(svmTrainPred, svmTrainResp) # finished data set
    svmTrainOrig <- cbind(svmTrainOrigPred, svmTrainOrigResp)
    colnames(svmTrain) <- c(paste("lag", lags:1,sep = ""), "response")
    colnames(svmTrainOrig) <- c(paste("lag", lags:1,sep = ""), "response")
    
    # model training - SVM + ARIMA
    tunedModel <- ksvm(response~., data = svmTrain, kernel = "rbfdot", kpar = list(sigma = dataset[[4]][[i]]$sigma),  
                       C = dataset[[4]][[i]]$C, epsilon = dataset[[4]][[i]]$epsilon) # training the SVR model
    fitted <- as.numeric((tunedModel@fitted[,1] * tunedModel@scaling$y.scale$`scaled:scale`) 
                         + tunedModel@scaling$y.scale$`scaled:center`) # fitted values from trained model
    trainResiduals[[i]] <- svmTrainOrigResp[2:505] - (fitted + svmTrainOrigResp[1:504]) # train set
    svrarimaModel <- auto.arima(ts(trainResiduals[[i]], frequency = 24)) # trains model
    
    # applying to test set
    testResiduals[[i]] <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)] - dataset[[3]][[i]] # test set
    svrarimaModel1 <- Arima(ts(testResiduals[[i]], frequency = 24), model = svrarimaModel) # applying model
    
    # predictions
    fc1 <- fitted(svrarimaModel1) # getting fitted values for ARIMA
    fittedSVRARIMA[[i]] <- as.numeric(fc1 + dataset[[3]][[i]]) # getting fitted values for SVRARIMA
    orig1 <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)]
    
    # performance
    svrarimaPerfFeat[[1]][i] <- sum(abs((orig1 - fittedSVRARIMA[[i]])))/ length(orig1)
    svrarimaPerfFeat[[2]][i] <- sum(abs((orig1 - fittedSVRARIMA[[i]]) / orig1)) / length(orig1)
    svrarimaPerfFeat[[3]][i] <- sqrt(sum((orig1 - fittedSVRARIMA[[i]])^2) / length(orig1))
    print(i)
  }
  
  
  result <- list(lags, fittedSVRARIMA, svrarimaPerfFeat, trainResiduals, testResiduals)
  
  
  return(result)
  
}


diffSVRARIMA2 <- functionSVRARIMA(2, 100, diffLag2)
diffSVRARIMA3 <- functionSVRARIMA(3, 100, diffLag3)
diffSVRARIMA4 <- functionSVRARIMA(4, 100, diffLag4)
diffSVRARIMA5 <- functionSVRARIMA(5, 100, diffLag5)

