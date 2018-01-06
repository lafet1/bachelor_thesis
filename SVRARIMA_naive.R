
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


functionSVRARIMA <- function(lags, iters) {
  trainResiduals <- list()
  testResiduals <- list()
  fittedSVRARIMA <- list()
  svrarimaPerfFeat <- list(a, b, c)
  
  for(i in 1:iters){
    # moving training data set
    svmTrainResp <- ts2[(25 + (i-1)*168):(528 + (i-1)*168)] # response in training
    svmTrainPred <- data.frame()
    for (j in 0:503){
      auxTrainPred <- c(ts2[((24 - lags + 1) + (i-1)*168 + j):(24 + (i-1)*168 + j)]) # predictors in training
      svmTrainPred <- rbind(svmTrainPred, auxTrainPred) # binding all predictors
    }
    svmTrain <- cbind(svmTrainPred, svmTrainResp) # finished data set
    colnames(svmTrain) <- c(paste("lag", lags:1,sep = ""), "response")
    
    # moving testing data set
    svmTestResp <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)] # response in testing
    svmTestPred <- data.frame()
    for (j in 0:167){
      auxTestPred <- c(ts2[((528 - lags + 1) + (i-1)*168 + j):(528 + (i-1)*168 + j)]) # predictors
      svmTestPred <- rbind(svmTestPred, auxTestPred) # binding all predictors
    }  
    svmTest <- cbind(svmTestPred, svmTestResp) # finished data set
    colnames(svmTest) <- c(paste("lag", lags:1,sep = ""), "response")
    
    # model training - SVM + ARIMA
    naiveModel <- ksvm(response~., data = svmTrain, kernel = "rbfdot",  
                       C = 1, epsilon = 0.1) # training the SVR model
    
    fitted <- as.numeric((naiveModel@fitted[,1] * naiveModel@scaling$y.scale$`scaled:scale`) 
                         + naiveModel@scaling$y.scale$`scaled:center`) # fitted values from trained model
    
    # creating test set - SVM
    trainResiduals[[i]] <- svmTrainResp - fitted # train set
    svrarimaModel <- auto.arima(trainResiduals[[i]]) # trains model
    
    # applying to test set
    predSVM <- predict(naiveModel, newdata = svmTest, type = "response")
    testResiduals[[i]] <- svmTestResp - predSVM # test set
    svrarimaModel1 <- Arima(ts(testResiduals[[i]], frequency = 24), model = svrarimaModel) # applying model
    
    # predictions
    fc1 <- fitted(svrarimaModel1) # getting fitted values for ARIMA
    fittedSVRARIMA[[i]] <- as.numeric(fc1 + predSVM) # getting fitted values for SVRARIMA
    
    # performance
    svrarimaPerfFeat[[1]][i] <- sum(abs((svmTestResp - fittedSVRARIMA[[i]])))/ length(svmTestResp)
    svrarimaPerfFeat[[2]][i] <- sum(abs((svmTestResp - fittedSVRARIMA[[i]]) 
                                        / svmTestResp)) / length(svmTestResp)
    svrarimaPerfFeat[[3]][i] <- sqrt(sum((svmTestResp - fittedSVRARIMA[[i]])^2) / length(svmTestResp))
    print(i)
  }
  
  
  result <- list(lags, fittedSVRARIMA, svrarimaPerfFeat, trainResiduals, testResiduals)
  
  
  return(result)
  
}


naiveSVRARIMA2 <- functionSVRARIMA(2, 100)
naiveSVRARIMA3 <- functionSVRARIMA(3, 100)
naiveSVRARIMA4 <- functionSVRARIMA(4, 100)
naiveSVRARIMA5 <- functionSVRARIMA(5, 100)
