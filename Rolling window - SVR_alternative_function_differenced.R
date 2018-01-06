
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


functionalizeListCreation <- function(n.of.lags, n.of.iters) {
  svrPerfFeat <- list(a, b, c)
  svrPredFeat <- list()
  svrParamsFeat <- list()
  
  # starting parallelization
  parallelStartSocket(3) 
  
  for(i in 1:n.of.iters){  # loop for training the actual model
    # n.of.lags lags used
    # moving training data set
    svmTrainOrigResp <- ts2[(24 + (i-1)*168):(528 + (i-1)*168)]
    svmTrainResp <- diff(ts2[(24 + (i-1)*168):(528 + (i-1)*168)], 1, 1) # response in training
    svmTrainOrigPred <- data.frame()
    svmTrainPred <- data.frame()
    for (j in 0:504){
      auxTrainOrigPred <- c(ts2[((23 - n.of.lags + 1) + (i-1)*168 + j):(23 + (i-1)*168 + j)]) # predictors in training
      svmTrainOrigPred <- rbind(svmTrainOrigPred, auxTrainOrigPred) # binding all predictors
    }
    for (j in 0:503){
      auxTrainPred <- c(diff(ts2[((23 - n.of.lags) + (i-1)*168 + j):(23 + (i-1)*168 + j)], 1, 1)) # predictors in training
      svmTrainPred <- rbind(svmTrainPred, auxTrainPred) # binding all predictors
    }
    svmTrain <- cbind(svmTrainPred, svmTrainResp) # finished data set
    svmTrainOrig <- cbind(svmTrainOrigPred, svmTrainOrigResp)
    colnames(svmTrain) <- c(paste("lag", n.of.lags:1,sep = ""), "response")
    colnames(svmTrainOrig) <- c(paste("lag", n.of.lags:1,sep = ""), "response")
    
    # moving testing data set
    svmTestOrigResp <- ts2[(528 + (i-1)*168):(696 + (i-1)*168)]
    svmTestResp <- diff(ts2[(528 + (i-1)*168):(696 + (i-1)*168)], 1, 1) # response in testing
    svmTestOrigPred <- data.frame()
    svmTestPred <- data.frame()
    for (j in 0:168){
      auxTestOrigPred <- c(ts2[((527 - n.of.lags + 1) + (i-1)*168 + j):(527 + (i-1)*168 + j)]) # predictors
      svmTestOrigPred <- rbind(svmTestOrigPred, auxTestOrigPred) # binding all predictors
    }
    for (j in 0:167){
      auxTestPred <- c(diff(ts2[((527 - n.of.lags) + (i-1)*168 + j):(527 + (i-1)*168 + j)], 1, 1)) # predictors
      svmTestPred <- rbind(svmTestPred, auxTestPred) # binding all predictors
    }
    svmTestOrig <- cbind(svmTestOrigPred, svmTestOrigResp)
    svmTest <- cbind(svmTestPred, svmTestResp) # finished data set
    colnames(svmTest) <- c(paste("lag", n.of.lags:1,sep = ""), "response")
    colnames(svmTestOrig) <- c(paste("lag", n.of.lags:1,sep = ""), "response")
    
    # parameter optimization
    regrTr1 <- makeRegrTask(data = svmTrain, target = "response")
    svmTune <- tuneParamsMultiCrit("regr.ksvm", task = regrTr1, resampling = rdescAlt, par.set = psAlt,
                                   measures = list(mae, rmse), control = ctrlAlt, show.info = FALSE)
    
    # optimized model learning
    optParams <- vector()
    for(k in 1:length(svmTune$x)){ # get the best set in case of multiple optima 
      optParams[k] <- svmTune$x[[k]]$C # high C tends to screw up predictions a bit
    }
    svrParamsFeat[[i]] <- svmTune$x[[which.min(optParams)]]
    lrnFilter2 <- setHyperPars(makeLearner("regr.ksvm"), par.vals = svmTune$x[[which.min(optParams)]])
    modelFinal <- train(lrnFilter2, task = regrTr1)
    
    # predictions
    regrTe1 <- makeRegrTask(data = svmTest, target = "response")
    predSVM <- predict(modelFinal, task = regrTe1)
    
    # measures
    svrPredFeat[[i]] <- c(svmTestOrig$response[1:168] + predSVM$data$response)
    prediction <- c(svmTestOrig$response[1:168] + predSVM$data$response)
    orig <- svmTestOrig$response[2:169]
    
    
    svrPerfFeat[[1]][i] <- sum(abs((orig - prediction)))/ length(orig)
    svrPerfFeat[[2]][i] <- sum(abs((orig - prediction) / orig)) / length(orig)
    svrPerfFeat[[3]][i] <- sqrt(sum((orig - prediction)^2) / length(orig))
    print(i)
  }
  
  
  result<-list(n.of.lags, svrPerfFeat, svrPredFeat, svrParamsFeat)
  
  # stopping parallelization
  parallelStop()
  
  
  return(result)
  
}

diffLag2 <- functionalizeListCreation(2, 100)
diffLag3 <- functionalizeListCreation(3, 100)
diffLag4 <- functionalizeListCreation(4, 100)
diffLag5 <- functionalizeListCreation(5, 100)

