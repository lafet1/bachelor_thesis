
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


functionalize.list.creation <- function(n.of.lags, n.of.iters) {
  svrPerfFeat <- list(a, b, c)
  svrPredFeat <- list()
  svrParamsFeat <- list()
  
  # starting parallelization
  parallelStartSocket(2) 
  
  for(i in 1:n.of.iters){  # loop for training the actual model
    # n.of.lags lags used
    # moving training data set
    svmTrainResp <- ts2[(25 + (i-1)*168):(528 + (i-1)*168)] # response in training
    svmTrainPred <- data.frame()
    for (j in 0:503){
      auxTrainPred <- c(ts2[((24 - n.of.lags + 1) + (i-1)*168 + j):(24 + (i-1)*168 + j)]) # predictors in training
      svmTrainPred <- rbind(svmTrainPred, auxTrainPred) # binding all predictors
    }
    svmTrain <- cbind(svmTrainPred, svmTrainResp) # finished data set
    colnames(svmTrain) <- c(paste("lag", n.of.lags:1,sep = ""), "response")
    
    # moving testing data set
    svmTestResp <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)] # response in testing
    svmTestPred <- data.frame()
    for (j in 0:167){
      auxTestPred <- c(ts2[((528 - n.of.lags + 1) + (i-1)*168 + j):(528 + (i-1)*168 + j)]) # predictors
      svmTestPred <- rbind(svmTestPred, auxTestPred) # binding all predictors
    }  
    svmTest <- cbind(svmTestPred, svmTestResp) # finished data set
    colnames(svmTest) <- c(paste("lag", n.of.lags:1,sep = ""), "response")
    
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
    svrPredFeat[[i]] <- predSVM
    
    
    svrPerfFeat[[1]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response)))/ length(predSVM$data$truth)
    svrPerfFeat[[2]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response) 
                                   / predSVM$data$truth)) / length(predSVM$data$truth)
    svrPerfFeat[[3]][i] <- sqrt(sum((predSVM$data$truth - predSVM$data$response)^2) / length(predSVM$data$truth))
    print(i)
  }
  
  
  result<-list(n.of.lags, svrPerfFeat, svrPredFeat, svrParamsFeat)
  
  # stopping parallelization
  parallelStop()
  
  
  return(result)
  
}


lag3 <- functionalize.list.creation(3, 100)
lag4 <- functionalize.list.creation(4, 100)
lag5 <- functionalize.list.creation(5, 100)

