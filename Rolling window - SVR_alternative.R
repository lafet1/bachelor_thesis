
# rm(list=ls())

library(githubinstall) # library for getting the under-development part of mlr package
# gh_install_packages("mlr-org/mlr", ref = "forecasting")
# installation of a package augmentation -- cross-validation for time-series 

library(RCurl)
library(dplyr)
library(kernlab) # SVR via ksvm() - caret does not support svm() with RBF kernel
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

#svrPerfFeat2 <- list(a, b, c)
#svrPredFeat2 <- list()
#svrParamsFeat2 <- list()

lag2 <- list(2, svrPerfFeat2, svrPredFeat2, svrParamsFeat2)


# for parameter tuning
psAlt <- makeParamSet(
  makeNumericParam("C", lower = -12, upper = 12, trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -12, upper = 12, trafo = function(x) 2^x),
  makeDiscreteParam("kernel", values = c("rbfdot")),
  makeNumericParam("epsilon", lower = -5, upper = 0, trafo = function(x) 10^x)
)
ctrlAlt <- makeTuneMultiCritControlRandom(maxit = 100L)
rdescAlt <- makeResampleDesc("FixedCV", initial.window = 0.5) # FixedCV is for time-series CV


# starting parallelization
parallelStartSocket(2) 

for(i in 1:100){  # loop for training the actual model
  # 2 lags used
    # moving training data set
    svmTrainResp <- ts2[(25 + (i-1)*168):(528 + (i-1)*168)] # response in training
    svmTrainPred <- data.frame()
    for (j in 0:503){
      auxTrainPred <- c(ts2[((24 - 2 + 1) + (i-1)*168 + j):(24 + (i-1)*168 + j)]) # predictors in training
      svmTrainPred <- rbind(svmTrainPred, auxTrainPred) # binding all predictors
    }
    svmTrain <- cbind(svmTrainPred, svmTrainResp) # finished data set
    colnames(svmTrain) <- c(paste("lag", 2:1,sep = ""), "response")
    
    # moving testing data set
    svmTestResp <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)] # response in testing
    svmTestPred <- data.frame()
    for (j in 0:167){
      auxTestPred <- c(ts2[((528 - 2 + 1) + (i-1)*168 + j):(528 + (i-1)*168 + j)]) # predictors
      svmTestPred <- rbind(svmTestPred, auxTestPred) # binding all predictors
    }  
    svmTest <- cbind(svmTestPred, svmTestResp) # finished data set
    colnames(svmTest) <- c(paste("lag", 2:1,sep = ""), "response")
    
    # parameter optimization
    regrTr1 <- makeRegrTask(data = svmTrain, target = "response")
    svmTune <- tuneParamsMultiCrit("regr.ksvm", task = regrTr1, resampling = rdescAlt, par.set = psAlt,
                                   measures = list(mae, rmse), control = ctrlAlt, show.info = FALSE)
    
    # optimized model learning
    optParams <- vector()
    for(k in 1:length(svmTune$x)){ # get the best set in case of multiple optima 
      optParams[k] <- svmTune$x[[k]]$C # high C tends to screw up predictions a bit
    }
    svrParamsFeat2[[i]] <- svmTune$x[[which.min(optParams)]]
    lrnFilter2 <- setHyperPars(makeLearner("regr.ksvm"), par.vals = svmTune$x[[which.min(optParams)]])
    modelFinal <- train(lrnFilter2, task = regrTr1)
    
    # predictions
    regrTe1 <- makeRegrTask(data = svmTest, target = "response")
    predSVM <- predict(modelFinal, task = regrTe1)
    
    # measures
    svrPredFeat2[[i]] <- predSVM
    svrPerfFeat2[[1]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response)))/ length(predSVM$data$truth)
    svrPerfFeat2[[2]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response) 
                               / predSVM$data$truth)) / length(predSVM$data$truth)
    svrPerfFeat2[[3]][i] <- sqrt(sum((predSVM$data$truth - predSVM$data$response)^2) / length(predSVM$data$truth))
    print(i)
}

# stopping parallelization
parallelStop()

############## rest done using a function ###############

svrPerfFeat3 <- list(a, b, c)
svrPredFeat3 <- list()
svrParamsFeat3 <- list()

svrPerfFeat4 <- list(a, b, c)
svrPredFeat4 <- list()
svrParamsFeat4 <- list()

svrPerfFeat5 <- list(a, b, c)
svrPredFeat5 <- list()
svrParamsFeat5 <- list()

# starting parallelization
parallelStartSocket(2) 

for(i in 1:100){  # loop for training the actual model
  # 3 lags used
  # moving training data set
  svmTrainResp <- ts2[(25 + (i-1)*168):(528 + (i-1)*168)] # response in training
  svmTrainPred <- data.frame()
  for (j in 0:503){
    auxTrainPred <- c(ts2[((24 - 3 + 1) + (i-1)*168 + j):(24 + (i-1)*168 + j)]) # predictors in training
    svmTrainPred <- rbind(svmTrainPred, auxTrainPred) # binding all predictors
  }
  svmTrain <- cbind(svmTrainPred, svmTrainResp) # finished data set
  colnames(svmTrain) <- c(paste("lag", 3:1,sep = ""), "response")
  
  # moving testing data set
  svmTestResp <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)] # response in testing
  svmTestPred <- data.frame()
  for (j in 0:167){
    auxTestPred <- c(ts2[((528 - 3 + 1) + (i-1)*168 + j):(528 + (i-1)*168 + j)]) # predictors
    svmTestPred <- rbind(svmTestPred, auxTestPred) # binding all predictors
  }  
  svmTest <- cbind(svmTestPred, svmTestResp) # finished data set
  colnames(svmTest) <- c(paste("lag", 3:1,sep = ""), "response")
  
  # parameter optimization
  regrTr1 <- makeRegrTask(data = svmTrain, target = "response")
  svmTune <- tuneParamsMultiCrit("regr.ksvm", task = regrTr1, resampling = rdescAlt, par.set = psAlt,
                                 measures = list(mae, rmse), control = ctrlAlt, show.info = FALSE)
  
  # optimized model learning
  optParams <- vector()
  for(k in 1:length(svmTune$x)){ # get the best set in case of multiple optima 
    optParams[k] <- svmTune$x[[k]]$C # high C tends to screw up predictions a bit
  }
  svrParamsFeat3[[i]] <- svmTune$x[[which.min(optParams)]]
  lrnFilter2 <- setHyperPars(makeLearner("regr.ksvm"), par.vals = svmTune$x[[which.min(optParams)]])
  modelFinal <- train(lrnFilter2, task = regrTr1)
  
  # predictions
  regrTe1 <- makeRegrTask(data = svmTest, target = "response")
  predSVM <- predict(modelFinal, task = regrTe1)
  
  # measures
  svrPredFeat3[[i]] <- predSVM
  svrPerfFeat3[[1]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response)))/ length(predSVM$data$truth)
  svrPerfFeat3[[2]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response) 
                                  / predSVM$data$truth)) / length(predSVM$data$truth)
  svrPerfFeat3[[3]][i] <- sqrt(sum((predSVM$data$truth - predSVM$data$response)^2) / length(predSVM$data$truth))
  print(i)
}

# stopping parallelization
parallelStop()

# starting parallelization
parallelStartSocket(2) 

for(i in 1:100){  # loop for training the actual model
  # 4 lags used
  # moving training data set
  svmTrainResp <- ts2[(25 + (i-1)*168):(528 + (i-1)*168)] # response in training
  svmTrainPred <- data.frame()
  for (j in 0:503){
    auxTrainPred <- c(ts2[((24 - 4 + 1) + (i-1)*168 + j):(24 + (i-1)*168 + j)]) # predictors in training
    svmTrainPred <- rbind(svmTrainPred, auxTrainPred) # binding all predictors
  }
  svmTrain <- cbind(svmTrainPred, svmTrainResp) # finished data set
  colnames(svmTrain) <- c(paste("lag", 4:1,sep = ""), "response")
  
  # moving testing data set
  svmTestResp <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)] # response in testing
  svmTestPred <- data.frame()
  for (j in 0:167){
    auxTestPred <- c(ts2[((528 - 4 + 1) + (i-1)*168 + j):(528 + (i-1)*168 + j)]) # predictors
    svmTestPred <- rbind(svmTestPred, auxTestPred) # binding all predictors
  }  
  svmTest <- cbind(svmTestPred, svmTestResp) # finished data set
  colnames(svmTest) <- c(paste("lag", 4:1,sep = ""), "response")
  
  # parameter optimization
  regrTr1 <- makeRegrTask(data = svmTrain, target = "response")
  svmTune <- tuneParamsMultiCrit("regr.ksvm", task = regrTr1, resampling = rdescAlt, par.set = psAlt,
                                 measures = list(mae, rmse), control = ctrlAlt, show.info = FALSE)
  
  # optimized model learning
  optParams <- vector()
  for(k in 1:length(svmTune$x)){ # get the best set in case of multiple optima 
    optParams[k] <- svmTune$x[[k]]$C # high C tends to screw up predictions a bit
  }
  svrParamsFeat4[[i]] <- svmTune$x[[which.min(optParams)]]
  lrnFilter2 <- setHyperPars(makeLearner("regr.ksvm"), par.vals = svmTune$x[[which.min(optParams)]])
  modelFinal <- train(lrnFilter2, task = regrTr1)
  
  # predictions
  regrTe1 <- makeRegrTask(data = svmTest, target = "response")
  predSVM <- predict(modelFinal, task = regrTe1)
  
  # measures
  svrPredFeat4[[i]] <- predSVM
  svrPerfFeat4[[1]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response)))/ length(predSVM$data$truth)
  svrPerfFeat4[[2]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response) 
                                  / predSVM$data$truth)) / length(predSVM$data$truth)
  svrPerfFeat4[[3]][i] <- sqrt(sum((predSVM$data$truth - predSVM$data$response)^2) / length(predSVM$data$truth))
  print(i)
}

# stopping parallelization
parallelStop()

# starting parallelization
parallelStartSocket(2) 

for(i in 1:100){  # loop for training the actual model
  # 5 lags used
  # moving training data set
  svmTrainResp <- ts2[(25 + (i-1)*168):(528 + (i-1)*168)] # response in training
  svmTrainPred <- data.frame()
  for (j in 0:503){
    auxTrainPred <- c(ts2[((24 - 5 + 1) + (i-1)*168 + j):(24 + (i-1)*168 + j)]) # predictors in training
    svmTrainPred <- rbind(svmTrainPred, auxTrainPred) # binding all predictors
  }
  svmTrain <- cbind(svmTrainPred, svmTrainResp) # finished data set
  colnames(svmTrain) <- c(paste("lag", 5:1,sep = ""), "response")
  
  # moving testing data set
  svmTestResp <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)] # response in testing
  svmTestPred <- data.frame()
  for (j in 0:167){
    auxTestPred <- c(ts2[((528 - 5 + 1) + (i-1)*168 + j):(528 + (i-1)*168 + j)]) # predictors
    svmTestPred <- rbind(svmTestPred, auxTestPred) # binding all predictors
  }  
  svmTest <- cbind(svmTestPred, svmTestResp) # finished data set
  colnames(svmTest) <- c(paste("lag", 5:1,sep = ""), "response")
  
  # parameter optimization
  regrTr1 <- makeRegrTask(data = svmTrain, target = "response")
  svmTune <- tuneParamsMultiCrit("regr.ksvm", task = regrTr1, resampling = rdescAlt, par.set = psAlt,
                                 measures = list(mae, rmse), control = ctrlAlt, show.info = FALSE)
  
  # optimized model learning
  optParams <- vector()
  for(k in 1:length(svmTune$x)){ # get the best set in case of multiple optima 
    optParams[k] <- svmTune$x[[k]]$C # high C tends to screw up predictions a bit
  }
  svrParamsFeat5[[i]] <- svmTune$x[[which.min(optParams)]]
  lrnFilter2 <- setHyperPars(makeLearner("regr.ksvm"), par.vals = svmTune$x[[which.min(optParams)]])
  modelFinal <- train(lrnFilter2, task = regrTr1)
  
  # predictions
  regrTe1 <- makeRegrTask(data = svmTest, target = "response")
  predSVM <- predict(modelFinal, task = regrTe1)
  
  # measures
  svrPredFeat5[[i]] <- predSVM
  svrPerfFeat5[[1]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response)))/ length(predSVM$data$truth)
  svrPerfFeat5[[2]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response) 
                                  / predSVM$data$truth)) / length(predSVM$data$truth)
  svrPerfFeat5[[3]][i] <- sqrt(sum((predSVM$data$truth - predSVM$data$response)^2) / length(predSVM$data$truth))
  print(i)
}

# stopping parallelization
parallelStop()



