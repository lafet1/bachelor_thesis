
# rm(list=ls())

library(RCurl)
library(dplyr)
library(e1071) # SVR via svm()
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

#svrPerf <- list(a, b, c)
#svrPred <- list()
#svrFeat <- list()
#svrParams <- list()

# for parameter tuning
ps <- makeParamSet(
  makeNumericParam("C", lower = -12, upper = 12, trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -12, upper = 12, trafo = function(x) 2^x),
  makeDiscreteParam("kernel", values = c("rbfdot")),
  makeNumericParam("epsilon", lower = -5, upper = 0, trafo = function(x) 10^x)
)
ctrl <- makeTuneMultiCritControlRandom(maxit = 100L)
rdesc <- makeResampleDesc("FixedCV", initial.window = 0.5) # FixedCV is for time-series CV

# for feature selection
rdesc1 <- makeResampleDesc("CV", iters = 50)
psFeatSel <- makeParamSet(makeDiscreteParam("fw.abs", values = seq(2, 6, 1)))
lrnFilter <- makeFilterWrapper(learner = "regr.ksvm", fw.method = "information.gain")
ctrl1 <- makeTuneMultiCritControlRandom(maxit = 50L)

# variable names
varNames <- c(paste("lag", 24:1, sep = ""), "response")

# starting parallelization
parallelStartSocket(2) 

for(i in 1:100){ # loop for getting the features
  
  # moving training data set
  svmTrainResp <- ts2[(25 + (i-1)*168):(528 + (i-1)*168)] # response in training
  svmTrainPred <- data.frame()
  for (j in 0:503){
    auxTrainPred <- c(ts2[(1 + (i-1)*168 + j):(24 + (i-1)*168 + j)]) # predictors in training
    svmTrainPred <- rbind(svmTrainPred, auxTrainPred) # binding all predictors
  }
  svmTrain <- cbind(svmTrainPred, svmTrainResp) # finished data set
  colnames(svmTrain) <- varNames
  
  # moving testing data set
  svmTestResp <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)] # response in testing
  svmTestPred <- data.frame()
  for (j in 0:167){
    auxTestPred <- c(ts2[(505 + (i-1)*168 + j):(528 + (i-1)*168 + j)]) # predictors
    svmTestPred <- rbind(svmTestPred, auxTestPred) # binding all predictors
  }  
  svmTest <- cbind(svmTestPred, svmTestResp) # finished data set
  colnames(svmTest) <- varNames
  
  # first the feature selection needs to be carried out -- selecting features
  regrTr <- makeRegrTask(data = svmTrain, target = "response")
  filter1a <- tuneParamsMultiCrit(learner = lrnFilter, task = regrTr, resampling = rdesc1, par.set = psFeatSel, 
                                  measures = list(mae, rmse), control = ctrl1, show.info = FALSE)
  
  # new learner based on the found features 
  lrnFilter1 <- makeFilterWrapper(learner = "regr.ksvm", fw.method = "information.gain", fw.abs = filter1a$x[[1]]$fw.abs)
  modelFilter <- train(lrnFilter1, regrTr)
  svrFeat[[i]] <- c(getFilteredFeatures(modelFilter), "response")
  print(i)
}

# stopping parallelization
parallelStop()

# starting parallelization
parallelStartSocket(2) 
  
for(i in 1:100){  # loop for training the actual model
  
  # moving training data set
  svmTrainResp <- ts2[(25 + (i-1)*168):(528 + (i-1)*168)] # response in training
  svmTrainPred <- data.frame()
  for (j in 0:503){
    auxTrainPred <- c(ts2[(1 + (i-1)*168 + j):(24 + (i-1)*168 + j)]) # predictors in training
    svmTrainPred <- rbind(svmTrainPred, auxTrainPred) # binding all predictors
  }
  svmTrain <- cbind(svmTrainPred, svmTrainResp) # finished data set
  colnames(svmTrain) <- varNames
  
  # moving testing data set
  svmTestResp <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)] # response in testing
  svmTestPred <- data.frame()
  for (j in 0:167){
    auxTestPred <- c(ts2[(505 + (i-1)*168 + j):(528 + (i-1)*168 + j)]) # predictors
    svmTestPred <- rbind(svmTestPred, auxTestPred) # binding all predictors
  }  
  svmTest <- cbind(svmTestPred, svmTestResp) # finished data set
  colnames(svmTest) <- varNames
  
  # parameter optimization
  regrTr1 <- makeRegrTask(data = svmTrain[svrFeat[[i]]], target = "response")
  svmTune <- tuneParamsMultiCrit("regr.ksvm", task = regrTr1, resampling = rdesc, par.set = ps,
                                 measures = list(mae, rmse), control = ctrl, show.info = FALSE)
  
  # optimized model learning
  optParams <- vector()
  for(k in 1:length(svmTune$x)){ # get the best set in case of multiple optima 
    optParams[k] <- svmTune$x[[k]]$C # high C tends to screw up predictions a bit
  }
  svrParams[[i]] <- svmTune$x[[which.min(optParams)]]
  lrnFilter2 <- setHyperPars(makeLearner("regr.ksvm"), par.vals = svmTune$x[[which.min(optParams)]])
  modelFinal <- train(lrnFilter2, task = regrTr1)
  
  # predictions
  regrTe1 <- makeRegrTask(data = svmTest[svrFeat[[i]]], target = "response")
  predSVM <- predict(modelFinal, task = regrTe1)
  
  # measures
  svrPred[[i]] <- predSVM
  svrPerf[[1]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response)))/ length(predSVM$data$truth)
  svrPerf[[2]][i] <- sum(abs((predSVM$data$truth - predSVM$data$response) 
                          / predSVM$data$truth)) / length(predSVM$data$truth)
  svrPerf[[3]][i] <- sqrt(sum((predSVM$data$truth - predSVM$data$response)^2) / length(predSVM$data$truth))
  print(i)
}

# stopping parallelization
parallelStop()


