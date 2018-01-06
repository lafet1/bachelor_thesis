
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

###### SVR hour-by-hour ######

# general variables for training

# for parameter tuning
ps <- makeParamSet(
  makeNumericParam("C", lower = -12, upper = 12, trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -12, upper = 12, trafo = function(x) 2^x),
  makeDiscreteParam("kernel", values = c("rbfdot")),
  makeNumericParam("epsilon", lower = -5, upper = 0, trafo = function(x) 10^x)
)
ctrl <- makeTuneControlGrid()
rdesc <- makeResampleDesc("FixedCV", initial.window = 0.5) # FixedCV is for time-series CV

# for feature selection
rdesc1 <- makeResampleDesc("CV", iters = 50) # FixedCV is for time-series CV
psFeatSel <- makeParamSet(makeDiscreteParam("fw.abs", values = seq(2, 6, 1)))
lrnFilter <- makeFilterWrapper(learner = "regr.ksvm", fw.method = "information.gain")


###### 1 #####

###### 3 weeks #####

# first the feature selection needs to be carried out -- selecting features
regrTr1ax <- makeRegrTask(data = train1c, target = "response")
filter1a <- tuneParams(learner = lrnFilter, task = regrTr1ax, resampling = rdesc1, par.set = psFeatSel, 
                                measures = list(mae, rmse), control = ctrl, show.info = FALSE)

filter1a
lrnFilter1 <- makeFilterWrapper(learner = "regr.ksvm", fw.method = "information.gain", fw.abs = filter1a$x$fw.abs)
modelFilter <- train(lrnFilter1, regrTr1ax)
getFilteredFeatures(modelFilter)

finFeat1a <- c(getFilteredFeatures(modelFilter), "response")

# parameter optimization
regrTr1a <- makeRegrTask(data = train1c[finFeat1a], target = "response")
regrTe1a <- makeRegrTask(data = test1[finFeat1a], target = "response")
res1a <- tuneParams("regr.ksvm", task = regrTr1a, resampling = rdesc, par.set = ps,
                          measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res1a
res1a$y

# optimized model learning
lrn1a <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res1a$x)
model1a <- train(lrn1a, task = regrTr1a)

# predictions
task.pred1a <- predict(model1a, task = regrTe1a)
task.pred1a

# test evaluation
plot(task.pred1a$data$truth, main = "Test model 1", type = "l")
lines(task.pred1a$data$response, type = "l", col = "red")

sum(abs((task.pred1a$data$truth - task.pred1a$data$response)))/ length(task.pred1a$data$truth)
sum(abs((task.pred1a$data$truth - task.pred1a$data$response) 
        / task.pred1a$data$truth)) / length(task.pred1a$data$truth)
sqrt(sum((task.pred1a$data$truth - task.pred1a$data$response)^2) / length(task.pred1a$data$truth))





##############above filtering values, below wrapper for feature selection ###############





###### b #####

# first the feature selection needs to be carried out -- selecting features
regrTr1bx <- makeRegrTask(data = train1b, target = "response")
sfeats1b <- selectFeatures(learner = "regr.ksvm", task = regrTr1bx, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats1b
finFeat1b <- c(sfeats1b$x, "response")

# parameter optimization
regrTr1b <- makeRegrTask(data = train1b[finFeat1b], target = "response")
regrTe1b <- makeRegrTask(data = test1[finFeat1b], target = "response")
res1b <- tuneParamsMultiCrit("regr.ksvm", task = regrTr1b, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res1b
res1b$y

# optimized model learning
lrn1b <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res1b$x[[1]])
model1b <- train(lrn1b, task = regrTr1b)

# predictions
task.pred1b <- predict(model1b, task = regrTe1b)
task.pred1b

# evaluation
plot(task.pred1b$data$truth, main = "Test model 1", type = "l")
lines(task.pred1b$data$response, type = "l", col = "red")

sum(abs((task.pred1b$data$truth - task.pred1b$data$response)))/ length(task.pred1b$data$truth)
sum(abs((task.pred1b$data$truth - task.pred1b$data$response) 
        / task.pred1b$data$truth)) / length(task.pred1b$data$truth)
sum((task.pred1b$data$truth - task.pred1b$data$response)^2) / length(task.pred1b$data$truth)

###### c #####

# first the feature selection needs to be carried out -- selecting features
regrTr1cx <- makeRegrTask(data = train1c, target = "response")
sfeats1c <- selectFeatures(learner = "regr.ksvm", task = regrTr1cx, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats1c
finFeat1c <- c(sfeats1c$x, "response")

# parameter optimization
regrTr1c <- makeRegrTask(data = train1c[finFeat1c], target = "response")
regrTe1c <- makeRegrTask(data = test1[finFeat1c], target = "response")
res1c <- tuneParamsMultiCrit("regr.ksvm", task = regrTr1c, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res1c
res1c$y

# optimized model learning
lrn1c <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res1c$x[[1]])
model1c <- train(lrn1c, task = regrTr1c)

# predictions
task.pred1c <- predict(model1c, task = regrTe1c)
task.pred1c

# test evaluation
plot(task.pred1c$data$truth, main = "Test model 1", type = "l")
lines(task.pred1c$data$response, type = "l", col = "red")

sum(abs((task.pred1c$data$truth - task.pred1c$data$response)))/ length(task.pred1c$data$truth)
sum(abs((task.pred1c$data$truth - task.pred1c$data$response) 
        / task.pred1c$data$truth)) / length(task.pred1c$data$truth)
sum((task.pred1c$data$truth - task.pred1c$data$response)^2) / length(task.pred1c$data$truth)

###### d #####

# first the feature selection needs to be carried out -- selecting features
regrTr1dx <- makeRegrTask(data = train1d, target = "response")
sfeats1d <- selectFeatures(learner = "regr.ksvm", task = regrTr1dx, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats1d
finFeat1d <- c(sfeats1d$x, "response")

# parameter optimization
regrTr1d <- makeRegrTask(data = train1d[finFeat1d], target = "response")
regrTe1d <- makeRegrTask(data = test1[finFeat1d], target = "response")
res1d <- tuneParamsMultiCrit("regr.ksvm", task = regrTr1d, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res1d
res1d$y

# optimized model learning
lrn1d <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res1d$x[[1]])
model1d <- train(lrn1d, task = regrTr1d)

# predictions
task.pred1d <- predict(model1d, task = regrTe1d)
task.pred1d

# test evaluation
plot(task.pred1d$data$truth, main = "Test model 1", type = "l")
lines(task.pred1d$data$response, type = "l", col = "red")

sum(abs((task.pred1d$data$truth - task.pred1d$data$response)))/ length(task.pred1d$data$truth)
sum(abs((task.pred1d$data$truth - task.pred1d$data$response) 
        / task.pred1d$data$truth)) / length(task.pred1d$data$truth)
sum((task.pred1d$data$truth - task.pred1d$data$response)^2) / length(task.pred1d$data$truth)

###### 2 #####

###### a #####

regrTr2ax <- makeRegrTask(data = train2a, target = "response")
sfeats2a <- selectFeatures(learner = "regr.ksvm", task = regrTr2ax, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats2a
finFeat2a <- c(sfeats2a$x, "response")

# parameter optimization
regrTr2a <- makeRegrTask(data = train2a[finFeat2a], target = "response")
regrTe2a <- makeRegrTask(data = test2[finFeat2a], target = "response")
res2a <- tuneParamsMultiCrit("regr.ksvm", task = regrTr2a, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res2a

# optimized model learning
lrn2a <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res2a$x[[1]])
model2a <- train(lrn2a, task = regrTr2a)

# predictions
task.pred2a <- predict(model2a, task = regrTe2a)
task.pred2a

# test evaluation
plot(task.pred2a$data$truth, main = "Test model 1", type = "l")
lines(task.pred2a$data$response, type = "l", col = "red")

sum(abs((task.pred2a$data$truth - task.pred2a$data$response)))/ length(task.pred2a$data$truth)
sum(abs((task.pred2a$data$truth - task.pred2a$data$response) 
        / task.pred2a$data$truth)) / length(task.pred2a$data$truth)
sum((task.pred2a$data$truth - task.pred2a$data$response)^2) / length(task.pred2a$data$truth)

###### b #####

regrTr2bx <- makeRegrTask(data = train2b, target = "response")
sfeats2b <- selectFeatures(learner = "regr.ksvm", task = regrTr2bx, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats2b
finFeat2b <- c(sfeats2b$x, "response")

# parameter optimization
regrTr2b <- makeRegrTask(data = train2b[finFeat2b], target = "response")
regrTe2b <- makeRegrTask(data = test2[finFeat2b], target = "response")
res2b <- tuneParamsMultiCrit("regr.ksvm", task = regrTr2b, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res2b

# optimized model learning
lrn2b <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res2b$x[[1]])
model2b <- train(lrn2b, task = regrTr2b)

# predictions
task.pred2b <- predict(model2b, task = regrTe2b)
task.pred2b

# test evaluation
plot(task.pred2b$data$truth, main = "Test model 1", type = "l")
lines(task.pred2b$data$response, type = "l", col = "red")

sum(abs((task.pred2b$data$truth - task.pred2b$data$response)))/ length(task.pred2b$data$truth)
sum(abs((task.pred2b$data$truth - task.pred2b$data$response) 
        / task.pred2b$data$truth)) / length(task.pred2b$data$truth)
sum((task.pred2b$data$truth - task.pred2b$data$response)^2) / length(task.pred2b$data$truth)

###### c #####

regrTr2cx <- makeRegrTask(data = train2c, target = "response")
sfeats2c <- selectFeatures(learner = "regr.ksvm", task = regrTr2cx, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats2c
finFeat2c <- c(sfeats2c$x, "response")

# parameter optimization
regrTr2c <- makeRegrTask(data = train2c[finFeat2c], target = "response")
regrTe2c <- makeRegrTask(data = test2[finFeat2c], target = "response")
res2c <- tuneParamsMultiCrit("regr.ksvm", task = regrTr2c, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res2c

# optimized model learning
lrn2c <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res2c$x[[1]])
model2c <- train(lrn2c, task = regrTr2c)

# predictions
task.pred2c <- predict(model2c, task = regrTe2c)
task.pred2c

# test evaluation
plot(task.pred2c$data$truth, main = "Test model 1", type = "l")
lines(task.pred2c$data$response, type = "l", col = "red")

sum(abs((task.pred2c$data$truth - task.pred2c$data$response)))/ length(task.pred2c$data$truth)
sum(abs((task.pred2c$data$truth - task.pred2c$data$response) 
        / task.pred2c$data$truth)) / length(task.pred2c$data$truth)
sum((task.pred2c$data$truth - task.pred2c$data$response)^2) / length(task.pred2c$data$truth)

###### d #####

# first the feature selection needs to be carried out -- selecting features
regrTr2dx <- makeRegrTask(data = train2d, target = "response")
sfeats2d <- selectFeatures(learner = "regr.ksvm", task = regrTr2dx, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats2d
finFeat2d <- c(sfeats2d$x, "response")

# parameter optimization
regrTr2d <- makeRegrTask(data = train2d[finFeat2d], target = "response")
regrTe2d <- makeRegrTask(data = test2[finFeat2d], target = "response")
res2d <- tuneParamsMultiCrit("regr.ksvm", task = regrTr2d, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res2d

# optimized model learning
lrn2d <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res2d$x[[1]])
model2d <- train(lrn2d, task = regrTr2d)

# predictions
task.pred2d <- predict(model2d, task = regrTe2d)
task.pred2d

# test evaluation
plot(task.pred2d$data$truth, main = "Test model 1", type = "l")
lines(task.pred2d$data$response, type = "l", col = "red")

sum(abs((task.pred2d$data$truth - task.pred2d$data$response)))/ length(task.pred2d$data$truth)
sum(abs((task.pred2d$data$truth - task.pred2d$data$response) 
        / task.pred2d$data$truth)) / length(task.pred2d$data$truth)
sum((task.pred2d$data$truth - task.pred2d$data$response)^2) / length(task.pred2d$data$truth)

###### 3 #####

###### a #####

# first the feature selection needs to be carried out -- selecting features
regrTr3ax <- makeRegrTask(data = train3a, target = "response")
sfeats3a <- selectFeatures(learner = "regr.ksvm", task = regrTr3ax, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats3a
finFeat3a <- c(sfeats3a$x, "response")

# parameter optimization
regrTr3a <- makeRegrTask(data = train3a[finFeat3a], target = "response")
regrTe3a <- makeRegrTask(data = test3[finFeat3a], target = "response")
res3a <- tuneParamsMultiCrit("regr.ksvm", task = regrTr3a, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res3a

# optimized model learning
lrn3a <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res3a$x[[1]])
model3a <- train(lrn3a, task = regrTr3a)

# predictions
task.pred3a <- predict(model3a, task = regrTe3a)
task.pred3a

# test evaluation
plot(task.pred3a$data$truth, main = "Test model 1", type = "l")
lines(task.pred3a$data$response, type = "l", col = "red")

sum(abs((task.pred3a$data$truth - task.pred3a$data$response)))/ length(task.pred3a$data$truth)
sum(abs((task.pred3a$data$truth - task.pred3a$data$response) 
        / task.pred3a$data$truth)) / length(task.pred3a$data$truth)
sum((task.pred3a$data$truth - task.pred3a$data$response)^2) / length(task.pred3a$data$truth)

###### b #####

# first the feature selection needs to be carried out -- selecting features
regrTr3bx <- makeRegrTask(data = train3b, target = "response")
sfeats3b <- selectFeatures(learner = "regr.ksvm", task = regrTr3bx, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats3b
finFeat3b <- c(sfeats3b$x, "response")

# parameter optimization
regrTr3b <- makeRegrTask(data = train3b[finFeat3b], target = "response")
regrTe3b <- makeRegrTask(data = test3[finFeat3b], target = "response")
res3b <- tuneParamsMultiCrit("regr.ksvm", task = regrTr3b, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res3b

# optimized model learning
lrn3b <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res3b$x[[1]])
model3b <- train(lrn3b, task = regrTr3b)

# predictions
task.pred3b <- predict(model3b, task = regrTe3b)
task.pred3b

# test evaluation
plot(task.pred3b$data$truth, main = "Test model 1", type = "l")
lines(task.pred3b$data$response, type = "l", col = "red")

sum(abs((task.pred3b$data$truth - task.pred3b$data$response)))/ length(task.pred3b$data$truth)
sum(abs((task.pred3b$data$truth - task.pred3b$data$response) 
        / task.pred3b$data$truth)) / length(task.pred3b$data$truth)
sum((task.pred3b$data$truth - task.pred3b$data$response)^2) / length(task.pred3b$data$truth)

###### c #####

# first the feature selection needs to be carried out -- selecting features
regrTr3cx <- makeRegrTask(data = train3c, target = "response")
sfeats3c <- selectFeatures(learner = "regr.ksvm", task = regrTr3cx, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats3c
finFeat3c <- c(sfeats3c$x, "response")

# parameter optimization
regrTr3c <- makeRegrTask(data = train3c[finFeat3c], target = "response")
regrTe3c <- makeRegrTask(data = test3[finFeat3c], target = "response")
res3c <- tuneParamsMultiCrit("regr.ksvm", task = regrTr3c, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res3c

# optimized model learning
lrn3c <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res3c$x[[1]])
model3c <- train(lrn3c, task = regrTr3c)

# predictions
task.pred3c <- predict(model3c, task = regrTe3c)
task.pred3c

# test evaluation
plot(task.pred3c$data$truth, main = "Test model 1", type = "l")
lines(task.pred3c$data$response, type = "l", col = "red")

sum(abs((task.pred3c$data$truth - task.pred3c$data$response)))/ length(task.pred3c$data$truth)
sum(abs((task.pred3c$data$truth - task.pred3c$data$response) 
        / task.pred3c$data$truth)) / length(task.pred3c$data$truth)
sum((task.pred3c$data$truth - task.pred3c$data$response)^2) / length(task.pred3c$data$truth)

###### d #####

# first the feature selection needs to be carried out -- selecting features
regrTr3dx <- makeRegrTask(data = train3d, target = "response")
sfeats3d <- selectFeatures(learner = "regr.ksvm", task = regrTr3dx, resampling = rdesc1, 
                           control = ctrl1, show.info = FALSE)
sfeats3d
finFeat3d <- c(sfeats3d$x, "response")

# parameter optimization
regrTr3d <- makeRegrTask(data = train3d[finFeat3d], target = "response")
regrTe3d <- makeRegrTask(data = test3[finFeat3d], target = "response")
res3d <- tuneParamsMultiCrit("regr.ksvm", task = regrTr3d, resampling = rdesc, par.set = ps,
                             measures = list(mae, rmse), control = ctrl, show.info = FALSE)
res3d

# optimized model learning
lrn3d <- setHyperPars(makeLearner("regr.ksvm"), par.vals = res3d$x[[1]])
model3d <- train(lrn3d, task = regrTr3d)

# predictions
task.pred3d <- predict(model3d, task = regrTe3d)
task.pred3d

# test evaluation
plot(task.pred3d$data$truth, main = "Test model 1", type = "l")
lines(task.pred3d$data$response, type = "l", col = "red")

sum(abs((task.pred3d$data$truth - task.pred3d$data$response)))/ length(task.pred3d$data$truth)
sum(abs((task.pred3d$data$truth - task.pred3d$data$response) 
        / task.pred3d$data$truth)) / length(task.pred3d$data$truth)
sum((task.pred3d$data$truth - task.pred3d$data$response)^2) / length(task.pred3d$data$truth)
