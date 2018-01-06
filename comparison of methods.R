
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


##### Naive SVRARIMA #####

# MAE
plot(naiveSVRARIMA2[[3]][[1]], type = "l")
lines(naiveSVRARIMA3[[3]][[1]], col = "red")
lines(naiveSVRARIMA4[[3]][[1]], col = "blue")
lines(naiveSVRARIMA5[[3]][[1]], col = "green")

sum(naiveSVRARIMA2[[3]][[1]])
sum(naiveSVRARIMA3[[3]][[1]])
sum(naiveSVRARIMA4[[3]][[1]])
sum(naiveSVRARIMA5[[3]][[1]])

# MAPE
plot(naiveSVRARIMA2[[3]][[2]], type = "l")
lines(naiveSVRARIMA3[[3]][[2]], col = "red")
lines(naiveSVRARIMA4[[3]][[2]], col = "blue")
lines(naiveSVRARIMA5[[3]][[2]], col = "green")

# RMSE
plot(naiveSVRARIMA2[[3]][[3]], type = "l")
lines(naiveSVRARIMA3[[3]][[3]], col = "red")
lines(naiveSVRARIMA4[[3]][[3]], col = "blue")
lines(naiveSVRARIMA5[[3]][[3]], col = "green")

sum(naiveSVRARIMA2[[3]][[3]])
sum(naiveSVRARIMA3[[3]][[3]])
sum(naiveSVRARIMA4[[3]][[3]])
sum(naiveSVRARIMA5[[3]][[3]])

##### SVRARIMA #####

# plots the measures over the 100-week period and also gives us the overall sum of the errors
# excluding MAPE because of the existence of 0 in the time series (i.e. infinity)
# MAE
plot(diffSVRARIMA2[[3]][[1]], type = "l")
lines(diffSVRARIMA3[[3]][[1]], col = "red")
lines(diffSVRARIMA4[[3]][[1]], col = "blue")
lines(diffSVRARIMA5[[3]][[1]], col = "green")

sum(diffSVRARIMA2[[3]][[1]])
sum(diffSVRARIMA3[[3]][[1]])
sum(diffSVRARIMA4[[3]][[1]]) # appears to be the best
sum(diffSVRARIMA5[[3]][[1]])

# MAPE
plot(diffSVRARIMA2[[3]][[2]], type = "l")
lines(diffSVRARIMA3[[3]][[2]], col = "red")
lines(diffSVRARIMA4[[3]][[2]], col = "blue")
lines(diffSVRARIMA5[[3]][[2]], col = "green")

# RMSE
plot(diffSVRARIMA2[[3]][[3]], type = "l")
lines(diffSVRARIMA3[[3]][[3]], col = "red")
lines(diffSVRARIMA4[[3]][[3]], col = "blue")
lines(diffSVRARIMA5[[3]][[3]], col = "green")

sum(diffSVRARIMA2[[3]][[3]])
sum(diffSVRARIMA3[[3]][[3]])
sum(diffSVRARIMA4[[3]][[3]])
sum(diffSVRARIMA5[[3]][[3]])

##### SVR #####

# MAE
plot(diffLag2[[2]][[1]], type = "l")
lines(diffLag3[[2]][[1]], col = "red")
lines(diffLag4[[2]][[1]], col = "blue")
lines(diffLag5[[2]][[1]], col = "green")

sum(diffLag2[[2]][[1]])
sum(diffLag3[[2]][[1]])
sum(diffLag4[[2]][[1]])
sum(diffLag5[[2]][[1]])

# MAPE
plot(diffLag2[[2]][[2]], type = "l")
lines(diffLag3[[2]][[2]], col = "red")
lines(diffLag4[[2]][[2]], col = "blue")
lines(diffLag5[[2]][[2]], col = "green")

# RMSE
plot(diffLag2[[2]][[3]], type = "l")
lines(diffLag3[[2]][[3]], col = "red")
lines(diffLag4[[2]][[3]], col = "blue")
lines(diffLag5[[2]][[3]], col = "green")

sum(diffLag2[[2]][[3]])
sum(diffLag3[[2]][[3]])
sum(diffLag4[[2]][[3]])
sum(diffLag5[[2]][[3]])

##### ARIMA #####

# plots the measures over the 100-week period and also gives us the overall sum of the errors
# excluding MAPE because of the existence of 0 in the time series (i.e. infinity)
# MAE
plot(arimaPerf[[1]][1:100], type = "l")
sum(arimaPerf[[1]][1:100])

# MAPE
plot(arimaPerf[[2]][1:100], type = "l")

# RMSE
plot(arimaPerf[[3]][1:100], type = "l")
sum(arimaPerf[[3]][1:100])

##### PSF #####

# plots the measures over the 100-week period and also gives us the overall sum of the errors
# excluding MAPE because of the existence of 0 in the time series (i.e. infinity)
# MAE
plot(unlist(PSF[[1]][[1]]), type = "l")
sum(PSF[[1]][[1]])

# MAPE
plot(unlist(PSF[[1]][[2]]), type = "l")

# RMSE
plot(unlist(PSF[[1]][[3]]), type = "l")
sum(PSF[[1]][[3]])



##### Comparison #####

# compares ARIMA and SVR and how often one outperforms another
arimaVsvr2 <- c(sum(arimaPerf[[1]][1:100] < diffLag2[[2]][[1]]), sum(arimaPerf[[3]][1:100] < diffLag2[[2]][[3]]))
arimaVsvr3 <- c(sum(arimaPerf[[1]][1:100] < diffLag3[[2]][[1]]), sum(arimaPerf[[3]][1:100] < diffLag3[[2]][[3]]))
arimaVsvr4 <- c(sum(arimaPerf[[1]][1:100] < diffLag4[[2]][[1]]), sum(arimaPerf[[3]][1:100] < diffLag4[[2]][[3]]))
arimaVsvr5 <- c(sum(arimaPerf[[1]][1:100] < diffLag5[[2]][[1]]), sum(arimaPerf[[3]][1:100] < diffLag5[[2]][[3]]))

# compares ARIMA and SVRARIMA and how often one outperforms another
arimaVhybrid2 <- c(sum(arimaPerf[[1]][1:100] < diffSVRARIMA2[[3]][[1]]), sum(arimaPerf[[3]][1:100] < diffSVRARIMA2[[3]][[3]]))
arimaVhybrid3 <- c(sum(arimaPerf[[1]][1:100] < diffSVRARIMA3[[3]][[1]]), sum(arimaPerf[[3]][1:100] < diffSVRARIMA3[[3]][[3]]))
arimaVhybrid4 <- c(sum(arimaPerf[[1]][1:100] < diffSVRARIMA4[[3]][[1]]), sum(arimaPerf[[3]][1:100] < diffSVRARIMA4[[3]][[3]]))
arimaVhybrid5 <- c(sum(arimaPerf[[1]][1:100] < diffSVRARIMA5[[3]][[1]]), sum(arimaPerf[[3]][1:100] < diffSVRARIMA5[[3]][[3]]))

# plots the difference between sVRARIMA and ARIMA - MAE, RMSE
plot(diffSVRARIMA2[[3]][[1]] - arimaPerf[[1]][1:100], type = "l")
abline(0, 0)
lines(diffSVRARIMA3[[3]][[1]] - arimaPerf[[1]][1:100], col = "red")
lines(diffSVRARIMA4[[3]][[1]] - arimaPerf[[1]][1:100], col = "blue")
lines(diffSVRARIMA5[[3]][[1]] - arimaPerf[[1]][1:100], col = "green")

plot(diffSVRARIMA2[[3]][[3]] - arimaPerf[[3]][1:100], type = "l")
abline(0, 0)
lines(diffSVRARIMA3[[3]][[3]] - arimaPerf[[3]][1:100], col = "red")
lines(diffSVRARIMA4[[3]][[3]] - arimaPerf[[3]][1:100], col = "blue")
lines(diffSVRARIMA5[[3]][[3]] - arimaPerf[[3]][1:100], col = "green")

##### Diebold-Mariano test #####

orig <- list()
for (i in 1:100){
  orig[[i]] <- ts2[(529 + (i-1)*168):(696 + (i-1)*168)]
}

dmTest <- function(orig, pred1, pred2){ # test for calculating all 100 statistics in 1 line
  outcome <- list(a, b)
  statistics <- vector()
  pvals <- vector()
  
  for (i in 1:100){
    e1 <- pred1[[i]] - orig[[i]]
    e2 <- pred2[[i]] - orig[[i]]
    outcome[[1]][[i]] <- dm.test(e1, e2, power = 1)
    statistics[[i]] <- outcome[[1]][[i]]$statistic
    pvals[[i]] <- outcome[[1]][[i]]$p.value
  }
  
  outcome[[2]] <- statistics
  outcome[[3]] <- pvals
  
  return(outcome)
}

dmTestSVR2 <- dmTest(orig, arimaPred, diffLag2[[3]]) # the tests themsleves
dmTestSVR3 <- dmTest(orig, arimaPred, diffLag3[[3]])
dmTestSVR4 <- dmTest(orig, arimaPred, diffLag4[[3]])
dmTestSVR5 <- dmTest(orig, arimaPred, diffLag5[[3]])

dmTestSVR2b <- dmTest(orig, diffLag2[[3]], diffSVRARIMA2[[2]]) #list(a, b, c)
#for(i in c(1:93, 95:100)){
#  dmTestSVR2b[[1]][[i]] <- dm.test(diffLag2[[3]][[i]] - orig[[i]], diffSVRARIMA2[[2]][[i]] - orig[[i]], power = 1)
#  dmTestSVR2b[[2]][i] <- dmTestSVR2b[[1]][[i]]$statistic
#  dmTestSVR2b[[3]][i] <- dmTestSVR2b[[1]][[i]]$p.value
#}
dmTestSVR3b <- dmTest(orig, diffLag3[[3]], diffSVRARIMA3[[2]])
dmTestSVR4b <- dmTest(orig, diffLag4[[3]], diffSVRARIMA4[[2]])
dmTestSVR5b <- dmTest(orig, diffLag5[[3]], diffSVRARIMA5[[2]])

dmTestSVRARIMA2 <- dmTest(orig, arimaPred, diffSVRARIMA2[[2]])
dmTestSVRARIMA3 <- dmTest(orig, arimaPred, diffSVRARIMA3[[2]])
dmTestSVRARIMA4 <- dmTest(orig, arimaPred, diffSVRARIMA4[[2]])
dmTestSVRARIMA5 <- dmTest(orig, arimaPred, diffSVRARIMA5[[2]])

plot(density(dmTestSVR2[[2]]))
abline(v = 1.96) 
abline(v = - 1.96)
plot(density(dmTestSVR3[[2]]))
abline(v = 1.96) 
abline(v = - 1.96)
plot(density(dmTestSVR4[[2]]))
abline(v = 1.96) 
abline(v = - 1.96)
plot(density(dmTestSVR5[[2]]))
abline(v = 1.96) 
abline(v = - 1.96)

c(sum(dmTestSVR2[[2]] < -1.96), sum(-1.96 < dmTestSVR2[[2]] & dmTestSVR2[[2]] < 1.96), 
  sum(dmTestSVR2[[2]] > 1.96))
c(sum(dmTestSVR3[[2]] < -1.96), sum(-1.96 < dmTestSVR3[[2]] & dmTestSVR3[[2]] < 1.96), 
  sum(dmTestSVR3[[2]] > 1.96))
c(sum(dmTestSVR4[[2]] < -1.96), sum(-1.96 < dmTestSVR4[[2]] & dmTestSVR4[[2]] < 1.96), 
  sum(dmTestSVR4[[2]] > 1.96))
c(sum(dmTestSVR5[[2]] < -1.96), sum(-1.96 < dmTestSVR5[[2]] & dmTestSVR5[[2]] < 1.96), 
  sum(dmTestSVR5[[2]] > 1.96))

plot(density(dmTestSVR2b[[2]], na.rm = T))
abline(v = 1.96) 
abline(v = - 1.96)
plot(density(dmTestSVR3b[[2]]))
abline(v = 1.96) 
abline(v = - 1.96)
plot(density(dmTestSVR4b[[2]]))
abline(v = 1.96) 
abline(v = - 1.96)
plot(density(dmTestSVR5b[[2]]))
abline(v = 1.96) 
abline(v = - 1.96)

c(sum(dmTestSVR2b[[2]] < -1.96, na.rm = T), sum(-1.96 < dmTestSVR2b[[2]] & dmTestSVR2b[[2]] < 1.96, na.rm = T), 
  sum(dmTestSVR2b[[2]] > 1.96, na.rm = T))
c(sum(dmTestSVR3b[[2]] < -1.96), sum(-1.96 < dmTestSVR3b[[2]] & dmTestSVR3b[[2]] < 1.96), 
  sum(dmTestSVR3b[[2]] > 1.96))
c(sum(dmTestSVR4b[[2]] < -1.96), sum(-1.96 < dmTestSVR4b[[2]] & dmTestSVR4b[[2]] < 1.96), 
  sum(dmTestSVR4b[[2]] > 1.96))
c(sum(dmTestSVR5b[[2]] < -1.96), sum(-1.96 < dmTestSVR5b[[2]] & dmTestSVR5b[[2]] < 1.96), 
  sum(dmTestSVR5b[[2]] > 1.96))

plot(density(dmTestSVRARIMA2[[2]])) 
abline(v = 1.96) 
abline(v = - 1.96)
plot(density(dmTestSVRARIMA3[[2]]))
abline(v = 1.96) 
abline(v = - 1.96)
plot(density(dmTestSVRARIMA4[[2]]))
abline(v = 1.96) 
abline(v = - 1.96)
plot(density(dmTestSVRARIMA5[[2]]))
abline(v = 1.96) 
abline(v = - 1.96)

c(sum(dmTestSVRARIMA2[[2]] < -1.96), sum(-1.96 < dmTestSVRARIMA2[[2]] & dmTestSVRARIMA2[[2]] < 1.96), 
  sum(dmTestSVRARIMA2[[2]] > 1.96))
c(sum(dmTestSVRARIMA3[[2]] < -1.96), sum(-1.96 < dmTestSVRARIMA3[[2]] & dmTestSVRARIMA3[[2]] < 1.96), 
  sum(dmTestSVRARIMA3[[2]] > 1.96))
c(sum(dmTestSVRARIMA4[[2]] < -1.96), sum(-1.96 < dmTestSVRARIMA4[[2]] & dmTestSVRARIMA4[[2]] < 1.96), 
  sum(dmTestSVRARIMA4[[2]] > 1.96))
c(sum(dmTestSVRARIMA5[[2]] < -1.96), sum(-1.96 < dmTestSVRARIMA5[[2]] & dmTestSVRARIMA5[[2]] < 1.96), 
  sum(dmTestSVRARIMA5[[2]] > 1.96))
