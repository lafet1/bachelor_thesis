
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


set.seed(100)


#3 weeks probably best, 2 weeks training period close

###### ARIMA data #####

arimaWeek1 <- ts(ts1[(8760+rn1-167):(8760+rn1)], frequency = 24)
arima2Weeks1 <- ts(ts1[(8760+rn1-335):(8760+rn1)], frequency = 24)
arima3Weeks1 <- ts(ts1[(8760+rn1-503):(8760+rn1)], frequency = 24)
arimaMonth1 <- ts(ts1[(8760+rn1-671):(8760+rn1)], frequency = 24)
arimaTest1 <- ts(ts1[(8760+rn1+1):(8760+rn1+168)], frequency = 24)


arimaWeek2 <- ts(ts1[(8760+rn2-167):(8760+rn2)], frequency = 24)
arima2Weeks2 <- ts(ts1[(8760+rn2-335):(8760+rn2)], frequency = 24)
arima3Weeks2 <- ts(ts1[(8760+rn2-503):(8760+rn2)], frequency = 24)
arimaMonth2 <- ts(ts1[(8760+rn2-671):(8760+rn2)], frequency = 24)
arimaTest2 <- ts(ts1[(8760+rn2+1):(8760+rn2+168)], frequency = 24)


arimaWeek3 <- ts(ts1[(8760+rn3-167):(8760+rn3)], frequency = 24)
arima2Weeks3 <- ts(ts1[(8760+rn3-335):(8760+rn3)], frequency = 24)
arima3Weeks3 <- ts(ts1[(8760+rn3-503):(8760+rn3)], frequency = 24)
arimaMonth3 <- ts(ts1[(8760+rn3-671):(8760+rn3)], frequency = 24)
arimaTest3 <- ts(ts1[(8760+rn3+1):(8760+rn3+168)], frequency = 24)

###### ARIMA modeling ######

###### 1 ###### 

# hour-by-hour predictions - train
arima.model1 <- auto.arima(arima2Weeks1, trace = T)
fc1 <- fitted(arima.model1)

arima.model1y <- auto.arima(arima3Weeks1, trace = T)
fc1y <- fitted(arima.model1y)

arima.model1x <- auto.arima(arimaMonth1, trace = T)
fc1x <- fitted(arima.model1x)


# plotting the actual and forecasted training values
plot(arima2Weeks1, type = "l", ylab = "Price", main = "Train model 1")
lines(fc1, col = "blue", type = "l")

plot(arima3Weeks1, type = "l", ylab = "Price", main = "Train model 1")
lines(fc1y, col = "blue", type = "l")

plot(arimaMonth1, type = "l", ylab = "Price", main = "Train model 1")
lines(fc1x, col = "blue", type = "l")

# calculating evaluation criteria
sum(abs(arima2Weeks1 - fc1))/ length(arima2Weeks1) # MAE
sum(abs((arima2Weeks1 - fc1)/ arima2Weeks1))/ length(arima2Weeks1)  # MAPE
sum((arima2Weeks1 - fc1)^2)/ length(arima2Weeks1) # RMSE

sum(abs(arima3Weeks1 - fc1y))/ length(arima3Weeks1) # MAE
sum(abs((arima3Weeks1 - fc1y)/ arima3Weeks1))/ length(arima3Weeks1)  # MAPE
sum((arima3Weeks1 - fc1y)^2)/ length(arima3Weeks1) # RMSE

sum(abs(arimaMonth1 - fc1x))/ length(arimaMonth1) # MAE
sum(abs((arimaMonth1 - fc1x)/ arimaMonth1))/ length(arimaMonth1)  # MAPE
sum((arimaMonth1 - fc1x)^2)/ length(arimaMonth1) # RMSE

# hour-by-hour predictions - test
arima.model1a <- Arima(arimaTest1, model = arima.model1)
fc1a <- fitted(arima.model1a)

arima.model1ay <- Arima(arimaTest1, model = arima.model1y)
fc1ay <- fitted(arima.model1ay)

arima.model1ax <- Arima(arimaTest1, model = arima.model1x)
fc1ax <- fitted(arima.model1ax)

#plotting the actual and forecasted test values
plot(arimaTest1, type = "l", ylab = "Price", main = "Test model 1")
lines(fc1a, col = "blue", type = "l")

plot(arimaTest1, type = "l", ylab = "Price", main = "Test model 1")
lines(fc1ay, col = "blue", type = "l")

plot(arimaTest1, type = "l", ylab = "Price", main = "Test model 1")
lines(fc1ax, col = "blue", type = "l")

#calculating evaluation criteria
sum(abs(arimaTest1 - fc1a))/ length(arimaTest1) # MAE
sum(abs((arimaTest1 - fc1a)/ arimaTest1))/ length(arimaTest1)  # MAPE
sum((arimaTest1 - fc1a)^2)/ length(arimaTest1) # RMSE

sum(abs(arimaTest1 - fc1ay))/ length(arimaTest1) # MAE
sum(abs((arimaTest1 - fc1ay)/ arimaTest1))/ length(arimaTest1)  # MAPE
sum((arimaTest1 - fc1ay)^2)/ length(arimaTest1) # RMSE

sum(abs(arimaTest1 - fc1ax))/ length(arimaTest1) # MAE
sum(abs((arimaTest1 - fc1ax)/ arimaTest1))/ length(arimaTest1)  # MAPE
sum((arimaTest1 - fc1ax)^2)/ length(arimaTest1) # RMSE


###### 2 ###### 

# hour-by-hour predictions - train
arima.model2 <- auto.arima(arima2Weeks2, trace = T)
fc2 <- fitted(arima.model2)

arima.model2y <- auto.arima(arima3Weeks2, trace = T)
fc2y <- fitted(arima.model2y)

arima.model2x <- auto.arima(arimaMonth2, trace = T)
fc2x <- fitted(arima.model2x)

#plotting the actual and forecasted training values
plot(arima2Weeks2, type = "l", ylab = "Price", main = "Train model 2")
lines(fc2, col = "blue", type = "l")

plot(arima3Weeks2, type = "l", ylab = "Price", main = "Train model 2")
lines(fc2y, col = "blue", type = "l")

plot(arimaMonth2, type = "l", ylab = "Price", main = "Train model 2")
lines(fc2x, col = "blue", type = "l")

#calculating evaluation criteria
sum(abs(arima2Weeks2 - fc2))/ length(arima2Weeks2) # MAE
sum(abs((arima2Weeks2 - fc2)/ arima2Weeks2))/ length(arima2Weeks2) # MAPE
sum((arima2Weeks2 - fc2)^2)/ length(arima2Weeks2) # RMSE

sum(abs(arima3Weeks2 - fc2y))/ length(arima3Weeks2) # MAE
sum(abs((arima3Weeks2 - fc2y)/ arima3Weeks2))/ length(arima3Weeks2) # MAPE
sum((arima3Weeks2 - fc2y)^2)/ length(arima3Weeks2) # RMSE

sum(abs(arimaMonth2 - fc2x))/ length(arimaMonth2) # MAE
sum(abs((arimaMonth2 - fc2x)/ arimaMonth2))/ length(arimaMonth2) # MAPE
sum((arimaMonth2 - fc2x)^2)/ length(arimaMonth2) # RMSE

# hour-by-hour predictions - test
arima.model2a <- Arima(arimaTest2, model = arima.model2)
fc2a <- fitted(arima.model2a)

arima.model2ay <- Arima(arimaTest2, model = arima.model2y)
fc2ay <- fitted(arima.model2ay)

arima.model2ax <- Arima(arimaTest2, model = arima.model2x)
fc2ax <- fitted(arima.model2ax)

#plotting the actual and forecasted test values
plot(arimaTest2, type = "l", ylab = "Price", main = "Test model 2")
lines(fc2a, col = "blue", type = "l")

plot(arimaTest2, type = "l", ylab = "Price", main = "Test model 2")
lines(fc2ay, col = "blue", type = "l")

plot(arimaTest2, type = "l", ylab = "Price", main = "Test model 2")
lines(fc2ax, col = "blue", type = "l")

# calculating evaluation criteria
sum(abs(arimaTest2 - fc2a))/ length(arimaTest2) # MAE
sum(abs((arimaTest2 - fc2a)/ arimaTest2))/ length(arimaTest2) # MAPE
sum((arimaTest2 - fc2a)^2)/ length(arimaTest2) # RMSE

sum(abs(arimaTest2 - fc2ay))/ length(arimaTest2) # MAE
sum(abs((arimaTest2 - fc2ay)/ arimaTest2))/ length(arimaTest2) # MAPE
sum((arimaTest2 - fc2ay)^2)/ length(arimaTest2) # RMSE

sum(abs(arimaTest2 - fc2ax))/ length(arimaTest2) # MAE
sum(abs((arimaTest2 - fc2ax)/ arimaTest2))/ length(arimaTest2) # MAPE
sum((arimaTest2 - fc2ax)^2)/ length(arimaTest2) # RMSE

###### 3 ###### 

# hour-by-hour predictions - train
arima.model3 <- auto.arima(arima2Weeks3, trace = T)
fc3 <- fitted(arima.model3)

arima.model3y <- auto.arima(arima3Weeks3, trace = T)
fc3y <- fitted(arima.model3y)

arima.model3x <- auto.arima(arimaMonth3, trace = T)
fc3x <- fitted(arima.model3x)

#plotting the actual and forecasted training values
plot(arima2Weeks3, type = "l", ylab = "Price", main = "Train model 3")
lines(fc3, col = "blue", type = "l")

plot(arima3Weeks3, type = "l", ylab = "Price", main = "Train model 3")
lines(fc3y, col = "blue", type = "l")

plot(arimaMonth3, type = "l", ylab = "Price", main = "Train model 3")
lines(fc3x, col = "blue", type = "l")

#calculating evaluation criteria
sum(abs(arima2Weeks3 - fc3))/ length(arima2Weeks3) # MAE
sum(abs((arima2Weeks3 - fc3)/ arima2Weeks2))/ length(arima2Weeks3) # MAPE
sum((arima2Weeks3 - fc3)^2)/ length(arima2Weeks3) # RMSE

sum(abs(arima3Weeks3 - fc3y))/ length(arima3Weeks3) # MAE
sum(abs((arima3Weeks3 - fc3y)/ arima3Weeks3))/ length(arima3Weeks3) # MAPE
sum((arima3Weeks3 - fc3y)^2)/ length(arima3Weeks3) # RMSE

sum(abs(arimaMonth3 - fc3x))/ length(arimaMonth3) # MAE
sum(abs((arimaMonth3 - fc3x)/ arimaMonth3))/ length(arimaMonth3) # MAPE
sum((arimaMonth3 - fc3x)^2)/ length(arimaMonth3) # RMSE

# hour-by-hour predictions - test
arima.model3a <- Arima(arimaTest3, model = arima.model3)
fc3a <- fitted(arima.model3a)

arima.model3ay <- Arima(arimaTest3, model = arima.model3y)
fc3ay <- fitted(arima.model3ay)

arima.model3ax <- Arima(arimaTest3, model = arima.model3x)
fc3ax <- fitted(arima.model3ax)

#plotting the actual and forecasted test values
plot(arimaTest3, type = "l", ylab = "Price", main = "Test model 3")
lines(fc3a, col = "blue", type = "l")

plot(arimaTest3, type = "l", ylab = "Price", main = "Test model 3")
lines(fc3ay, col = "blue", type = "l")

plot(arimaTest3, type = "l", ylab = "Price", main = "Test model 3")
lines(fc3ax, col = "blue", type = "l")

# calculating evaluation criteria
sum(abs(arimaTest3 - fc3a))/ length(arimaTest3) # MAE
sum(abs((arimaTest3 - fc3a)/ arimaTest3))/ length(arimaTest3) # MAPE
sum((arimaTest3 - fc3a)^2)/ length(arimaTest3) # RMSE

sum(abs(arimaTest3 - fc3ay))/ length(arimaTest3) # MAE
sum(abs((arimaTest3 - fc3ay)/ arimaTest3))/ length(arimaTest3) # MAPE
sum((arimaTest3 - fc3ay)^2)/ length(arimaTest3) # RMSE

sum(abs(arimaTest3 - fc3ax))/ length(arimaTest3) # MAE
sum(abs((arimaTest3 - fc3ax)/ arimaTest3))/ length(arimaTest3) # MAPE
sum((arimaTest3 - fc3ax)^2)/ length(arimaTest3) # RMSE

