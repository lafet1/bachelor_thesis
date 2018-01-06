
library(ggplot2)

# densities
plot(density(arimaPerf[[1]][1:100]), ylim = c(0, 0.5))
lines(density(SVRARIMA4[[3]][[1]]), col = "red")
#lines(density(unlist(PSF[[1]][[1]])), col = "blue")

# defense graph
pdf(file="defense.pdf", height = 4, width = 5)
print(par(mfrow = c(1, 1)), plot(DayandHour, masterdata$Average, type = "l", xlab = "Day", ylab = "Price",
                                 cex.axis = 1.2, cex.lab = 1.2))
dev.off()


##### graph export #####

# ARIMA graph
pdf(file="ARIMA.pdf", height = 4, width = 5)
print(par(mfrow = c(1, 1)), plot(arimaPerf[[1]][1:100], type = "l", ylab = "MAE", xlab = "Week", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

pdf(file="ARIMArmse.pdf", height = 4, width = 5)
print(par(mfrow = c(1, 1)), plot(arimaPerf[[3]][1:100], type = "l", ylab = "RMSE", xlab = "Week", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

# PSF graph
pdf(file="PSF.pdf", height = 4, width = 5)
print(par(mfrow = c(1, 1)), plot(unlist(PSF[[1]][[1]]), type = "l", ylab = "MAE", xlab = "Week", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

pdf(file="PSFrmse.pdf", height = 4, width = 5)
print(par(mfrow = c(1, 1)), plot(unlist(PSF[[1]][[3]]), type = "l", ylab = "RMSE", xlab = "Week", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

# SVR graph
pdf(file="SVRmae.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(diffLag2[[2]][[1]], type = "l", ylab = "MAE", xlab = "Week", main = "2 lagged prices", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffLag3[[2]][[1]], type = "l", ylab = "MAE", xlab = "Week", main = "3 lagged prices", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffLag4[[2]][[1]], type = "l", ylab = "MAE", xlab = "Week", main = "4 lagged prices", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffLag5[[2]][[1]], type = "l", ylab = "MAE", xlab = "Week", main = "5 lagged prices", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

pdf(file="SVRrmse.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(diffLag2[[2]][[3]], type = "l", ylab = "RMSE", xlab = "Week", main = "2 lagged prices", cex.axis = 1.2, cex.lab = 1.2),  
      plot(diffLag3[[2]][[3]], type = "l", ylab = "RMSE", xlab = "Week", main = "3 lagged prices", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffLag4[[2]][[3]], type = "l", ylab = "RMSE", xlab = "Week", main = "4 lagged prices", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffLag5[[2]][[3]], type = "l", ylab = "RMSE", xlab = "Week", main = "5 lagged prices", cex.axis = 1.2, cex.lab = 1.2))
dev.off()


# SVRARIMA graph
pdf(file="SVRARIMAmae.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(diffSVRARIMA2[[3]][[1]], type = "l", ylab = "MAE", xlab = "Week", main = "2 lagged prices", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffSVRARIMA3[[3]][[1]], type = "l", ylab = "MAE", xlab = "Week", main = "3 lagged prices", cex.axis = 1.2, cex.lab = 1.2), 
      plot(diffSVRARIMA4[[3]][[1]], type = "l", ylab = "MAE", xlab = "Week", main = "4 lagged prices", cex.axis = 1.2, cex.lab = 1.2), 
      plot(diffSVRARIMA5[[3]][[1]], type = "l", ylab = "MAE", xlab = "Week", main = "5 lagged prices", cex.axis = 1.2, cex.lab = 1.2)
)
dev.off()

pdf(file="SVRARIMArmse.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(diffSVRARIMA2[[3]][[3]], type = "l", ylab = "RMSE", xlab = "Week", main = "2 lagged prices", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffSVRARIMA3[[3]][[3]], type = "l", ylab = "RMSE", xlab = "Week", main = "3 lagged prices", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffSVRARIMA4[[3]][[3]], type = "l", ylab = "RMSE", xlab = "Week", main = "4 lagged prices", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffSVRARIMA5[[3]][[3]], type = "l", ylab = "RMSE", xlab = "Week", main = "5 lagged prices", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

# SVR v ARIMA
pdf(file="SVRvARIMA1.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(diffLag2[[2]][[1]] - arimaPerf[[1]][1:100], type = "l", ylab = "MAE", xlab = "Week", main = "2 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffLag3[[2]][[1]] - arimaPerf[[1]][1:100], type = "l", ylab = "MAE", xlab = "Week", main = "3 lags", cex.axis = 1.2, cex.lab = 1.2), 
      plot(diffLag4[[2]][[1]] - arimaPerf[[1]][1:100], type = "l", ylab = "MAE", xlab = "Week", main = "4 lags", cex.axis = 1.2, cex.lab = 1.2), 
      plot(diffLag5[[2]][[1]] - arimaPerf[[1]][1:100], type = "l", ylab = "MAE", xlab = "Week", main = "5 lags", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

pdf(file="SVRvARIMA2.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(diffLag2[[2]][[3]] - arimaPerf[[3]][1:100], type = "l", ylab = "RMSE", xlab = "Week", main = "2 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffLag3[[2]][[3]] - arimaPerf[[3]][1:100], type = "l", ylab = "RMSE", xlab = "Week", main = "3 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffLag4[[2]][[3]] - arimaPerf[[3]][1:100], type = "l", ylab = "RMSE", xlab = "Week", main = "4 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffLag5[[2]][[3]] - arimaPerf[[3]][1:100], type = "l", ylab = "RMSE", xlab = "Week", main = "5 lags", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

# SVRARIMA v ARIMA
pdf(file="SVRARIMAvARIMA1.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(diffSVRARIMA2[[3]][[1]] - arimaPerf[[1]][1:100], type = "l", ylab = "MAE", xlab = "Week", main = "2 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffSVRARIMA3[[3]][[1]] - arimaPerf[[1]][1:100], type = "l", ylab = "MAE", xlab = "Week", main = "3 lags", cex.axis = 1.2, cex.lab = 1.2), 
      plot(diffSVRARIMA4[[3]][[1]] - arimaPerf[[1]][1:100], type = "l", ylab = "MAE", xlab = "Week", main = "4 lags", cex.axis = 1.2, cex.lab = 1.2), 
      plot(diffSVRARIMA5[[3]][[1]] - arimaPerf[[1]][1:100], type = "l", ylab = "MAE", xlab = "Week", main = "5 lags", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

pdf(file="SVRARIMAvARIMA2.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(diffSVRARIMA2[[3]][[3]] - arimaPerf[[3]][1:100], type = "l", ylab = "RMSE", xlab = "Week", main = "2 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffSVRARIMA3[[3]][[3]] - arimaPerf[[3]][1:100], type = "l", ylab = "RMSE", xlab = "Week", main = "3 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffSVRARIMA4[[3]][[3]] - arimaPerf[[3]][1:100], type = "l", ylab = "RMSE", xlab = "Week", main = "4 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(diffSVRARIMA5[[3]][[3]] - arimaPerf[[3]][1:100], type = "l", ylab = "RMSE", xlab = "Week", main = "5 lags", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

#Diebold-Mariano test
pdf(file="dmSVRARIMA.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(density(dmTestSVRARIMA2[[2]]) , type = "l", xlab = "DM statistic", main = "2 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(density(dmTestSVRARIMA2[[2]]), type = "l", xlab = "DM statistic", main = "3 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(density(dmTestSVRARIMA2[[2]]), type = "l", xlab = "DM statistic", main = "4 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(density(dmTestSVRARIMA2[[2]]), type = "l", xlab = "DM statistic", main = "5 lags", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

pdf(file="dmSVR.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(density(dmTestSVR2[[2]]), type = "l", xlab = "DM statistic", main = "2 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(density(dmTestSVR3[[2]]), type = "l", xlab = "DM statistic", main = "3 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(density(dmTestSVR4[[2]]), type = "l", xlab = "DM statistic", main = "4 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(density(dmTestSVR5[[2]]), type = "l", xlab = "DM statistic", main = "5 lags", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

pdf(file="dmSVRb.pdf", height = 8, width = 8)
print(par(mfrow = c(2, 2)), plot(density(dmTestSVR2b[[2]], na.rm = T), type = "l", xlab = "DM statistic", main = "2 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(density(dmTestSVR3b[[2]], na.rm = T), type = "l", xlab = "DM statistic", main = "3 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(density(dmTestSVR4b[[2]], na.rm = T), type = "l", xlab = "DM statistic", main = "4 lags", cex.axis = 1.2, cex.lab = 1.2),
      plot(density(dmTestSVR5b[[2]], na.rm = T), type = "l", xlab = "DM statistic", main = "5 lags", cex.axis = 1.2, cex.lab = 1.2))
dev.off()

# density
pdf(file="density.pdf", height = 5, width = 5)
print(par(mfrow = c(1, 1)), plot(density(diffSVRARIMA5[[3]][[1]]), xlab = "MAE", main = "", cex.axis = 1.2, cex.lab = 1.2, ylim = c(0, 0.4)), 
      lines(density(arimaPerf[[1]][1:100]), col = "red"),
      lines(density(diffLag5[[2]][[1]]), col = "blue"))
dev.off()













