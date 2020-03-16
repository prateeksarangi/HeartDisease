tableAccuracy <- read.csv("~/HeartDisease/tableAccuracy.csv", header=FALSE, sep=";")
tableMSE <- read.csv("~/HeartDisease/tableMSE.csv", header=FALSE, sep=";")

meanAccuracy <- colMeans(tableAccuracy[sapply(tableAccuracy, is.numeric)])
meanMSE <- colMeans(tableMSE[sapply(tableMSE, is.numeric)])
itr <- 1:1500

plot(itr, meanAccuracy, "l", main = "Accuracy of Model",
     xlab = "Iteration", ylab = "Accuracy",
     col = "blue", lwd = 2)

plot(itr, meanMSE, "l", main = "Accuracy of Model",
     xlab = "Iteration", ylab = "Accuracy",
     col = "orange", lwd = 2)

LayerAcc <- read.csv("~/HeartDisease/LayerAcc.csv", header=FALSE)
LayerAcc.bar <- barplot(LayerAcc$V2, main="Car Distribution",
        names.arg=LayerAcc$V1, col = 'blue')
ablines(x = LayerAcc.bar, y = LayerAcc$V2[3])

