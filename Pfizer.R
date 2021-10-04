library(faraway)
library(readr)
pfizer <- read_csv("PFE.csv")
View(pfizer)

library(mlbench)
library(caret)
library(corrplot)

validationIndex <- createDataPartition(pfizer$Volume, p = .8, list = FALSE)
validation <- pfizer[-validationIndex,]
pfe <- pfizer[validationIndex,]

dim(pfe)
sapply(pfe, class)
head(pfe, n = 20)
pfe[,1] <- as.numeric(as.character((pfe[,1])))
pfe <- pfe[,-1]
sapply(pfe, class)
cor(pfe[,1:6])

#Histograms with attributes#
hist(pfe$Open, main = "Open")
hist(pfe$High, main = "High")
hist(pfe$Low, main = "Low")
hist(pfe$Close, main = "Close")
hist(pfe$`Adj Close`, main = "Adj Close")
hist(pfe$Volume, main = "Volume")

#Density Plots with attributes#
plot(density(pfe$Open), main = "Open")
plot(density(pfe$High), main = "High")
plot(density(pfe$Low), main = "Low")
plot(density(pfe$Close), main = "Close")
plot(density(pfe$`Adj Close`), main = "Adj Close")
plot(density(pfe$Volume), main = "Volume")

#Boxplots with attributes#
boxplot(pfe$Open, main = "Open")
boxplot(pfe$High, main = "High")
boxplot(pfe$Low, main = "Low")
boxplot(pfe$Close, main = "Close")
boxplot(pfe$`Adj Close`, main = "Adj Close")
boxplot(pfe$Volume, main = "Volume")

#Scatter plot Matrix#
pairs(pfe[,1:6])

#Correlation Plot#
correlations <- cor(pfe[,1:6])
corrplot(correlations, method = "circle")

#Run the algorithms using 10-fold cross-validation#
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "RMSE"

#Linear Model#
set.seed(7)
fit.lm <- train(Volume ~ ., data = pfe, method = "lm", metric = metric, preProc = c("center", "scale"), trControl = trainControl)

#General Linear Model
set.seed(7)
fit.glm <- train(Volume ~ ., data = pfe, method = "glm", metric = metric, preProc = c("center", "scale"), trControl = trainControl)

#Penalized Linear Model (GLMNET)#
set.seed(7)
fit.glmnet <- train(Volume ~ ., data = pfe, method = "glmnet", metric = metric, preProc = c("center", "scale"), trControl = trainControl)

#Support Vector Machines#
set.seed(7)
fit.svm <- train(Volume ~ ., data = pfe, method = "svmRadial", metric = metric, preProc = c("center", "scale"), trControl = trainControl)

#k-Nearest Neighbors#
set.seed(7)
fit.knn <- train(Volume ~ ., data = pfe, method = "knn", metric = metric, preProc = c("center", "scale"), trControl = trainControl)

outcome <- resamples(list(LM = fit.lm, GLM = fit.glm, GLMNET = fit.glmnet, SVM = fit.svm, KNN = fit.knn))
summary(outcome)
dotplot(outcome)

#Remove the correlated attributes#
#Find the attributes that are highly correlated#
set.seed(7)
ghost <- .7
highlyCorrelated <- findCorrelation(correlations, cutoff = ghost)

for (value in highlyCorrelated) {
  print(names(pfe)[value])
}

#Create a new dataset without highly correlated features#
pfeFeatures <- pfe[,-highlyCorrelated]
dim(pfeFeatures)

#Second Comparison without the highly correlated features#
#Run the algorithms using 10-fold cross-validation#
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "RMSE"

#Linear Model#
set.seed(7)
fit.lm <- train(Volume ~ ., data = pfeFeatures, method = "lm", metric = metric, preProc = c("center", "scale"), trControl = trainControl)

#General Linear Model
set.seed(7)
fit.glm <- train(Volume ~ ., data = pfeFeatures, method = "glm", metric = metric, preProc = c("center", "scale"), trControl = trainControl)

#Penalized Linear Model (GLMNET)#
set.seed(7)
fit.glmnet <- train(Volume ~ ., data = pfeFeatures, method = "glmnet", metric = metric, preProc = c("center", "scale"), trControl = trainControl)

#Support Vector Machines#
set.seed(7)
fit.svm <- train(Volume ~ ., data = pfeFeatures, method = "svmRadial", metric = metric, preProc = c("center", "scale"), trControl = trainControl)

#k-Nearest Neighbors#
set.seed(7)
fit.knn <- train(Volume ~ ., data = pfeFeatures, method = "knn", metric = metric, preProc = c("center", "scale"), trControl = trainControl)

outcome2 <- resamples(list(LM = fit.lm, GLM = fit.glm, GLMNET = fit.glmnet, SVM = fit.svm, KNN = fit.knn))
summary(outcome2)
dotplot(outcome2)

#Third Comparison using the Box-Cox Transformation#
#Run the algorithms using 10-fold cross-validation#
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "RMSE"

#Linear Model#
set.seed(7)
fit.lm <- train(Volume ~ ., data = pfe, method = "lm", metric = metric, preProc = c("center", "scale", "BoxCox"), trControl = trainControl)

#General Linear Model
set.seed(7)
fit.glm <- train(Volume ~ ., data = pfe, method = "glm", metric = metric, preProc = c("center", "scale", "BoxCox"), trControl = trainControl)

#Penalized Linear Model (GLMNET)#
set.seed(7)
fit.glmnet <- train(Volume ~ ., data = pfe, method = "glmnet", metric = metric, preProc = c("center", "scale", "BoxCox"), trControl = trainControl)

#Support Vector Machines#
set.seed(7)
fit.svm <- train(Volume ~ ., data = pfe, method = "svmRadial", metric = metric, preProc = c("center", "scale", "BoxCox"), trControl = trainControl)

#k-Nearest Neighbors#
set.seed(7)
fit.knn <- train(Volume ~ ., data = pfe, method = "knn", metric = metric, preProc = c("center", "scale", "BoxCox"), trControl = trainControl)

outcome3 <- resamples(list(LM = fit.lm, GLM = fit.glm, GLMNET = fit.glmnet, SVM = fit.svm, KNN = fit.knn))
summary(outcome3)
dotplot(outcome3)

#Apply Ensemble Methods#
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "RMSE"

#Random Forest#
set.seed(7)
fit.rf <- train(Volume ~ ., data = pfe, method = "rf", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Stochastic Gradient Boosting Model#
set.seed(7)
fit.gbm <- train(Volume ~ ., data = pfe, method = "gbm", metric = metric, preProc = c("BoxCox"), trControl = trainControl, verbose = FALSE)

#Cubist#
set.seed(7)
fit.cubist <- train(Volume ~ ., data = pfe, method = "cubist", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Compare the ensemble algorithms#
ensemble <- resamples(list(RF = fit.rf, GBM = fit.gbm, CUBIST = fit.cubist))
summary(ensemble)
dotplot(ensemble)

#Cubist Wins#

#Look deeper into winning model#
print(fit.cubist)

#Train the Final Model#
library(Cubist)
set.seed(7)
x <- pfe[,1:5]
y <- pfe[,6]
preprocessParams <- preProcess(x)
tX <- sample(1:nrow(pfe), floor(.8*nrow(pfe)))
p <- c("Open", "High", "Low", "Close", "Adj Close")
tXp <- pfe[tX, p]
tXr <- pfe$Volume[tX]
fM <- cubist(x = tXp, y = tXr, commitees = 20, neighbors = 5)
fM
summary(fM)
predictions <- predict(fM, tXp)

#Compute the RMSE & R^2#
rmse <- sqrt(mean((predictions - tXr)^2))
r2 <- cor(predictions, tXr)^2
print(rmse) #8992134#
print(r2) #0.7907419#

