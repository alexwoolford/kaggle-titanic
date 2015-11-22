#
set.seed(1234)

train <- read.csv("data/train.csv", na.strings = "")
train <- train[, sapply(train, nlevels) < 32]

train$PassengerId <- NULL

train$Survived <- as.factor(train$Survived)

# Split the data into train and test
train.idx <- createDataPartition(train$Survived, p = 0.8, list = FALSE)

train.train <- train[train.idx, ]
train.test <- train[-train.idx, ]


model.rf <- randomForest(Survived ~ ., data = train.train, na.action = na.roughfix)


# Plot the importance 'tornado' chart
model.rf.importance <- importance(model.rf)
attribute <- row.names(data.frame(model.rf.importance))
model.rf.importance <- arrange(cbind(attribute, data.frame(model.rf.importance)), -MeanDecreaseGini)
model.rf.importance$attribute <- factor(model.rf.importance$attribute, levels = rev(model.rf.importance$attribute))
qplot(MeanDecreaseGini, attribute, data = model.rf.importance)

confusionMatrix(predict(model.rf, train.test), train.test$Survived)



train <- read.csv("data/train.csv", na.strings = "")
train <- train[, sapply(train, nlevels) < 32]

train$PassengerId <- NULL

train$Survived <- as.factor(train$Survived)

model.rf <- randomForest(Survived ~ ., data = train, na.action = na.roughfix)


test <- read.csv("data/test.csv", na.strings = "")
test <- test[, sapply(test, nlevels) < 32]

test.passenger.ids <- test$PassengerId
test$PassengerId <- NULL

test.roughfix <- na.roughfix(test)

predict(model.rf, test.roughfix)

model.rf.predictions <- data.frame(cbind(test.passenger.ids, as.character(predict(model.rf, test.roughfix))))
names(model.rf.predictions) <- c("PassengerId", "Survived")

write.csv(model.rf.predictions, file = "titanic_test_predictions.csv", row.names = FALSE, quote = FALSE)






