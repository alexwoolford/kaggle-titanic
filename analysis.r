library(Amelia)
library(caret)
library(randomForest)
library(ggplot2)

set.seed(1234)

train <- read.csv("data/train.csv")

# The passenger ID is just an assigned integer, and is not a true attribute of the passenger. This column is excluded.
train$PassengerId <- NULL

# There are 891 records in the training data of which 177 were missing age data. Approx. 20% of the records had missing data.
# Undoubtedly, the records with missing values have informational value and, given the relatively small number of records in
# the dataset, it seems that we sould not simply discard them.
num.imputed.datasets <- 10
amelia.imputations <- amelia(train, m = num.imputed.datasets, idvars = c("Name", "Sex", "Ticket", "Cabin", "Embarked"))$imputations

imputed.ages.df <- NA
for (i in 1:num.imputed.datasets){
  temp.df <- data.frame(amelia.imputations[[i]]$Age)
  names(temp.df) <- paste("imp", i, sep = "_")
  imputed.ages.df <<- cbind(imputed.ages.df, temp.df)
}
imputed.ages.df$imputed.ages.df <- NULL
imputed.ages <- as.integer(rowMeans(imputed.ages.df))

# The imputed ages were used to populate the missing ages in the training data.
train$Age[is.na(train$Age)] <- imputed.ages[is.na(train$Age)]
train$Age[train$Age < 0] <- 0

# Split the data into train and test
train.idx <- createDataPartition(train$Survived, p = 0.8, list = FALSE)

train.train <- train[train.idx, ]
train.test <- train[-train.idx, ]

# R's implementation of Random Forest doesn't work with factors with cardinality of more than 32 levels, so remove the
# higher cardinality factors:
train.train$Name <- NULL
train.train$Ticket <- NULL
train.train$Cabin <- NULL

# Train the model, and review the importance:
model.rf <- randomForest(Survived ~ ., data = train.train)
model.rf.importance <- importance(model.rf)
attribute <- row.names(data.frame(model.rf.importance))
model.rf.importance <- arrange(cbind(attribute, data.frame(model.rf.importance)), -IncNodePurity)

model.rf.importance$attribute <- factor(model.rf.importance$attribute, levels = rev(model.rf.importance$attribute))

qplot(IncNodePurity, attribute, data = model.rf.importance)

confusionMatrix(round(predict(model.rf, train.test), 0), train.test$Survived)

# Confusion Matrix and Statistics
# 
#           Reference
# Prediction   0   1
#          0 103  21
#          1   5  49

#            Accuracy : 0.8539          
#              95% CI : (0.7933, 0.9023)
# No Information Rate : 0.6067          
# P-Value [Acc > NIR] : 4.78e-13        



