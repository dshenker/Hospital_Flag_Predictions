#A basic neural net using just age and sex to predict current flags

library(data.table) 
library(neuralnet)
library(caret)
library(pROC)
data_pat <- as.data.table(readRDS(file = '/Users/dshenker/Desktop/Kharrazi_ML_Materials/data/data_pat_clean.rds'))       # loading the data -- please adjust path to file based on your folder structure
dt <- data.frame(data_pat$age, data_pat$sex_num, data_pat$admit_flg_current)
set.seed(100)
index <- sample(1:nrow(dt),round(0.75*nrow(dt))) #get set of indices for training set
train <- dt[index,]
test <- dt[-index,]
nn = neuralnet(data_pat.admit_flg_current ~ data_pat.sex_num + data_pat.age, data = train, hidden = 4, act.fct = "logistic", linear.output = FALSE) #train neural net
predict_train = neuralnet::compute(nn, train) #get predictions on training set
prob_train <- predict_train$net.result #extract the probabilities
predict_test = neuralnet::compute(nn, test) #get predictions on test set
prob_test <- predict_test$net.result #extract the probabilities
test$data_pat.predict <- as.vector(predict_test$net.result)
model_admit_flg_roc <- roc(data_pat.admit_flg_current ~ data_pat.predict, data = test)     # apply the ROC function to calculate the ROC information
plot(model_admit_flg_roc) #plot the curve (looking for distance from diagonal line)
as.numeric(model_admit_flg_roc$auc) #get area under curve value
coords(model_admit_flg_roc, 'best', 'threshold', transpose = FALSE) #get best threshold                               
test_predictions <- ifelse(prob_test > 0.07604299, 1, 0) #use threshold to get zeros and ones
accuracy_test <- (test_predictions == test$data_pat.admit_flg_current) #test equality
pct_correct <- mean(accuracy_test) #get percent correct value
pct_correct #print value
table(test$data_pat.admit_flg_current, test_predictions) #print the confusion matrix