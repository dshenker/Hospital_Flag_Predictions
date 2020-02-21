#Neural net predicting future flag using more variables

library(data.table) 
library(neuralnet)
library(caret)
library(pROC)
data_pat <- as.data.table(readRDS(file = '/Users/dshenker/Desktop/Kharrazi_ML_Materials/data/data_pat_clean.rds')) # loading the data -- please adjust path to file based on your folder structure
data_pat <- select(data_pat, admit_flg_future, age, sex_num, admit_flg_current, er_visit_ct_current, admit_ct_current, cost_current_total, hcc_ct, zip_income, cost_future_total) #selecting the columns we'll use
set.seed(100)

#scaling the data
maxs <- apply(data_pat[,2:10], 2, max) #get max values 
mins <- apply(data_pat[,2:10], 2, min) #get min values
scaled_pat <- as.data.frame(scale(data_pat[,2:10],center = mins, scale = maxs - mins)) #scale the data
data_pat <- data.frame(data_pat$admit_flg_future, scaled_pat)

index <- sample(1:nrow(data_pat),round(0.75*nrow(data_pat))) #get indices for split
train <- data_pat[index,] #get training set
test <- data_pat[-index,] #get test set
nn = neuralnet(data_pat.admit_flg_future~age + sex_num + admit_flg_current + er_visit_ct_current + admit_ct_current + cost_current_total + hcc_ct + zip_income, data = train, hidden = 4, act.fct = "logistic", linear.output = FALSE) #train the neural net

predict_train = neuralnet::compute(nn, train) #get predictions on training set
prob_train <- predict_train$net.result #extract the probabilities
predict_test = neuralnet::compute(nn, test) #get predictions on test set
prob_test <- predict_test$net.result #extract the probabilities
test$data_pat.predict <- as.vector(predict_test$net.result)
model_admit_flg_roc <- roc(data_pat.admit_flg_future ~ data_pat.predict, data = test)     # apply the ROC function to calculate the ROC information
plot(model_admit_flg_roc) #plot the curve (looking for distance from diagonal line)
as.numeric(model_admit_flg_roc$auc) #get area under curve value
coords(model_admit_flg_roc, 'best', 'threshold', transpose = FALSE) #get best threshold                               

test_predictions <- ifelse(prob_test > 0.07937862, 1, 0) #use threshold to get zeros and ones
accuracy_test <- (test_predictions == test$data_pat.admit_flg_future) #test equality
pct_correct <- mean(accuracy_test) #get percent correct value
pct_correct #print value
table(test$data_pat.admit_flg_future, test_predictions) #print the confusion matrix




#Now, try neural net with two hidden layers
train_2 <- data_pat[index,] #get training set
test_2 <- data_pat[-index,] #get test set
set.seed(100)
nn_2 = neuralnet(data_pat.admit_flg_future~age + sex_num + admit_flg_current + er_visit_ct_current + admit_ct_current + cost_current_total + hcc_ct + zip_income, data = train_2, hidden = c(3, 3), act.fct = "logistic", linear.output = FALSE, threshold = .02) #train the neural net
predict_train_2 = neuralnet::compute(nn_2, train_2) #get predictions on training set
prob_train_2 <- predict_train_2$net.result #extract the probabilities
predict_test_2 = neuralnet::compute(nn_2, test_2) #get predictions on test set
prob_test_2 <- predict_test_2$net.result #extract the probabilities
test_2$data_pat.predict <- as.vector(predict_test_2$net.result)
model_admit_flg_roc_2 <- roc(data_pat.admit_flg_future ~ data_pat.predict, data = test_2)     # apply the ROC function to calculate the ROC information
plot(model_admit_flg_roc_2) #plot the curve (looking for distance from diagonal line)
as.numeric(model_admit_flg_roc_2$auc) #get area under curve value
coords(model_admit_flg_roc_2, 'best', 'threshold', transpose = FALSE) #get best threshold                               

test_predictions_2 <- ifelse(prob_test_2 > 0.1050435, 1, 0) #use threshold to get zeros and ones

accuracy_test_2 <- (test_predictions_2 == test_2$data_pat.admit_flg_future) #test equality
pct_correct_2 <- mean(accuracy_test_2) #get percent correct value
pct_correct_2 #print value
table(test_2$data_pat.admit_flg_future, test_predictions_2) #print the confusion matrix






