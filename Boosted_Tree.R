#Trying out a boosted tree on the data

require(gbm)
library(data.table) 
data_pat <- as.data.table(readRDS(file = '/Users/dshenker/Desktop/Kharrazi_ML_Materials/data/data_pat_clean.rds')) # loading the data -- please adjust path to file based on your folder structure
admit_past = ifelse(data_pat$admit_flg_current == 1, "Yes", "No")
data_pat = data.frame(data_pat, admit_past)
set.seed(101)
index <- sample(1:nrow(data_pat),round(0.75*nrow(data_pat))) #get set of indices for training set
train <- data_pat[index,]
test <- data_pat[-index,]
boost.hospital = gbm(admit_flg_future~age + sex_num + admit_past + er_visit_ct_current + admit_ct_current + cost_current_total + hcc_ct + zip_income, data = train, distribution = "bernoulli", n.trees = 10000, shrinkage = 0.01, interaction.depth = 4) #train boosted tree
summary(boost.hospital)
predictions <- predict(boost.hospital, newdata = test, n.trees = 1000, type = "response")
n.trees = seq(from = 100, to = 10000, by = 100)
predmat = predict(boost.hospital, newdata = test, n.trees = n.trees, type = "response")
dim(predmat)

boost.err = with(test, apply(predmat == admit_flg_future, 2, mean )) #get the error
plot(n.trees, boost.err, ylab = "Accuracy", xlab = "# Trees", main = "Boosting Test Error") #plot error
