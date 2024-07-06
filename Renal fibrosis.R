library(tidyverse) # data tidy
library(caret) # machine learning
library(pROC) # ROC analysis
library(glmnet) # LASSO
library(rms) # nomogram
library(rmda) # decision curve
library(ModelGood) # calibration curve


rm(list = ls())

set.seed(80)

# import data
f_name <- 'D:/ROIforfinal60g/统计学处理/影像组学标签/PAV最好的分组/P/P.csv'
dt <- read_csv(f_name)

dt$Label <- as.factor(dt$Label)

idx_train <- createDataPartition(dt$Label, p = 0.7, list = FALSE)
dt_train_0 <- dt[idx_train, ]
dt_test_0 <- dt[-idx_train, ]

# preprocessing
st_prep <- preProcess(dt_train_0, method = c('center', 'scale', 'medianImpute'))
dt_train_1 <- predict(st_prep, dt_train_0)
dt_test_1 <- predict(st_prep, dt_test_0)


cor_mat <- cor(dt_train_1[, -1], method = 'spearman')
idx_ex <- findCorrelation(cor_mat, cutoff = 0.75) + 1
idx_inc <- setdiff(c(1:ncol(dt_train_1)), idx_ex)

dt_train_2 <- dt_train_1[, idx_inc]
dt_test_2 <- dt_test_1[, idx_inc]


y <- dt_train_2$Label
x <- as.matrix(dt_train_2[, -1])

# lasso
cv.fit <- cv.glmnet(x, y, family = 'binomial')
fit <- glmnet(x, y, family = 'binomial')

# plot lasso
plot(cv.fit)
plot(fit, s = cv.fit$lambda.min, xvar = 'lambda')
abline(v = log(cv.fit$lambda.min))

coefs <- coef(fit, s = cv.fit$lambda.min)

dt_feature <- tibble(Feature = coefs@Dimnames[[1]][coefs@i + 1], 
                     Coefficients = coefs@x)

# plot coefficients
dt_feature_tidy <- dt_feature[-1, ]
dt_feature_tidy$Coefficients <- abs(dt_feature_tidy$Coefficients)
dt_feature_tidy <- arrange(dt_feature_tidy, Coefficients)

p <- ggplot(data = dt_feature_tidy) + 
  geom_col(aes(x = Feature, y = Coefficients), fill = 'red') + 
  coord_flip() + scale_x_discrete(limits = rev(dt_feature_tidy$Feature))


res_train_bin <- predict(fit, newx = as.matrix(dt_train_2[, -1]), 
                         s = cv.fit$lambda.min, 
                         type = 'class')
res_train_prob <- as.vector(predict(fit, newx = as.matrix(dt_train_2[, -1]), 
                                    s = cv.fit$lambda.min, 
                                    type = 'response'))

res_test_bin <- predict(fit, newx = as.matrix(dt_test_2[, -1]), 
                        s = cv.fit$lambda.min, 
                        type = 'class')
res_test_prob <- as.vector(predict(fit, newx = as.matrix(dt_test_2[, -1]), 
                                   s = cv.fit$lambda.min, 
                                   type = 'response'))

dt_train_radscore <- tibble(Label = dt_train_2$Label, Radscore = res_train_prob)
dt_test_radscore <- tibble(Label = dt_test_2$Label, Radscore = res_test_prob)

roc_train <- pROC::roc(Label~Radscore, data = dt_train_radscore)
roc_test <- pROC::roc(Label~Radscore, data = dt_test_radscore)

# plot roc
plot(roc_train, col = 'red', legacy.axes = T)
plot(roc_test, print.auc=T, col = 'blue', add = T)
legend(x = 0.3, y = 0.2, legend = c(roc_train$auc, roc_test$auc), 
       col = c('red', 'blue'), lty = 1)


#plot nom
CTVa <-dt_train_2$chenshuai
dt_train_CTV <- tibble(Label = dt_train_2$Label, CTV = CTVa)
a <- cbind(dt_train_CTV,dt_train_radscore)
ddist_train <- datadist(a)
options(datadist = 'ddist_train')
mod_train <- lrm(Label~., data = a, x = TRUE, y = TRUE)
nom <- nomogram(mod_train, lp = F, fun = plogis, fun.at = seq(0, 1, by = 0.3))
plot(nom)




# plot nomogram
ddist_train <- datadist(dt_train_radscore)
options(datadist = 'ddist_train')
mod_train <- lrm(Label~., data = dt_train_radscore, x = TRUE, y = TRUE)
nom <- nomogram(mod_train, lp = F, fun = plogis, fun.at = seq(0, 1, by = 0.3))


# plot calibration
b <-CTVa+res_train_prob
bscore <- tibble(Label = dt_train_2$Label, Radscore = b)
calPlot2(mod_train, data = bscore, legend = F, 
         col = '#FF83FA', lty = 4, showY = F)
calPlot2(mod_train, data = dt_test_radscore, add = T, col = '#912CEE',
         lty = 5)

legend(x = 0.8, y = 0.2, legend = c('bscore', 'Testing'), 
       col = c('#FF83FA', '#912CEE'), lty = c(4, 5))

# plot decision curve
dt_dca <- dt_train_radscore
dt_dca$Label <- ifelse(dt_dca$Label == '0', 0, 1)
dca <- decision_curve(Label~Radscore, 
                      data = dt_dca)
plot_decision_curve(dca, confidence.intervals = F, 
                    col = c('red', 'blue', 'black'), 
                    curve.names = c('Radscore'))