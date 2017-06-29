library(readr)
library(ggplot2)
library(randomForest)
library(xgboost)

train <- read_csv("./data/train.csv")
test <- read_csv("./data/test.csv")
train_label <- read_csv("./data/train_label.csv")

status_bar <- ggplot(train_label, aes(status_group)) +
  geom_bar(aes(fill=status_group)) +
  geom_text(stat='count', aes(label=..count..), vjust = -0.2) +
  ggtitle("Status Group Frequncy") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(family= 'Arial Unicode MS'))

print(status_bar)
ggsave("status_bar.png", width = 10, height = 10, dpi = 200)

alldata <- rbind(train, test)
alldata = alldata[,-which(names(alldata) %in% c('id'))]

alldata$funder[alldata$funder == '0'] = NA
alldata$funder[alldata$funder == '-'] = NA

alldata$installer[alldata$funder == '0'] = NA
alldata$installer[alldata$funder == '-'] = NA

alldata$longitude[alldata$longitude == 0] = NA
alldata$latitude[alldata$latitude > -0.1] = NA

alldata$wpt_name[alldata$wpt_name == 'none'] = NA

alldata$construction_year[alldata$construction_year==0] = NA

alldata$public_meeting[alldata$public_meeting == 'True'] = 1
alldata$public_meeting[alldata$public_meeting == 'False'] = 0
alldata$public_meeting = as.numeric(alldata$public_meeting)

alldata$permit[alldata$permit == 'True'] = 1
alldata$permit[alldata$permit == 'False'] = 0
alldata$permit = as.numeric(alldata$permit)


na_ratio = colSums(is.na(alldata))/nrow(alldata)

na_ratio_df <- data.frame(feature = names(na_ratio), ratio = na_ratio)
na_ratio_bar <- ggplot(data = na_ratio_df) +
  geom_bar(aes(x = feature, y = ratio), stat = "identity", fill = 'deepskyblue', alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("NA Ratio") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(family= 'Arial Unicode MS'))

print(na_ratio_bar)
ggsave("na_ratio_bar.png", width = 10, height = 10, dpi = 200)


focused_na_ratio <- na_ratio[na_ratio != 0]
focused_na_ratio_df <- data.frame(feature = names(focused_na_ratio), ratio = focused_na_ratio)
focused_na_ratio_bar <- ggplot(data = focused_na_ratio_df) +
  geom_bar(aes(x = feature, y = ratio), stat = "identity", fill = 'deepskyblue', alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("NA Ratio") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(family= 'Arial Unicode MS'))

print(focused_na_ratio_bar)
ggsave("focused_na_ratio_bar.png", width = 10, height = 10, dpi = 200)




clean_train = alldata[1:nrow(train),]
clean_test = alldata[nrow(train)+1:nrow(alldata),]


feature_names = c()
chi2_stats = c()

for (i in 1:ncol(clean_train)){
  
  # print(names(clean_train)[i])
  
  x = clean_train[,i][[1]]
  x = x[!is.na(clean_train[,i])]
  y = train_label$status_group
  y = y[!is.na(clean_train[,i])]
  
  if(length(unique(x)) == 1){
    stat = 0
  }else{
    stat = chisq.test(x,y)$statistic
  }

  # print(stat)
  
  feature_names = c(feature_names, names(clean_train)[i])
  chi2_stats = c(chi2_stats, stat)
  
}


chi2_stat_df = data.frame(feature = feature_names, chi2_statistic = chi2_stats)
chi2_stat_bar <- ggplot(data = chi2_stat_df) +
  geom_bar(aes(x = feature, y = chi2_statistic), stat = "identity", fill = 'coral', alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Features to Status - Chi2 Statistic") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(family= 'Arial Unicode MS')) +
  coord_flip()

print(chi2_stat_bar)
ggsave("chi2_stat_bar.png", width = 10, height = 10, dpi = 200)


focused_chi2_stat_df = data.frame(feature = feature_names[order(chi2_stats, decreasing = T)][1:10], chi2_statistic = chi2_stats[order(chi2_stats, decreasing = T)][1:10])
focused_chi2_stat_df$feature = factor(focused_chi2_stat_df$feature, levels = rev(unique(focused_chi2_stat_df$feature)))
focused_chi2_stat_df$chi2_statistic = as.numeric(focused_chi2_stat_df$chi2_statistic)
focused_chi2_stat_bar <- ggplot(data = focused_chi2_stat_df) +
  geom_bar(aes(x = feature, y = chi2_statistic), stat = "identity", fill = 'coral', alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Features to Status - Chi2 Statistic") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(family= 'Arial Unicode MS')) +
  coord_flip()

print(focused_chi2_stat_bar)
ggsave("focused_chi2_stat_bar.png", width = 10, height = 10, dpi = 200)


numeric_alldata = alldata
alldata = as.data.frame(alldata)
train_label = as.data.frame(train_label)

for(i in 1:ncol(alldata)){
  print(i)
  print(names(alldata)[i])
  print(sum(is.na(alldata[,i])))
  if(lapply(alldata, mode)[i] == 'numeric'){
    replace_num = median(alldata[,i], na.rm = TRUE)
    print(replace_num)
    alldata[is.na(alldata[,i]), i] = replace_num
    numeric_alldata[,i] = as.numeric(alldata[,i])
  }else{
    alldata[is.na(alldata[,i]), i] = 'Other'
    numeric_alldata[,i] = as.numeric(factor(alldata[,i]))
  }
  
}



numeric_train = numeric_alldata[1:nrow(train),]
label = as.numeric(factor(train_label$status_group))

smp_size <- floor(0.9 * nrow(numeric_train))
train_ind <- sample(seq_len(nrow(numeric_train)), size = smp_size)

train_x = numeric_train[train_ind, ]
valid_x = numeric_train[-train_ind, ]
train_y = label[train_ind]
valid_y = label[-train_ind]



dtrain <- xgb.DMatrix(as.matrix(train_x), label = train_y)
dvalid <- xgb.DMatrix(as.matrix(valid_x), label = valid_y)

param <- list(max_depth = 15, eta = 0.03, silent = 1)
watchlist <- list(eval = dvalid, train = dtrain)
xgb_model = xgb.train(param, dtrain, watchlist = watchlist, nrounds = 300)
xgb_imp = xgb.importance(colnames(numeric_train), model = xgb_model)



xgb_imp_df = data.frame(feature = xgb_imp$Feature,  imp = xgb_imp$Gain)
xgb_imp_bar <- ggplot(data = xgb_imp_df) +
  geom_bar(aes(x = feature, y = imp), stat = "identity", fill = 'purple', alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Features to Status - XGB Importance") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(family= 'Arial Unicode MS')) +
  coord_flip()

print(xgb_imp_bar)
ggsave("xgb_imp_bar.png", width = 10, height = 10, dpi = 200)



focused_xgb_imp_df = data.frame(feature = xgb_imp$Feature[order(xgb_imp$Gain, decreasing = T)][1:10],  imp = xgb_imp$Gain[order(xgb_imp$Gain, decreasing = T)][1:10])
focused_xgb_imp_df$feature = factor(focused_xgb_imp_df$feature, levels = rev(unique(focused_xgb_imp_df$feature)))
focused_xgb_imp_df$imp = as.numeric(focused_xgb_imp_df$imp)
focused_xgb_imp_bar <- ggplot(data = focused_xgb_imp_df) +
  geom_bar(aes(x = feature, y = imp), stat = "identity", fill = 'purple', alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Features to Status - XGB Importance") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(family= 'Arial Unicode MS')) +
  coord_flip()

print(focused_xgb_imp_bar)
ggsave("focused_xgb_imp_bar.png", width = 10, height = 10, dpi = 200)



# rf = randomForest(
#                     x = train_x,
#                     y = train_y,
#                     ntree = 500,
#                     xtest = valid_x,
#                     ytest = valid_y
#                     
# )
# print(rf)
# 
# rf_imp_df = data.frame(feature = rownames(importance(rf)), imp = importance(rf))
# colnames(rf_imp_df) = c('feature','imp')
# rownames(rf_imp_df) = NULL
# 
# 
# rf_imp_bar <- ggplot(data = rf_imp_df) +
#   geom_bar(aes(x = feature, y = imp), stat = "identity", fill = 'green', alpha = 0.8) +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
#   ggtitle("Features to Status - RF Importance") +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   theme(text = element_text(family= 'Arial Unicode MS')) +
#   coord_flip()
# 
# print(rf_imp_bar)
# ggsave("rf_imp_bar.png", width = 10, height = 10, dpi = 200)
# 
# 
# 
# focused_rf_imp_df = data.frame(feature = rownames(importance(rf))[order(importance(rf), decreasing = T)][1:10],  imp = importance(rf)[order(importance(rf), decreasing = T)][1:10])
# colnames(focused_rf_imp_df) = c('feature','imp')
# rownames(focused_rf_imp_df) = NULL
# focused_rf_imp_df$feature = factor(focused_rf_imp_df$feature, levels = rev(unique(focused_rf_imp_df$feature)))
# focused_rf_imp_df$imp = as.numeric(focused_rf_imp_df$imp)
# focused_rf_imp_bar <- ggplot(data = focused_rf_imp_df) +
#   geom_bar(aes(x = feature, y = imp), stat = "identity", fill = 'green', alpha = 0.8) +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
#   ggtitle("Features to Status - RF Importance") +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   theme(text = element_text(family= 'Arial Unicode MS')) +
#   coord_flip()
# 
# print(focused_rf_imp_bar)
# ggsave("focused_rf_imp_bar.png", width = 10, height = 10, dpi = 200)
# 
