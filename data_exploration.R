library(readr)
library(ggplot2)

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
ggsave("status_bar.png", width = 10, height = 10, dpi = 800)

alldata <- rbind(train, test)

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
ggsave("na_ratio_bar.png", width = 10, height = 10, dpi = 800)


focused_na_ratio <- na_ratio[na_ratio != 0]
focused_na_ratio_df <- data.frame(feature = names(focused_na_ratio), ratio = focused_na_ratio)
focused_na_ratio_bar <- ggplot(data = focused_na_ratio_df) +
  geom_bar(aes(x = feature, y = ratio), stat = "identity", fill = 'deepskyblue', alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("NA Ratio") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(family= 'Arial Unicode MS'))

print(focused_na_ratio_bar)
ggsave("focused_na_ratio_bar.png", width = 10, height = 10, dpi = 800)




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
ggsave("chi2_stat_bar.png", width = 10, height = 10, dpi = 800)


focused_chi2_stat_df = data.frame(feature = feature_names[chi2_stats > 25000], chi2_statistic = chi2_stats[chi2_stats > 25000])
focused_chi2_stat_bar <- ggplot(data = focused_chi2_stat_df) +
  geom_bar(aes(x = feature, y = chi2_statistic), stat = "identity", fill = 'coral', alpha = 0.8) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Features to Status - Chi2 Statistic") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(text = element_text(family= 'Arial Unicode MS')) +
  coord_flip()

print(focused_chi2_stat_bar)
ggsave("focused_chi2_stat_bar.png", width = 10, height = 10, dpi = 800)

