library(ggplot2)
library(reshape2) # library for reshaping data (tall-narrow <-> short-wide)

setwd("~/OneDrive/PostgraduateProject/dataPreProcess") 

library(readr)
pop3SSE <- read_csv("pop3SSE.csv", col_names = c("para", "attribute", "value"))

ggplot(data= pop3SSE[c(1:12),c(1:3)], aes(x=para, y=value))+
  geom_bar(stat = "identity")+
  theme_classic()+
  xlab("parameter")+
  labs(color = "L")+
  ylab("SSE")


pop3outRaw <- read_csv("pop3outRaw.csv")

pop3out = melt(pop3outRaw, id.vars=c("id"))

ggplot(data = pop3out, aes(x = variable, y=value))+
  geom_boxplot(fill = "lightblue", alpha = 0.24)+
  geom_point(data= pop3SSE[c(37:48),c(1:3)], aes(x=para, y=value), color="blue")+
  theme_classic()+
  xlab("parameter")+
  labs(color = "L")+
  ylab("value")
