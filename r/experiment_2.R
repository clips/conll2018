library(lattice)
library(tidyr)
library(plyr)
library(nlme)
library(lme4)
library(ggplot2)


df_1 <- read.table("experiment_mlp_nld_all_words.csv", header = T, sep = ",", comment.char="", quote="")
df_1$language = "Dutch"
df_1$freq = scale(log10(df_1$freq))
df_1$length = scale(df_1$length)

df_2 <- read.table("experiment_mlp_eng-uk_all_words.csv", header = T, sep = ",", comment.char="", quote="")
df_2$language = "English"
df_2$freq = scale(log10(df_2$freq))
df_2$length = scale(df_2$length)

df_3 <- read.table("experiment_mlp_fra_all_words.csv", header = T, sep = ",", comment.char="", quote="")
df_3$language = "French"
df_3$freq = scale(log10(df_3$freq))
df_3$length = scale(df_3$length)

df_d = df_1

df_fourteen = df_d[df_d$o_f == "fourteen",]
df_fourteen$score = scale(df_fourteen$score)[0:length(df_fourteen$score)]
df_seriol = df_d[df_d$o_f == "weighted bigrams",]
df_seriol$score = scale(df_seriol$score)[0:length(df_seriol$score)]
df_symbol = df_d[df_d$o_f == "one hot",]
df_symbol$score = scale(df_symbol$score)[0:length(df_symbol$score)]
df_wickel = df_d[df_d$o_f == "wickelfeatures",]
df_wickel$score = scale(df_wickel$score)[0:length(df_wickel$score)]

base = lm(rt ~ length + freq, data = df_fourteen)
fourteen = lm(rt ~ length + freq + score, data = df_fourteen)
seriol = lm(rt ~ length + freq + score, data = df_seriol)
symbol = lm(rt ~ length + freq + score, data = df_symbol)
wickel = lm(rt ~ length + freq + score, data = df_wickel)

s = summary(base); s$coefficients[c(2,3), 1]; s$adj.r.squared
s = summary(fourteen); s$coefficients[c(2,3,4), 1]; s$adj.r.squared
s = summary(symbol); s$coefficients[c(2,3,4), 1]; s$adj.r.squared
s = summary(seriol); s$coefficients[c(2,3,4), 1]; s$adj.r.squared
s = summary(wickel); s$coefficients[c(2,3,4), 1]; s$adj.r.squared

df_d = df_2

df_fourteen = df_d[df_d$o_f == "fourteen",]
df_fourteen$score = scale(df_fourteen$score)[0:length(df_fourteen$score)]
df_seriol = df_d[df_d$o_f == "weighted bigrams",]
df_seriol$score = scale(df_seriol$score)[0:length(df_seriol$score)]
df_symbol = df_d[df_d$o_f == "one hot",]
df_symbol$score = scale(df_symbol$score)[0:length(df_symbol$score)]
df_wickel = df_d[df_d$o_f == "wickelfeatures",]
df_wickel$score = scale(df_wickel$score)[0:length(df_wickel$score)]

base = lm(rt ~ length + freq, data = df_fourteen)
fourteen = lm(rt ~ length + freq + score, data = df_fourteen)
seriol = lm(rt ~ length + freq + score, data = df_seriol)
symbol = lm(rt ~ length + freq + score, data = df_symbol)
wickel = lm(rt ~ length + freq + score, data = df_wickel)

s = summary(base); s$coefficients[c(2,3), 1]; s$adj.r.squared
s = summary(fourteen); s$coefficients[c(2,3,4), 1]; s$adj.r.squared
s = summary(symbol); s$coefficients[c(2,3,4), 1]; s$adj.r.squared
s = summary(seriol); s$coefficients[c(2,3,4), 1]; s$adj.r.squared
s = summary(wickel); s$coefficients[c(2,3,4), 1]; s$adj.r.squared

df_d = df_3

df_fourteen = df_d[df_d$o_f == "fourteen",]
df_fourteen$score = scale(df_fourteen$score)[0:length(df_fourteen$score)]
df_seriol = df_d[df_d$o_f == "weighted bigrams",]
df_seriol$score = scale(df_seriol$score)[0:length(df_seriol$score)]
df_symbol = df_d[df_d$o_f == "one hot",]
df_symbol$score = scale(df_symbol$score)[0:length(df_symbol$score)]
df_wickel = df_d[df_d$o_f == "wickelfeatures",]
df_wickel$score = scale(df_wickel$score)[0:length(df_wickel$score)]

base = lm(rt ~ length + freq, data = df_fourteen)
fourteen = lm(rt ~ length + freq + score, data = df_fourteen)
seriol = lm(rt ~ length + freq + score, data = df_seriol)
symbol = lm(rt ~ length + freq + score, data = df_symbol)
wickel = lm(rt ~ length + freq + score, data = df_wickel)

s = summary(base); s$coefficients[c(2,3), 1]; s$adj.r.squared
s = summary(fourteen); s$coefficients[c(2,3,4), 1]; s$adj.r.squared
s = summary(symbol); s$coefficients[c(2,3,4), 1]; s$adj.r.squared
s = summary(seriol); s$coefficients[c(2,3,4), 1]; s$adj.r.squared
s = summary(wickel); s$coefficients[c(2,3,4), 1]; s$adj.r.squared
