library(boot)
library(plyr)
library(dplyr)
library(nlme)
library(lme4)

df_1 <- read.table("../data/experiment_raw_nld_all_words.csv", header = T, sep = ",", comment.char="", quote="")
df_1$language = "Dutch"
df_1$freq = scale(log10(df_1$freq + 1))
df_1$rt = scale(log10(df_1$rt+1))
df_1$length = scale(df_1$length)

df_2 <- read.table("../data/experiment_raw_eng-uk_all_words.csv", header = T, sep = ",", comment.char="", quote="")
df_2$language = "English"
df_2$freq = scale(log10(df_2$freq + 1))
df_2$rt = scale(log10(df_2$rt+1))
df_2$length = scale(df_2$length)

df_3 <- read.table("../data/experiment_raw_fra_all_words.csv", header = T, sep = ",", comment.char="", quote="")
df_3$language = "French"
df_3$freq = scale(log10(df_3$freq + 1))
df_3$rt = scale(log10(df_3$rt+1))
df_3$length = scale(df_3$length)

# Bootstrap 95% CI for R-Squared
# function to obtain R-Squared from the data
rsq <- function(formula, formula_2, data, indices) {
  d <- data[indices,] # allows boot to select sample
  fit <- lm(formula, data=d)
  fit_2 <- lm(formula_2, data=d)
  r_2 = summary(fit)$adj.r.squared
  r_1 = summary(fit_2)$adj.r.squared
  return(c(r_1 - r_2, r_1, r_2))
}

df_d = df_1

df_symbol = df_d[df_d$o_f == "one hot",]
df_wickel = df_d[df_d$o_f == "wickelfeatures",]

df_merge = merge(df_wickel, df_symbol, by=c("language", "iter", "freq", "length", "ortho_form", "rt"))

results_dutch <- boot(data=df_merge, statistic=rsq,
                 R=100000, formula=rt ~ length + freq + score.x,
                 formula_2=rt ~ length + freq + score.y)

print("Done symbol")
print("Done Dutch")

df_d = df_2

df_wickel = df_d[df_d$o_f == "wickelfeatures",]
df_symbol = df_d[df_d$o_f == "one hot",]
df_old = df_d[df_d$o_f == "old_20",]

df_merge = merge(df_old, df_symbol, by=c("language", "iter", "freq", "length", "ortho_form", "rt"))

results_english_symbol <- boot(data=df_merge, statistic=rsq,
                R=100000, formula=rt ~ length + freq + score.x,
                formula_2=rt ~ length + freq + score.y)

df_merge = merge(df_old, df_wickel, by=c("language", "iter", "freq", "length", "ortho_form", "rt"))

print("Done symbol")

results_english_wickel <- boot(data=df_merge, statistic=rsq,
                        R=100000, formula=rt ~ length + freq + score.x,
                        formula_2=rt ~ length + freq + score.y)
print("Done wickel")
print("Done english")
df_d = df_3
df_symbol = df_d[df_d$o_f == "one hot",]
df_old = df_d[df_d$o_f == "old_20",]
df_wickel = df_d[df_d$o_f == "wickelfeatures",]

df_merge = merge(df_old, df_symbol, by=c("language", "iter", "freq", "length", "ortho_form", "rt"))

results_french_symbol <- boot(data=df_merge, statistic=rsq,
                               R=100000, formula=rt ~ length + freq + score.x,
                               formula_2=rt ~ length + freq + score.y)

print("Done symbol")

df_merge = merge(df_old, df_wickel, by=c("language", "iter", "freq", "length", "ortho_form", "rt"))

results_french_wickel <- boot(data=df_merge, statistic=rsq,
                               R=100000, formula=rt ~ length + freq + score.x,
                               formula_2=rt ~ length + freq + score.y)
print("Done wickel")
print("Done")
