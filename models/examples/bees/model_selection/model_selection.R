###########################################################
## MODEL SELECTION FOR BEES DATA                         ##
##                                                       ##
## 06.08.21                                              ##
## Source this file to run all analyses and make plots   ##
###########################################################


### PREPROCESSING ###

#clear environment
rm(list = ls())
gc()
#set working directory to current file location
setwd(dirname(parent.frame(2)$ofile))

#load packages
library(ggplot2)
library(rasterVis)
library(grid)
library(RColorBrewer)
library(geoelectrics)
library(binom)
col1 <- brewer.pal(n = 3, name = "Dark2")

#read data for parameter values (optimised points) and rational function values
dat_params <- read.table('dat_parameters.txt', row.names = 1)
dat_funcs  <- read.table('dat_functions.txt', row.names = 1)

#compute linear and sigmoidal parameter values from optimised points
linear_params <- function(r,d){
  res <- c(r)
  for(i in 1:9){
    x   <- r + i*d
    res <- c(res, x)
  }
  return(res)
}

sigmoidal_params <- function(k,v,n,r){
  res <- c(r)
  for (i in 1:9){
    x   <- r + ((v-r)/(1+(k/i)^n))
    res <- c(res, x)
  }
  return(res)
}

linear_p              <- linear_params(dat_params['linear',1], dat_params['linear',2])
dat_params['linear',] <- linear_p

sigmoid_p                <- sigmoidal_params(dat_params['sigmoidal',1], dat_params['sigmoidal',2],
                                             dat_params['sigmoidal',3], dat_params['sigmoidal',4])
dat_params['sigmoidal',] <- sigmoid_p




### NORMALIZATION (for linear model) ###

#read ranges of MH results
dat_mhranges <- read.table('mh_ranges.txt', row.names = 1)

#compute weights with min-max normalization
mhranges     <- dat_mhranges['upper',] - dat_mhranges['lower',]
norm_weights <- 1 - (mhranges - min(mhranges))/(max(mhranges) - min(mhranges))




### RESIDUAL SUM of SQUARES (RSS) ###

#for rational function values (compared to real data)
rss_func_lin <- sum((dat_funcs['data',] - dat_funcs['linear',])^2)
rss_func_sig <- sum((dat_funcs['data',] - dat_funcs['sigmoidal',])^2)

#for parameter values (compared to agnostic model)
rss_param_lin <- sum((dat_params['agnostic',] - dat_params['linear',])^2)
rss_param_sig <- sum((dat_params['agnostic',] - dat_params['sigmoidal',])^2)
#for parameter values, weighted by MH ranges
rss_param_lin_norm <- sum(((dat_params['agnostic',] - dat_params['linear',]) * norm_weights)^2)   
rss_param_sig_norm <- sum(((dat_params['agnostic',] - dat_params['sigmoidal',]) * norm_weights)^2)




### AKAIKE INFORMATION CRITERION (AIC) ###
cat("\n--------------------------------------------------------\n\n")
cat("AKAIKE INFORMATION CRITERION\n\n")

#AIC formula
aic <- function(n, rss, k){
  n * log(rss/n) + 2*k
}

#for rational function values
aic_func_lin <- aic(n = 11, rss = rss_func_lin, k = 2)
aic_func_sig <- aic(n = 11, rss = rss_func_sig, k = 4)
cat("Rational function values\n")
cat("Linear model:    ", aic_func_lin, "\n")
cat("Sigmoidal model: ", aic_func_sig, "\n\n")

#for parameter values
aic_param_lin <- aic(n = 10, rss = rss_param_lin, k = 2)
aic_param_sig <- aic(n = 10, rss = rss_param_sig, k = 4)
cat("Parameter values\n")
cat("Linear model:    ", aic_param_lin, "\n")
cat("Sigmoidal model: ", aic_param_sig, "\n\n")
#for parameter values, weighted by MH ranges
aic_param_lin_norm <- aic(n = 10, rss = rss_param_lin_norm, k = 2)
aic_param_sig_norm <- aic(n = 10, rss = rss_param_sig_norm, k = 4)
cat("Weighted parameter values\n")
cat("Linear model:    ", aic_param_lin_norm, "\n")
cat("Sigmoidal model: ", aic_param_sig_norm, "\n")


### R SQUARED ###
cat("\n--------------------------------------------------------\n\n")
cat("R SQUARED\n\n")

#R^2 formula 
rsquared <- function(rss, tss){
  1 - rss/tss
}

#for parameter values of linear model (compared to total sum of squares of agnostic model)
tss_param_agn <- sum((dat_params['agnostic',] - rowMeans(dat_params['agnostic',]))^2)
r2_param_lin  <- rsquared(rss_param_lin, tss_param_agn)
cat("Parameter values\n")
cat("Linear model:    ", r2_param_lin, "\n\n")
#for parameter values, weighted by MH ranges
tss_param_agn_norm <- sum(((dat_params['agnostic',] - rowMeans(dat_params['agnostic',])) * norm_weights)^2)
r2_param_lin_norm  <- rsquared(rss_param_lin_norm, tss_param_agn_norm)
cat("Weighted parameter values\n")
cat("Linear model:    ", r2_param_lin_norm, "\n")


### RESIDUAL PLOTS ###
cat("\n--------------------------------------------------------\n\n")
cat("RESIDUAL PLOTS\n\n")

#compute residuals for parameter values of linear model 
residuals_param_lin      <- dat_params['linear',] - dat_params['agnostic',]
residuals_param_lin_norm <- residuals_param_lin * norm_weights

#residual plots
png("residuals_linear.png", width = 8, height = 6, units = 'in', res = 300)
plot(x = c(0:9), y = residuals_param_lin, pch = 19, xlab = "i", 
     ylab = "Residuals", xlim = c(0,9), ylim = c(-0.1, 0.1), xaxt = "n")
abline(a = 0, b = 0, lty = 2)
axis(side = 1, seq(0,9,1))
dev.off()
cat("Residual plot saved in 'residuals_linear.png'\n")

png("residuals_linear_norm.png", width = 8, height = 6, units = 'in', res = 300)
plot(x = c(0:9), y = residuals_param_lin_norm, pch = 19, xlab = "i", 
     ylab = "Normalized Residuals", xlim = c(0,9), 
     ylim = c(-0.1, 0.1), xaxt = "n")
abline(a = 0, b = 0, lty = 2)
axis(side = 1, seq(0,9,1))
dev.off()
cat("Normalized residual plot saved in 'residuals_linear_norm.png'\n")




### Q-Q PLOTS ###
cat("\n--------------------------------------------------------\n\n")
cat("Q-Q PLOTS\n\n")

#quantiles of residuals; rank from lowest to highest
q_res  <- unlist(unname(sort(residuals_param_lin)))
#quantiles of normal distribution
q_norm <- qnorm(ppoints(1:10))

#qqplot: observed against theoretical
png("residuals_qq.png", width = 8, height = 6, units = 'in', res = 300)
plot(x = q_norm, y = q_res, xlab = "Theoretical normal quantiles", 
     ylab = "Residual quantiles", pch = 19, xlim = c(-2,2), ylim = c(-0.1,0.1))
abline(lm(q_res ~ q_norm), lty = 2)
dev.off()
cat("Q-Q plot saved in 'residuals_qq.png'\n")

#qqplot with normalized residuals
q_res_norm <- unlist(unname(sort(residuals_param_lin_norm)))
png("residuals_qq_norm.png", width = 8, height = 6, units = 'in', res = 300)
plot(x = q_norm, y = q_res_norm, xlab = "Theoretical normal quantiles", 
     ylab = "Residual quantiles", pch = 19, xlim = c(-2,2), ylim = c(-0.1,0.1))
abline(lm(q_res_norm ~ q_norm), lty = 2)
dev.off()
cat("Q-Q plot with normalized residuals saved in 'residuals_qq_norm.png'")
