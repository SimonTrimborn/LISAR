# Algorithm to estimate LISAR model with LASSO, SCAD and Adaptive LASSO penalty
#
##############
# For details about the models and algorithms, please see:
# Zhang, K. and Trimborn, S. (2023) Influential assets in Large-Scale Vector AutoRegressive Models 
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4619531
##############
#
# Inputs:
# 'stock.subset' : xts object of stock prices with dimension T x N, T the number of observations and N the number of stocks 
# 'd' : Day from dataset to use for estimation
#
# Choices:
# 'Model' : Model to estimate, either LISAR.LASSO, LISAR.SCAD, LISAR.Adap.LASSO
# 'Select' : Select evaluation criteria for models, either MSFE, AIC, BIC
# 'reoptim' : When solution for lambda sequences was found, shall the solution be further refined for other lambdas? TRUE, FALSE
# 'Lags' : Maximum number of lags
# 'a.pen' : Tapering off parameter for SCAD penalty
# 'alpha.pens' : Vector or single number specifying the mixing parameters alpha. Shall be between (0,1). See paper for details
# 'gamma.pens' : Vector or single number specifying the gamma parameter for Adaptive LASSO. Shall be larger 0. 
# 'lambda1_seq' : Number between (0,1) indicating the tapering off parameter for the lambda sequence for the lag penalization
# 'lambda2_seq' : Number between (0,1) indicating the tapering off parameter for the lambda sequence for the column penalization
# 'lambda3_seq' : Number between (0,1) indicating the tapering off parameter for the lambda sequence for the individual parameter penalization
#
# Output: 
# EvaluateModel$Model : Optimal LISAR model
# EvaluateModel$Eval.Model : Evaluation parameters of the optimal LISAR model
# EvaluateModel$Lambdas.Model : Lambdas, alpha and gamma for optimal LISAR model

rm(list=ls(all=TRUE))

libraries = c("xts")
lapply(libraries, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(libraries, library, quietly = TRUE, character.only = TRUE)

source("LISAR_lambda_select.R")
source("LISAR_helper_functions.R")
source("LISAR_SCAD.R")
source("LISAR_LASSO.R")
source("LISAR_AdapLASSO.R")
source("LISAR_alpha_select.R")
source("LISAR_LASSO2.R")
source("LISAR_evaluation.R")

# Choose model: LISAR.LASSO, LISAR.SCAD, LISAR.Adap.LASSO
Model = "LISAR.LASSO" 
# Choose evaluation criteria for models: MSFE, AIC, BIC
Select = "MSFE"
# Refine the solution found under the lambda, alpha, gamma setting?: TRUE, FALSE
reoptim = FALSE 

# Maximum number of lags
Lags = 4 
# Tapering off parameter for SCAD
a.pen = 3.7
# Mixing parameter to assign stronger/weaker regularization to non-/influencers. 
# Can be a vector or single number. Shall be between (0,1)
alpha.pens = c(0.3, 0.5, 0.7)
# Gamma parameters for Adaptive LASSO. 
# Can be a vector or single number. Shall be between (0,infinity)
gamma.pens = c(0.5,1,2)

# Tapering off parameters for the 3 lambda sequences. Shall be between (0,1)
lambda1_seq = 0.5
lambda2_seq = 0.5
lambda3_seq = 0.5

# Control parameter for algorithm
eps1 = 0.0001
# Control parameter for refining solution to find optimal lambda combination
eps2 = 0.0001

########
# Data
########
load("Financials.RData")

stock.date = as.data.frame(table(format(index(stock.subset), "%Y-%m-%d")))
stock.date = as.vector(stock.date$Var1)

# select day
d = 1

dat = stock.subset[stock.date[d]]
dat = dat[ , colSums(is.na(dat)) != nrow(dat)] 
dat = na.omit(na.locf(diff(log(dat))))
dat = scale(dat)

TTs = nrow(dat)

N = ncol(dat)


tmp = t(as.matrix(dat))

# training data
Ydata = list()
for (i in 0:Lags) {
  Ydata[[i+1]] = tmp[, (Lags+1-i):(dim(tmp)[2]/3-i)] 
}

# evaluation data
Ydata_eval = list()
for (i in 0:Lags) {
  Ydata_eval[[i+1]] = tmp[, (dim(tmp)[2]/3+(1-i)):(2*dim(tmp)[2]/3-i)]
}

# out of sample data
Ydata_ofs = list()
for (i in 0:Lags) {
  Ydata_ofs[[i+1]] = tmp[, (2*dim(tmp)[2]/3+(1-i)):(dim(tmp)[2]-i)]
}

lambdas = LambdaSequence(Ydata = Ydata, Lags = Lags, N = N, sequence.l.1 = lambda1_seq, sequence.l.2 = lambda2_seq, sequence.l.3 = lambda3_seq)

lambda1 = lambdas[[1]] 
lambda2 = lambdas[[2]] 
lambda3 = lambdas[[3]] 

##########################
### LISAR SCAD
##########################
if (Model == "LISAR.SCAD") {
  store.time1 = Sys.time()
  store_model_alpha = list()
  
  for (k in 1:length(alpha.pens)) {
    alpha_opt = alpha.pens[k]
    
    message(paste0("START estimation ", Model, " with alpha = ", alpha_opt))
    pb = txtProgressBar(min = 0, max = length(lambda1) * length(lambda2)  * length(lambda3), initial = 0, style=3) 
    store_model_alpha[[k]] = LISAR_SCAD(N=N, TT=TT, Lags = Lags, Ydata = Ydata, lambda1 = lambda1, lambda2 = lambda2, lambda3 = lambda3, a.pen = a.pen, eps1 = eps1, alpha.pen=alpha_opt, reshape = FALSE)
    close(pb)
    message(paste0("DONE estimation ", Model, " with alpha = ", alpha_opt))
    
    lambda1_new = lambda1
    lambda2_new = lambda2
    lambda3_new = lambda3
    lambdas_and_opt_model = LambdaSelect(Select, store_model_alpha[[k]], Ydata_eval, Model)
    lambdas_upper_lower = lambdas_and_opt_model[[1]]
    new_opt_model = lambdas_and_opt_model[[2]]
    last_opt_model = lapply(new_opt_model, function(x) x + 1)
    lambda1_new = seq(lambda1_new[lambdas_upper_lower[[1]][1]], lambda1_new[lambdas_upper_lower[[1]][2]], length.out = 10)
    lambda2_new = seq(lambda2_new[lambdas_upper_lower[[2]][1]], lambda2_new[lambdas_upper_lower[[2]][2]], length.out = 10)
    lambda3_new = seq(lambda3_new[lambdas_upper_lower[[3]][1]], lambda3_new[lambdas_upper_lower[[3]][2]], length.out = 10)
    
    # Refine optimal lambda combination
    if (reoptim == TRUE) {
      message(paste0("START refined estimation ", Model, " with alpha = ", alpha_opt))
      while (any((unlist(last_opt_model) - unlist(new_opt_model)) > eps2)) {
        pb = txtProgressBar(min = 0, max = length(lambda1_new) * length(lambda2_new)  * length(lambda3_new), initial = 0, style=3) 
        store_model_alpha[[k]] = LISAR_SCAD(N=N, TT=TT, Lags = Lags, Ydata = Ydata, 
                                         lambda1 = lambda1_new, 
                                         lambda2 = lambda2_new, 
                                         lambda3 = lambda3_new, a.pen = a.pen, eps1 = eps1, 
                                         alpha.pen=alpha_opt, 
                                         reshape = TRUE)
        close(pb)
        
        lambdas_and_opt_model = LambdaSelect(Select, store_model_alpha[[k]], Ydata_eval, Model)
        lambdas_upper_lower = lambdas_and_opt_model[[1]]
        
        lambda1_select = lambdas_and_opt_model[[3]][[1]]
        lambda2_select = lambdas_and_opt_model[[3]][[2]]
        lambda3_select = lambdas_and_opt_model[[3]][[3]]
        
        lambda1_new = seq(lambda1_new[lambdas_upper_lower[[1]][1]], lambda1_new[lambdas_upper_lower[[1]][2]], length.out = 10)
        lambda2_new = seq(lambda2_new[lambdas_upper_lower[[2]][1]], lambda2_new[lambdas_upper_lower[[2]][2]], length.out = 10)
        lambda3_new = seq(lambda3_new[lambdas_upper_lower[[3]][1]], lambda3_new[lambdas_upper_lower[[3]][2]], length.out = 10)
        
        last_opt_model = new_opt_model
        new_opt_model = lambdas_and_opt_model[[2]]
      }
      message(paste0("DONE refined estimation ", Model, " with alpha = ", alpha_opt))
    }
  }
  
  alpha_new = AlphaSelect(Select, store_model_alpha,Ydata_eval,alpha.pens)
  which_model = which(alpha.pens == alpha_new[[1]])

  store_model_alpha_select = store_model_alpha[[which_model]]
  
  store.time2 = Sys.time()
  store.time2 - store.time1
}


##########################
### LISAR LASSO
##########################
if (Model == "LISAR.LASSO") {
  store.time1 = Sys.time()
  store_model_alpha = list()
  
  for (k in 1:length(alpha.pens)) {
    alpha_opt = alpha.pens[k]
    
    message(paste0("START estimation ", Model, " with alpha = ", alpha_opt))
    pb = txtProgressBar(min = 0, max = length(lambda1) * length(lambda2)  * length(lambda3), initial = 0, style=3)
    store_model_alpha[[k]] = LISAR_LASSO(N=N, TT=TT, Lags = Lags, Ydata = Ydata, lambda1 = lambda1, lambda2 = lambda2, lambda3 = lambda3, a.pen = a.pen, eps1 = eps1, alpha.pen=alpha_opt, reshape = FALSE)
    close(pb)
    message(paste0("DONE estimation ", Model, " with alpha = ", alpha_opt))
    
    lambda1_new = lambda1
    lambda2_new = lambda2
    lambda3_new = lambda3
    lambdas_and_opt_model = LambdaSelect(Select, store_model_alpha[[k]], Ydata_eval, Model)
    lambdas_upper_lower = lambdas_and_opt_model[[1]]
    new_opt_model = lambdas_and_opt_model[[2]]
    last_opt_model = lapply(new_opt_model, function(x) x + 1)
    lambda1_new = seq(lambda1_new[lambdas_upper_lower[[1]][1]], lambda1_new[lambdas_upper_lower[[1]][2]], length.out = 10)
    lambda2_new = seq(lambda2_new[lambdas_upper_lower[[2]][1]], lambda2_new[lambdas_upper_lower[[2]][2]], length.out = 10)
    lambda3_new = seq(lambda3_new[lambdas_upper_lower[[3]][1]], lambda3_new[lambdas_upper_lower[[3]][2]], length.out = 10)
    
    # Refine optimal lambda combination
    if (reoptim == TRUE) {
      message(paste0("START refined estimation ", Model, " with alpha = ", alpha_opt))
      while (any((unlist(last_opt_model) - unlist(new_opt_model)) > eps2)) {
        pb = txtProgressBar(min = 0, max = length(lambda1_new) * length(lambda2_new)  * length(lambda3_new), initial = 0, style=3) 
        store_model_alpha[[k]] = LISAR_LASSO(N=N, TT=TT, Lags = Lags, Ydata = Ydata, 
                                          lambda1 = lambda1_new, 
                                          lambda2 = lambda2_new, 
                                          lambda3 = lambda3_new, a.pen = a.pen, eps1 = eps1, 
                                          alpha.pen=alpha_opt, 
                                          reshape = TRUE)
        close(pb)
        
        lambdas_and_opt_model = LambdaSelect(Select, store_model_alpha[[k]], Ydata_eval, Model)
        lambdas_upper_lower = lambdas_and_opt_model[[1]]
        
        lambda1_select = lambdas_and_opt_model[[3]][[1]]
        lambda2_select = lambdas_and_opt_model[[3]][[2]]
        lambda3_select = lambdas_and_opt_model[[3]][[3]]
        
        lambda1_new = seq(lambda1_new[lambdas_upper_lower[[1]][1]], lambda1_new[lambdas_upper_lower[[1]][2]], length.out = 10)
        lambda2_new = seq(lambda2_new[lambdas_upper_lower[[2]][1]], lambda2_new[lambdas_upper_lower[[2]][2]], length.out = 10)
        lambda3_new = seq(lambda3_new[lambdas_upper_lower[[3]][1]], lambda3_new[lambdas_upper_lower[[3]][2]], length.out = 10)
        last_opt_model = new_opt_model
        new_opt_model = lambdas_and_opt_model[[2]]
      }
      message(paste0("DONE refined estimation ", Model, " with alpha = ", alpha_opt))
    }
  }
  
  alpha_new = AlphaSelect(Select, store_model_alpha,Ydata_eval,alpha.pens)
  which_model = which(alpha.pens == alpha_new[[1]])
  
  store_model_alpha_select = store_model_alpha[[which_model]]
  
  store.time2 = Sys.time()
  store.time2 - store.time1
}



##########################
### LISAR Adaptive LASSO
##########################
if (Model == "LISAR.Adap.LASSO") {
  store.time1 = Sys.time()
  store_model_alpha_select = list()
  
  for (adap.loop in 1:length(gamma.pens)) {
    gamma.pen = gamma.pens[adap.loop]
    store_model_alpha = list()
    for (k in 1:length(alpha.pens)) {
      alpha_opt = alpha.pens[k]
      
      message(paste0("START estimation ", Model, " with alpha = ", alpha_opt, " and gamma = ", gamma.pen))
      pb = txtProgressBar(min = 0, max = length(lambda1) * length(lambda2)  * length(lambda3), initial = 0, style=3)
      store_model_alpha[[k]] = LISAR_AdapLASSO(N=N, TT=TT, Lags = Lags, Ydata = Ydata, lambda1 = lambda1, lambda2 = lambda2, lambda3 = lambda3, gamma.pen = gamma.pen, eps1 = eps1, alpha.pen=alpha_opt, reshape = FALSE)
      close(pb)
      message(paste0("DONE estimation ", Model, " with alpha = ", alpha_opt, " and gamma = ", gamma.pen))
      
      lambda1_new = lambda1
      lambda2_new = lambda2
      lambda3_new = lambda3
      lambdas_and_opt_model = LambdaSelect(Select, store_model_alpha[[k]], Ydata_eval, Model)
      lambdas_upper_lower = lambdas_and_opt_model[[1]]
      new_opt_model = lambdas_and_opt_model[[2]]
      last_opt_model = lapply(new_opt_model, function(x) x + 1)
      lambda1_new = seq(lambda1_new[lambdas_upper_lower[[1]][1]], lambda1_new[lambdas_upper_lower[[1]][2]], length.out = 10)
      lambda2_new = seq(lambda2_new[lambdas_upper_lower[[2]][1]], lambda2_new[lambdas_upper_lower[[2]][2]], length.out = 10)
      lambda3_new = seq(lambda3_new[lambdas_upper_lower[[3]][1]], lambda3_new[lambdas_upper_lower[[3]][2]], length.out = 10)
      
      # Refine optimal lambda combination
      if (reoptim == TRUE) {
        message(paste0("START refined estimation ", Model, " with alpha = ", alpha_opt, " and gamma = ", gamma.pen))
        while (any((unlist(last_opt_model) - unlist(new_opt_model)) > eps2)) {

          pb = txtProgressBar(min = 0, max = length(lambda1_new) * length(lambda2_new)  * length(lambda3_new), initial = 0, style=3) 
          store_model_alpha[[k]] = LISAR_AdapLASSO(N=N, TT=TT, Lags = Lags, Ydata = Ydata, 
                                                lambda1 = lambda1_new, 
                                                lambda2 = lambda2_new, 
                                                lambda3 = lambda3_new, gamma.pen = gamma.pen, eps1 = eps1, 
                                                alpha.pen=alpha_opt, 
                                                reshape = TRUE)
          close(pb)
          
          lambdas_and_opt_model = LambdaSelect(Select, store_model_alpha[[k]], Ydata_eval, Model)
          lambdas_upper_lower = lambdas_and_opt_model[[1]]
          
          lambda1_select = lambdas_and_opt_model[[3]][[1]]
          lambda2_select = lambdas_and_opt_model[[3]][[2]]
          lambda3_select = lambdas_and_opt_model[[3]][[3]]
          
          lambda1_new = seq(lambda1_new[lambdas_upper_lower[[1]][1]], lambda1_new[lambdas_upper_lower[[1]][2]], length.out = 10)
          lambda2_new = seq(lambda2_new[lambdas_upper_lower[[2]][1]], lambda2_new[lambdas_upper_lower[[2]][2]], length.out = 10)
          lambda3_new = seq(lambda3_new[lambdas_upper_lower[[3]][1]], lambda3_new[lambdas_upper_lower[[3]][2]], length.out = 10)
          
          last_opt_model = new_opt_model
          new_opt_model = lambdas_and_opt_model[[2]]
        }
        message(paste0("DONE refined estimation ", Model, " with alpha = ", alpha_opt, " and gamma = ", gamma.pen))
      }
    }
    
    alpha_new = AlphaSelect(Select, store_model_alpha, Ydata_eval, alpha.pens)
    which_model = which(alpha.pens == alpha_new[[1]])
    store_model_alpha_select[[adap.loop]] = store_model_alpha[[which_model]]
  }
  store.time2 = Sys.time()
  store.time2 - store.time1
}

EvaluateModel = EvaluateLossFunction(Select, Model, store_model_alpha_select, alpha.pens, gamma.pens, 
                                   Ydata_eval, Ydata_ofs, 
                                   store.time2, 
                                   store.time1)


# ResModel = list(EvaluateModel, Ydata, Ydata_eval, Ydata_ofs)


