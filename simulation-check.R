# Joshua Alley
# Texas A&M University
# Simulate fake data from ML model and attempt to recover parameters

# Initial idea from: http://modernstatisticalworkflow.blogspot.com/2017/04/an-easy-way-to-simulate-fake-data-from.html 


# working directory set through projects

# load packages 
library(MASS)
library(tidyverse)
library(rstan)
library(shinystan)
library(bayesplot)


# set-up global STAN options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


# set seed
set.seed(12)

# Set-up simulatiom data
n.sim = 2000 # 2000 simulated observations

# sparse matrix simulation of membership matrix
state.mem.mat.sim <- rsparsematrix(n.sim, 100,
                                   density = .1,
                                   rand.x = rnorm,
                                   symmetric = FALSE)
state.mem.mat.sim
state.mem.mat.sim <- as.matrix(state.mem.mat.sim)


# pull data into a list and add some simulations
sim.data.noest <- list(N = n.sim, # 2000 obs
                    y = rt(n.sim, 2), # t-dist outcome
                    state = rep(1:50, times = n.sim/50), # 50 states
                    S = 50, 
                    year = rep(1:200, times = n.sim/200), # 200 years
                    T = 200,
                    A = 100, # 100 alliances
                    Z = state.mem.mat.sim, 
                    X = cbind(rnorm(100, mean = 0, sd = 1), rbinom(100, size = 1, prob = .25)), # alliance level variables
                    L = 2, # two alliance-level variables
                    W = mvrnorm(n.sim, mu = c(1, -1), Sigma = matrix(c(4,2,2,1),2,2)), # simulate IVs from multivariate normal dist
                    M = 2, # two state-level variables
                run_estimation = 0
  )


# Run STAN model 
compiled.ml <- stan_model("ml-model-sim.stan")

# Run model to generate draws from posterior and parameter values
sim.out <- sampling(compiled.ml, data = sim.data.noest,
                    iter = 1000, warmup = 500, chains = 4)

# Check diagnostics
check_hmc_diagnostics(sim.out)



# Pull parameter values and simulated data from one draw of posterior (12th)
draw <- 12
y.sim <- extract(sim.out, pars = "y_sim")[[1]][draw, ] # outcome 
true.beta <- extract(sim.out, pars = "beta")[[1]][draw, ] # beta estimates
true.gamma <- extract(sim.out, pars = "gamma")[[1]][draw, ] # gamma estimates
true.lambda <- extract(sim.out, pars = "lambda")[[1]][draw, ] # lambda estimates
true.sigma <- extract(sim.out, pars = "sigma")[[1]][draw] # sigma estimates
true.sigma.state <- extract(sim.out, pars = "sigma_state")[[1]][draw] # sigma state estimates
true.sigma.year <- extract(sim.out, pars = "sigma_year")[[1]][draw] # sigma year estimates
true.sigma.all <- extract(sim.out, pars = "sigma_all")[[1]][draw] # sigma alliance estimates



# Fit model on draws from generated quantities block. 
# keep same data on IVs and structure as generated above
sim.data.est <- sim.data.noest # copy list 
sim.data.est$y <- y.sim # replace simulated y with y from STAN draw
sim.data.est$run_estimation <- 1 # estimate the likelihood


# run the model on this simulated data: attempt to recover parameters
sim.out.est <- sampling(compiled.ml, data = sim.data.est,
                    iter = 1000, warmup = 500, chains = 4)


# Check diagnostics
check_hmc_diagnostics(sim.out.est)


### Plot estimated parameters against "true values" from earlier simulated data
sim.est.sum <- extract(sim.out.est, pars = c("beta", "gamma", "lambda", "sigma", 
                                             "sigma_state", "sigma_year", "sigma_all"), 
                       permuted = TRUE)

colnames(sim.est.sum$beta) <- c("beta1", "beta2")
colnames(sim.est.sum$gamma) <- c("gamma1", "gamma2")
colnames(sim.est.sum$lambda) <- paste0("lambda", 1:100)


# Calculate accuracy of credible intervals: how many intervals contain true lambda? 
lambda.summary.sim <- summary(sim.out.est, pars = c("lambda"), probs = c(0.05, 0.95))$summary
lambda.summary.sim <- cbind.data.frame(true.lambda, lambda.summary.sim)

# create a dummy indicator of accurate coverage
lambda.summary.sim$accurate <- ifelse(lambda.summary.sim$true.lambda > lambda.summary.sim$`5%` & # greater than lower bound
                                       lambda.summary.sim$true.lambda < lambda.summary.sim$`95%`,
                                     1, 0) # smaller than upper bound
sum(lambda.summary.sim$accurate) 
# gets 90/100 pars right


# Start with beta- second-level regression parameters
mcmc_areas(sim.est.sum$beta, pars = c("beta1"), prob = .9) +
  vline_at(true.beta[1], color = "red", size = 2) 
mcmc_areas(sim.est.sum$beta, pars = c("beta2"), prob = .9) +
  vline_at(true.beta[2], color = "red", size = 2) 



# then gamma- first level regression parameters 
mcmc_areas(sim.est.sum$gamma, pars = c("gamma1"), prob = .9) +
  vline_at(true.gamma[1], color = "red", size = 2) 
mcmc_areas(sim.est.sum$gamma, pars = c("gamma2"), prob = .9) +
  vline_at(true.gamma[2], color = "red", size = 2) 


# now lambda parameters
mcmc_areas(sim.est.sum$lambda, pars = c("lambda1"), prob = .9) +
  vline_at(true.lambda[1], color = "red", size = 2) 
mcmc_areas(sim.est.sum$lambda, pars = c("lambda50"), prob = .9) +
  vline_at(true.lambda[50], color = "red", size = 2) 
mcmc_areas(sim.est.sum$lambda, pars = c("lambda100"), prob = .9) +
  vline_at(true.lambda[100], color = "red", size = 2) 




# check whether model recovers variance parameter and hyperparameters
# use ggplot b/c mcmc_areas doesn't take numeric vector input
# first-level variance sigma
ggplot(as.data.frame(sim.est.sum$sigma), aes(x = sim.est.sum$sigma)) + geom_density() +
 geom_vline(xintercept = c(true.sigma), color = "red", size = 2) 

# state variance hyperparameter
ggplot(as.data.frame(sim.est.sum$sigma_state), aes(x = sim.est.sum$sigma_state)) + geom_density() +
  geom_vline(xintercept = c(true.sigma.state), color = "red", size = 2) 


# year variance hyperparameter
ggplot(as.data.frame(sim.est.sum$sigma_year), aes(x = sim.est.sum$sigma_year)) + geom_density() +
  geom_vline(xintercept = c(true.sigma.year), color = "red", size = 2) 


# alliance coefficient variance parameter
ggplot(as.data.frame(sim.est.sum$sigma_all), aes(x = sim.est.sum$sigma_all)) + geom_density() +
  geom_vline(xintercept = c(true.sigma.all), color = "red", size = 2) 

