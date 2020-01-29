setwd('~/Desktop/STOR757/')
n = 10
r = 5
x = c(12, 11, 6, 12, 11, 0, 4, 6, 5, 6)

library("rstan")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# conjugate beta prior
prior = c(1, 1)

#stan
conjugate_data <- list(x=x, r=r, N = n, a=prior)
fitted_conjugate = stan_model(file='NB_conjugate.stan') #stan code compilation
stan_conjugate = sampling(fitted_conjugate, data = conjugate_data) #generate data
plot(stan_conjugate,show_density=TRUE)
summary(stan_conjugate)$summary

#direct plot comparison
conjugate_draws <- extract(stan_conjugate) #get draws from sampler
ggplot(data.frame(p=conjugate_draws$p),aes(p,))+geom_density()+
  stat_function(fun=dbeta, args=list(shape1=prior[1]+sum(x), shape2=prior[2]+n*r),color="green")
(sum(x)+prior[1])/(prior[1]+sum(x)+prior[2]+n*r)



#Jeffreys

#stan
jeffreys_data <- list(x=x, r=r, N = n)  
fitted_jeffreys = stan_model(file='NB_jeffreys.stan') #stan code compilation
stan_jeffreys = sampling(fitted_jeffreys, data = jeffreys_data) #generate data
plot(stan_jeffreys,show_density=TRUE)
summary(stan_jeffreys)$summary

#direct plot comparison
jeffreys_draws <- extract(stan_jeffreys) #get draws from sampler
ggplot(data.frame(p=jeffreys_draws$p),aes(p,))+geom_density()+
  stat_function(fun=dbeta, args=list(shape1=0.5+sum(x), shape2=n*r),color="blue")+
  stat_function(fun=dbeta, args=list(shape1=prior[1]+sum(x), shape2=prior[2]+n*r),color="green")
(0.5+sum(x))/(0.5+sum(x)+n*r)

mu = 2
sigma = 0.5
logit_data = list(x=x, r=r, mu=mu, sigma=sigma, N=n)

# logit transformation
#stan
fitted_normal = stan_model(file='NB_logit.stan') #stan code compilation
stan_normal = sampling(fitted_normal, data = logit_data) #generate data
plot(stan_normal, show_density=TRUE, par="p")
summary(stan_normal)$summary

#comparison of all three
normal_draws <- extract(stan_normal) #get draws from sampler
ggplot(data.frame(p=normal_draws$p),aes(p,))+geom_density()+
  stat_function(fun=dbeta, args=list(shape1=0.5+sum(x), shape2=n*r),color="blue")+
  stat_function(fun=dbeta, args=list(shape1=prior[1]+sum(x), shape2=prior[2]+n*r),color="green")

