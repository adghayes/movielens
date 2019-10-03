################################
# Model Data + Calculate RMSEs #
################################
library(tidyverse)
library(caret)
load("./data/movielens.rda")

###################################
#   SIMPLE AVERAGE APPROACH (c)   #
###################################
mu <- edx %>% pull(rating) %>% mean()

c_predict <- function(newdata){
  rep(mu, nrow(newdata))
}

c_prediction <- c_predict(validation)
c_RMSE <- RMSE(validation$rating, c_prediction)

###########################################
#   MOVIE + USER EFFECTS APPROACH (muf)   #
###########################################
movie_effects <- edx %>% 
  group_by(movieId) %>%
  summarize(m_i = mean(rating - mu), n_i = n())

user_effects <- edx %>% 
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarize(u_j = mean(rating - mu - m_i), n_j = n())

muf_predict <- function(newdata){
  newdata %>% 
    left_join(movie_effects, by = "movieId") %>%
    left_join(user_effects, by = "userId") %>%
    mutate(rating = mu + m_i + u_j) %>%
    pull(rating)
}

muf_prediction <- muf_predict(validation)
muf_RMSE <- RMSE(validation$rating, muf_prediction)

##########################################################
# MOVIE + USER EFFECTS APPROACH w/ REGULARIZATION (mufr) #
##########################################################
calibrate_lambda <- FALSE
lambda <- c(3.87, 4.97)

if(calibrate_lambda){
  mufr_predict_pars <- function(train, test, lambda_m, lambda_u){
    movie_effects_r_temp <- train %>% 
      group_by(movieId) %>%
      summarize(
        m_i = sum(rating - mu)/(n() + lambda_m),
        n_i = n()
      )
    
    user_effects_r_temp <- train %>% 
      left_join(movie_effects_r_temp, by = "movieId") %>%
      group_by(userId) %>%
      summarize(
        u_j = sum(rating - mu - m_i)/(n() + lambda_u),
        n_j = n()
      )
    
    test %>% 
      left_join(movie_effects_r_temp, by = "movieId") %>%
      left_join(user_effects_r_temp, by = "userId") %>%
      mutate(rating = mu + m_i + u_j) %>% pull(rating)
  }
  
  # Determining lambda(s) requires cross-validation #
  k <- 5
  set.seed(23, sample.kind="Rounding")
  edx_folds <- createFolds(y = edx$rating, k = k)
  
  # Removing unknown users/movies from test partitions
  clean_edx_fold <- function(fold){
    test <- bind_cols(edx[fold,], index = fold)
    train <- edx[-fold,]
    test <- test %>%
      semi_join(train, by = "movieId") %>%
      semi_join(train, by = "userId")
    test$index
  }
  
  edx_fold_sizes_pre <- sapply(edx_folds, length) # pre-clean sizes
  edx_folds <- lapply(edx_folds, clean_edx_fold)
  edx_fold_sizes_post <- sapply(edx_folds, length) # post-clean sizes
  
  # Optimizing Lambda on the average RMSEs of cross validated sets
  cross_val_rmse <- function(lambda){
    mean(sapply(edx_folds, function(fold){
      RMSE(mufr_predict_pars(edx[-fold,],edx[fold,],lambda[1],lambda[2]), edx[fold,]$rating)
    }))
  }
  
  lambda_optim <- optim(par = lambda, fn = cross_val_rmse, control = list(maxit = 2))
  lambda <- lambda_optim$par
}

# Now with a decent lambda we can build and evaluate the tuned model
movie_effects_r <- edx %>% 
  group_by(movieId) %>%
  summarize(
    m_i = sum(rating - mu)/(n() + lambda[1]),
    n_i = n()
  )

user_effects_r <- edx %>% 
  left_join(movie_effects_r, by = "movieId") %>%
  group_by(userId) %>%
  summarize(
    u_j = sum(rating - mu - m_i)/(n() + lambda[2]),
    n_j = n()
  )

mufr_predict <- function(newdata){
  newdata %>% 
    left_join(movie_effects_r, by = "movieId") %>%
    left_join(user_effects_r, by = "userId") %>%
    mutate(rating = mu + m_i + u_j) %>% pull(rating)
}

mufr_prediction <- mufr_predict(validation)
mufr_RMSE <- RMSE(validation$rating, mufr_prediction)

###################
# Recommender Lab #
###################
# https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf

library(recommenderlab)
n_movies <- 500
n_users <- 10000

movie_res <- edx %>% 
  left_join(movie_effects_r, by = "movieId") %>%
  left_join(user_effects_r, by = "userId") %>%
  mutate(rating = rating - mu - m_i - u_j) %>%
  group_by(movieId) %>% summarise(n = n(), sd = sd(rating)) %>%
  mutate(wt = n*sd) %>% arrange(desc(wt))

top_n_movies <- movie_res %>% top_n(n_movies, wt = wt) %>% pull(movieId)

user_res <- edx %>% 
  left_join(movie_effects_r, by = "movieId") %>%
  left_join(user_effects_r, by = "userId") %>%
  mutate(rating = rating - mu - m_i - u_j) %>%
  filter(movieId %in% top_n_movies) %>%
  group_by(userId) %>% summarise(n = n(), sd = sd(rating)) %>%
  mutate(wt = n*sd) %>% arrange(desc(wt)) %>% 
  select(userId, n, sd, wt)

top_n_users <- user_res %>% top_n(n_users, wt = wt) %>% pull(userId)

q <- edx %>% 
  filter(userId %in% top_n_users & movieId %in% top_n_movies) %>%
  select(userId, movieId, rating) %>% as("realRatingMatrix")

rec <- Recommender(as(q, "realRatingMatrix"), method = "SVDF", parameter = 
                     list(normalize = "center", k = 35))

rl_predict <- function(newdata){
  
}

vx <- validation %>% 
  filter(movieId %in% top_n_movies & userId %in% top_n_users) %>%
  select(userId, movieId, rating)


p <- recommenderlab::predict(object = rec, newdata = q, type = "ratings")
pm <- p %>% 
  as("matrix") %>% 
  as.data.frame() %>%
  rownames_to_column(var = "userId") %>%
  remove_rownames() %>%
  gather(movieId, rating, -userId, na.rm = TRUE) %>%
  mutate(
    movieId = as.integer(movieId),
    userId = as.integer(userId)
  )

r1 <- RMSE(vx$rating, mufr_predict(vx))

ub <- vx %>% select(movieId, userId) %>%
  left_join(pm, by = c("movieId","userId")) %>% pull(rating) 

r2 <- RMSE(vx$rating, ub)
r2