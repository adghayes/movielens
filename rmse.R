################################
# Model Data + Calculate RMSEs #
################################
load("./data/movielens.rda")


### SIMPLE AVERAGE APPROACH (c) ###

mu <- edx %>% pull(rating) %>% mean()

c_predict <- function(newdata){
  rep(mu, nrow(newdata))
}

c_prediction <- c_predict(validation)
c_RMSE <- RMSE(validation$rating, c_prediction)

### MOVIE + USER EFFECTS APPROACH (muf) ###
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

### MOVIE + USER EFFECTS w/ REGULARIZATION APPROACH (mufr) ###

# To determine a lambda, we will need to use cross-validation on the training
# set, so need to do k-fold split.
k <- 2
set.seed(23, sample.kind="Rounding")
edx_folds <- createFolds(y = edx$rating, k = k)

# CUSTOM CROSS VALIDATION
# This function removes users and movies from each "test" fold that don't appear 
# in the rest of the edx set. As done
clean_edx_fold <- function(fold){
  test <- bind_cols(edx[fold,], index = fold)
  train <- edx[-fold,]
  test <- test %>%
    semi_join(train, by = "movieId") %>%
    semi_join(train, by = "userId")
  test$index
}

sapply(edx_folds, length) # pre-clean sizes
edx_folds <- lapply(edx_folds, clean_edx_fold)
sapply(edx_folds, length) # post-clean sizes

# MOVIE AVERAGE w/ REGULARIZATION (mr)
apply_mr <- function(train, test, lambda){
  mr_i <- train %>% group_by(movieId) %>%
    summarize(
      mr_i = sum(rating- mu_all)/(n() + lambda),
      n_i = n()
    )
  
  predicted <- test %>% left_join(mr_i, by = "movieId") %>%
    mutate(rating = mr_i + mu_all) %>% pull(rating)
  
  RMSE(predicted, test$rating)
}

apply_mr_folds <- function(lambda){
  mean(sapply(edx_folds, function(fold){
    apply_mr(edx[-fold,],edx[fold,],lambda)
  }))
}

lambdas <- seq(2,2.3,.05)
rmses <- sapply(lambdas, apply_mr_folds)
plot(lambdas,rmses)
lambda <- lambdas[which.min(rmses)]

mr_predict <- function(newdata){
  mr_i <- edx %>% group_by(movieId) %>%
    summarize(
      mr_i = sum(rating- mu_all)/(n() + lambda),
      n_i = n()
    )
  
  newdata %>% left_join(mr_i, by = "movieId") %>%
    mutate(rating = mr_i + mu_all) %>% pull(rating)
}

apply_prediction(mr_predict, validation)

# MOVIE + USER AVERAGE w/ REGULARIZATION (mur)
apply_mur <- function(train, test, lambda_m, lambda_u){
  mr_i <- train %>% group_by(movieId) %>%
    summarize(
      mr_i = sum(rating- mu_all)/(n() + lambda_m),
      n_i = n()
    )
  
  ur_i <- train %>% left_join(mr_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(
      ur_i = sum(rating - mu_all - mr_i)/(n() + lambda_u),
      n_i = n()
    )
  
  predicted <- test %>% 
    left_join(mr_i, by = "movieId") %>%
    left_join(ur_i, by = "userId") %>%
    mutate(rating = mu_all + mr_i + ur_i) %>% pull(rating)
  
  RMSE(predicted, test$rating)
}

apply_mur_folds <- function(lambda_m, lambda_u){
  mean(sapply(edx_folds, function(fold){
    apply_mur(edx[-fold,],edx[fold,],lambda_m, lambda_u)
  }))
}

lambda_m <- 4.05
lambda_u <- 4.95
rmses <- sapply(lambda_m, function(x){
  apply_mur_folds(x, lambda_u)
})
plot(lambda_m, rmses)

mur_predict <- function(newdata){
  mr_i <- edx %>% group_by(movieId) %>%
    summarize(
      mr_i = sum(rating- mu_all)/(n() + lambda_m),
      n_i = n()
    )
  
  ur_i <- edx %>% left_join(mr_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(
      ur_i = sum(rating - mu_all - mr_i)/(n() + lambda_u),
      n_i = n()
    )
  
  newdata %>% 
    left_join(mr_i, by = "movieId") %>%
    left_join(ur_i, by = "userId") %>%
    mutate(rating = mu_all + mr_i + ur_i) %>% pull(rating)
}

apply_prediction(mur_predict, validation)
