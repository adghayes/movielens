apply_prediction <- function(f, newdata){
  actual <- newdata$rating
  predicted <- f(newdata)
  RMSE(actual,predicted)
}

# SIMPLE AVERAGE APPROACH (constant)
mu_all <- edx %>% pull(rating) %>% mean()

constant_predict <- function(newdata){
  newdata %>% mutate(rating = c(mu_all)) %>% pull(rating)
}

apply_prediction(constant_predict, edx)
apply_prediction(constant_predict, validation)

# MOVIE AVERAGE APPROACH (m)
m_i <- edx %>% group_by(movieId) %>%
  summarize(m_i = mean(rating) - mu_all)

m_predict <- function(newdata){
  newdata %>% left_join(m_i, by = "movieId") %>%
    mutate(rating = m_i + mu_all) %>%
    pull(rating)
}

apply_prediction(m_predict, edx)
apply_prediction(m_predict, validation)

m_prediction <- m_predict(validation)
validation %>% bind_cols(prediction = m_prediction) %>%
  group_by(movieId,title,genres) %>%
  summarize(rmse = RMSE(prediction, rating), numRating = n()) %>%
  arrange(desc(rmse)) %>%
  ggplot(aes(x = rmse, y = numRating)) + geom_point()

ggsave("figs/unregularized_m_model.png")

# MOVIE AVERAGE w/ REGULARIZATION (mr)
# To determine a lambda we will need to use cross-validation on the training
# set, so need to do k-fold split.
k <- 3
set.seed(23, sample.kind="Rounding")
edx_folds <- createFolds(y = edx$rating, k = k)

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

lambda <- optim(par = 5, fn = apply_mr_folds, upper = 25, lower = .1, method = "L-BFGS-B")


mov_r_predict <- function(newdata){
  m_ir_temp <- edx %>% group_by(movieId) %>%
    summarize(
      m_i = mean(rating) - mu_all,
      n_i = n()
    ) %>%
    mutate(m_i = n_i*m_i/(lambda_m + n_i))
  
  newdata %>% select(movieId, userId) %>% 
    left_join(m_ir_temp, by = "movieId") %>%
    mutate(rating = m_i + mu_all)
}

apply_prediction(mov_r_predict, validation)
