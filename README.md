Modeling Movie Recommendations
================
Andrew Hayes
10/3/2019

This repository is a rating prediction project based off of the [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize), in which participants were challenged to surpass Netflix's accuracy in predicting users' movie ratings on the test set. The goal of the competition and of this project was to minimize the prediction error, specifically the root mean-squared error (RMSE). I implemented a series of increasingly complex models on a subset of the [MovieLens dataset](https://grouplens.org/datasets/movielens/10m/) and observe significant decreases in RMSE from a trivial, baseline algorithm. For a more detailed summary, please read **report.pdf**.

The project was done for edx's *Data Science: Capstone* course with methods from the *Data Science: Machine Learning* course. The recommenderlab R package was also used. For more information on recommenderlab, see [the paper](https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf) by one of recommenderlab's creator, Michael Hahsler. 
