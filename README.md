### [Report](/report.pdf) - Movie Rating Prediction Models
### [Shiny App](https://adghayes.shinyapps.io/movielensRegularizationExplorer/) - Visualize Dataset and Effects of Regularization


## Description

This repository is a rating prediction project based off of the [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize), in which participants were challenged to surpass Netflix's accuracy in predicting users' movie ratings on the test set. The goal of the competition and of this project was to minimize the prediction error, specifically the root mean-squared error (RMSE). I implemented a series of increasingly complex models on a subset of the [MovieLens dataset](https://grouplens.org/datasets/movielens/10m/) and observe significant decreases in RMSE from a trivial, baseline algorithm. For a detailed summary, please read [report.pdf](/report.pdf). To see the code generating the models, see [rmse.R](/rmse.R). To play around with the regularization effect, check out the associated [shiny app](https://adghayes.shinyapps.io/movielensRegularizationExplorer/).

The project was done for edx's *Data Science: Capstone* course with methods from the *Data Science: Machine Learning* course. The recommenderlab R package was also used. For more information on recommenderlab, see [the paper by one of recommenderlab's creator](https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf), Michael Hahsler. 

## Files

The files in the repository include:

 - dataset.R - script to download and import MovieLens data and split into training and test sets. Stores data in data/movielens.rda (not in repo)
 - rmse.R - main analysis script which builds, tunes, and runs prediction algorithms. Loads data from data/movielens.rda and saves output statistics to data/output.rda
 - report.Rmd - reproducible report in R Markdown format
 - report.pdf - output of report.Rmd and the best way to read results
 - refresh.R - shortcut script to rerun entire pipeline: dataset creation, analysis, and report generation
 - shiny - contains shiny app source app.R, source data for app data.rda, and html files used in the shiny app
