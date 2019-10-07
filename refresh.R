# running this script will reproduce the entire project
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)
source("dataset.R", echo = TRUE) # regenerate train and test sets
source("rmse.R", echo = TRUE) # rerun analysis
rmarkdown::render("report.Rmd", "pdf_document") # reproduce PDF report
