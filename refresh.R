# running these four lines will reproduce the entire project
source("dataset.R", echo = TRUE) # regenerage train and test sets
source("rmse.R", echo = TRUE) # rerun analysis
rmarkdown::render("report.Rmd", "pdf_document") # reproduce PDF report
rmarkdown::render("report.Rmd", "github_document", "README.md") # reproduce README.md