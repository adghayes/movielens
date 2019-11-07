library(tidyverse)
library(shiny)
library(shinythemes)
library(ggvis)
library(data.table)
load("./data.rda")
n_movies <- nrow(raw_effects)
raw_effects <- raw_effects %>%
  mutate(title = str_trunc(title, 35, "right"))

ui <- fluidPage(theme = shinytheme("cerulean"),
  titlePanel("MovieLens Regularization Explorer"),
  sidebarLayout(
    position = "right",
    sidebarPanel(
      sliderInput(inputId = "lambda", 
                  label = "Lambda", 
                  value = 0, 
                  min = min(rmses_tbl$lambda), 
                  max = max(rmses_tbl$lambda),
                  step = lambda_step),
      sliderInput(inputId = "top_n", 
                  label = "Top/Bottom Highlights", 
                  value = 500, 
                  min = 5, 
                  max = 500,
                  step = 5)
    ),
    mainPanel(
      tabsetPanel(id = "tabs",
        tabPanel("Effect Plot", ggvisOutput("effects")),
        tabPanel("Top 10", tableOutput("top_movies")),
        tabPanel("Bottom 10", tableOutput("bottom_movies")),
        tabPanel("Parameter Tuning", ggvisOutput("tuning"))
      )
    )
  ),
  hr(),
  htmlOutput("description")
)

server <- function(input, output) {
  
  effects <- reactive({
    f.effects(raw_effects, input$lambda) %>%
      arrange(desc(effect)) %>%
      mutate(ranking = row_number(),
             rev_ranking = as.integer(n_movies + 1 - ranking),
             Place = if_else(ranking <= input$top_n,
                           paste("Top ", input$top_n),
                           if_else(rev_ranking <= input$top_n,
                                   paste("Bottom ", input$top_n), 
                                   "Other"))
      )
  })
  
  movie_tooltip <- function(x) {
    if (is.null(x)) return(NULL)
    if (is.null(x$movieId)) return(NULL)
    
    # Pick out the movie with this ID
    const_effects <- isolate(effects())
    movie <- const_effects[const_effects$movieId == x$movieId, ]
    
    paste0("<b>", movie$title, "</b><br>",
           "Num. Ratings: " , format(movie$n,big.mark = ","), "<br>",
           "Movie Effect: ", format(movie$effect, digits = 2)
    )
  }
  
  effects_plot <- reactive({
    effects %>% 
      ggvis(~log10(n), ~effect) %>%
      layer_points(fill = ~Place, size := 50, size.hover := 200, 
                   fillOpacity := .2, fillOpacity.hover := .8,
                   key := ~movieId) %>%
      scale_ordinal("fill", range = c("purple", "gray", "brown")) %>%
      scale_numeric("y", domain = c(-3.1, 1.6)) %>%
      add_axis("x", title = "Number of Ratings (Log. 10 Scale)") %>%
      add_axis("y", title = "Movie Effect") %>%
      add_tooltip(movie_tooltip, "hover") %>%
      set_options(resizable = FALSE, width = "auto")
      
  })
  
  effects_plot %>% bind_shiny("effects")
  
  output$top_movies <- renderTable(
    effects() %>%
      head(n = 10) %>%
      select(Ranking = ranking, 
             Title = title, 
             `Num. Reviews` = n, 
             `Mean Effect` = effect),
    striped = TRUE)
  
  output$bottom_movies <- renderTable(
    effects() %>%
      tail(n = 10) %>%
      arrange(rev_ranking) %>%
      select(Ranking = rev_ranking, 
             Title = title, 
             `Num. Reviews` = n, 
             `Mean Effect` = effect),
    striped = TRUE)
  
  tuning_plot <- reactive({
    rmses_tbl %>% 
      mutate(highlighted = factor(
        if_else(round(lambda,2) == input$lambda, "Yes", "No"), 
        levels = c("No", "Yes"))) %>%
      arrange(highlighted) %>%
      ggvis(~lambda, ~rmse, size = ~highlighted, shape = ~highlighted) %>%
      layer_points() %>%
      scale_ordinal("size", range = c(10,100)) %>%
      scale_ordinal("shape", range = c("dot","cross")) %>%
      add_axis("y", title = "Root Mean Squared Error (RMSE)", title_offset = 70) %>%
      add_axis("x", title = "Lambda") %>%
      hide_legend(scales = c("size","shape")) %>%
      set_options(resizable = FALSE, width = "auto")
  })
  
  tuning_plot %>% bind_shiny("tuning")
  
  output$description <- reactive({
    if (input$tabs == "Effect Plot"){
      paste("<p>The data displayed here is derived from the ",
            "famous <a href = \"https://grouplens.org/datasets/movielens/",
            "\">MovieLens dataset</a> of users' movie ratings (10M). Here we ",
            "observe and play with the average ratings of each movie in ",
            "the 10M dataset. In the above plot, \"Movie Effect\" is the ",
            "deviation of a movie's average rating from the overall average",
            " rating.<br><br>",
            "<a href = \"https://rafalab.github.io/dsbook/",
            "large-datasets.html#regularization\"><b>Regularization</b></a> ",
            "is the process of normalizing group statistics based ",
            "on sample size. In the case of movie ratings, it means ",
            "regressing a movie's mean rating back to overal mean rating ",
            "for small sample sizes. Without doing so, our results indicate ",
            "that the best movies, by mean movie rating, are movies ",
            "that have only a handful of five star ratings. This is ",
            "problematic for estimating the true mean rating for any movie ",
            "with few ratings. ",
            "With regularization, each movie's mean is calculated with an added ",
            "parameter, lambda, in the demoninator, and in this applet you can ",
            "experiment with different values of lambda. For a more formal ",
            "summary of the problem and process of predicting movie means ",
            "check out the associated report on <a href = \"https://github.com/",
            "adghayes/movielens/blob/master/report.pdf\">my github</a>.<br><br>",
            "In the plot above, sample size is plotted ",
            "against adjusted average movie rating. With no adjustment, ",
            "lambda = 0, many of our \"Top\" movies have small sample ",
            "sizes. As you increase lambda, you will see the \"Top\"",
            "movies shift toward the more popular movies.</p>",
            sep = "")
    } else if (input$tabs == "Top 10"){
      paste("<p>Observe how the the Top 10 movies changes as ",
            "the regularization parameter changes. With no regularization ",
            "the Top 10 are all fairly obscure movies with very few, ",
            "albeit positive, reviews. As the regularization parameter increases ",
            "these movies are displaced by better-known movies recognizable ",
            "from any top movie list like <a href = \"https://www.imdb.com/",
            "search/title/?groups=top_250&sort=user_rating\">IMDb's.</a></p>",
            sep = "")
    } else if (input$tabs == "Bottom 10"){
      paste("<p>The Bottom 10 movies have a similar pattern as ",
            "the Top 10 in that as lambda increases, movies ",
            "with a few ratings are replaced with more reliably ",
            "bad movies, as expected. However, unlike in the Top 10",
            " there are many movies which stay stably in the Bottom 10 ",
            "even as lambda increases. One hypothesis for why this is the ",
            "case is that there are no bad movies with incredibly high ",
            "sample sizes. Good movies are rewatched and re-rated, so the ",
            "sample size gets very large whereas bad movies a simply watched ",
            "less. As a result, regularization is less impactful.</p>",
            sep = "")
    } else if (input$tabs == "Parameter Tuning"){
      paste("<p>A common use for rating prediction models is to predict ",
            "the users' ratings for movies they haven't seen yet. In ",
            "a simple popularity model, in which we predict users rate ",
            "movies according to the average rating for that movie, ",
            "regularization leads to better predictions. After splitting ",
            "the MovieLens dataset into a 9/10 training set and a 1/10 ",
            "test set, predictions were made on the training set for ",
            "various values of lambda, plotted above.</p>",
            sep = "")
    } 
  })
}

shinyApp(ui = ui, server = server)
