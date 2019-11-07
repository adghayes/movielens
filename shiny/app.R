library(tidyverse)
library(shiny)
library(ggvis)
library(data.table)
load("./data.rda")
n_movies <- nrow(raw_effects)

ui <- fluidPage(
  titlePanel("Regularization Explorer"),

  sidebarPanel(
    sliderInput(inputId = "lambda", 
                label = "Lambda", 
                value = 1, 
                min = min(rmses_tbl$lambda), 
                max = max(rmses_tbl$lambda),
                step = lambda_step),
    sliderInput(inputId = "top_n", 
                label = "Top N (Highlight)", 
                value = 25, 
                min = 0, 
                max = 500,
                step = 5)
  ),
  mainPanel(
    tabsetPanel(
      tabPanel("Effect Plot", ggvisOutput("effects")),
      tabPanel("Top Movies", tableOutput("top_movies"))
    )
  )
 
)

server <- function(input, output) {
  
  effects <- reactive({
    f.effects(raw_effects, input$lambda) %>%
      arrange(desc(effect)) %>%
      mutate(ranking = row_number(),
             rev_ranking = n_movies - ranking,
             top = if_else(ranking <= input$top_n, "Yes", "No"))
  })

  
  movie_tooltip <- function(x) {
    if (is.null(x)) return(NULL)
    
    # Pick out the movie with this ID
    movie <- effects[effects$movieId == x$movieId, ]
    
    paste0("<b>", movie$title, "</b><br>",
           "Num. Ratings: " , movie$n, "<br>",
           "Movie Effect:", movie$effect
    )
  }
  
  scatter <- reactive({
    effects %>% 
      ggvis(~log10(n), ~effect) %>%
      layer_points(fill = ~top, size := 50, size.hover := 200, 
                   fillOpacity := .2, fillOpacity.hover := .8) %>%
      scale_ordinal("fill", range = c("gray", "purple")) %>%
      add_axis("x", title = "Number of Ratings (Log. 10 Scale)") %>%
      add_axis("y", title = "Average Rating") %>%
      add_legend("fill", title = paste("Top ", input$top_n), values = c("No", "Yes"))
      
  })
  
  scatter %>% bind_shiny("effects")
  
  output$top_movies <-renderTable(effects() %>%
    filter(ranking <= 20) %>%
    arrange(ranking) %>%
    select(Ranking = ranking, 
           Title = title, 
           `Number of Reviews` = n, 
           `Average Rating` = effect))
}

shinyApp(ui = ui, server = server)

