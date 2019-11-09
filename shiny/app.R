library(tidyverse)
library(shiny)
library(shinythemes)
library(ggvis)
load("./data.rda")

top_ns <- unlist(list(1,
  seq(5, 50, 5),
  seq(60, 100, 10),
  seq(150, 500, 50),
  seq(600, 1000, 100),
  seq(1500, 2500, 500)))

# 1-N mapping of movies to genres
movie_genres <- raw_effects %>%
  select(movieId, genres) %>%
  separate_rows(genres, sep = "\\|") %>%
  rename(genre = genres) %>%
  mutate(genre = if_else(genre == "(no genres listed)", 
                         "None Specified", genre))

# List of unique genres used for checkbox selection
unique_genres <- movie_genres %>%
  pull(genre) %>% unique()

# Truncate movie titles for display
raw_effects <- raw_effects %>%
  mutate(title = str_remove(title, "\\s+\\(.*\\)")) %>% 
  mutate(title = str_replace(title, "^(.*),\\s(The|A|An|Le|Les|La|Las|El|Lose|Das|Da|De)$", "\\2 \\1"))

raw_effects %>% 
  extract(title, c("name", "article"), regex = "(.*),\\s([a-zA-Z]*$)", remove = FALSE) %>% 
  filter((article %in% c("A", "The", "Le", "Les", "An", NA, "La", "Das", "El", "Los", "Da", "De"))) %>% 
  print(n = 40)

ui <- fluidPage(
  theme = shinytheme("cerulean"),
  titlePanel("MovieLens Regularization Explorer"),
  sidebarLayout(
    position = "right",
    sidebarPanel(
      sliderInput(inputId = "lambda", 
                  label = "Lambda", 
                  value = min(rmses_tbl$lambda), 
                  min = min(rmses_tbl$lambda), 
                  max = max(rmses_tbl$lambda),
                  step = lambda_step),
      conditionalPanel(
        condition = "input.tabs != 'Parameter Tuning'",
        checkboxInput(inputId = "customize_genres",
                      label = "Customize Genres Included?"),
        conditionalPanel(
          condition = "input.customize_genres == true",
          checkboxGroupInput(inputId = "selected_genres", 
                      label = "Genres:", 
                      choices = unique_genres, 
                      selected = unique_genres,
                      inline = TRUE,
                      width = "100%")
        )
      ),
      conditionalPanel(
        condition = "input.tabs == 'Effect Plot'",
        sliderInput(inputId = "top_n", 
                    label = "Top/Bottom Highlights", 
                    value = 500, 
                    min = 0, 
                    max = max(top_ns),
                    step = 5)
      )
    ),
    mainPanel(
      tabsetPanel(id = "tabs",
        tabPanel("Effect Plot", ggvisOutput("effects"),
                 htmlOutput("stats")),
        tabPanel("Top 10", tableOutput("top_movies")),
        tabPanel("Bottom 10", tableOutput("bottom_movies")),
        tabPanel("Parameter Tuning", ggvisOutput("tuning"))
      )
    )
  ),
  hr(),
  uiOutput("description")
)



server <- function(input, output, session) {
  
  
  # Reactive df of raw effects
  selected_effects <- reactive({
    req(input$selected_genres)
    if(input$customize_genres == FALSE){
      raw_effects
    } else {
      raw_effects %>%
        left_join(movie_genres, by = "movieId") %>%
        filter(genre %in% input$selected_genres) %>%
        select(-genre) %>%
        distinct()
    }
  })
  
  selected_effects_d <- debounce(selected_effects, 1000)
  
  # Reactive df of regularized effects
  effects <- reactive({
    # Regularize, Order, and Rank movies
    f.regularize(selected_effects_d(), input$lambda) %>%
      arrange(desc(effect)) %>%
      mutate(
        ranking = row_number(),
        rev_ranking = as.integer(max(ranking) + 1 - ranking)
      )
  })
  
  # Track our max possible highlights based on data
  # with max defined as less than a third of all movies
  max_top_n <- reactive({
    n_movies <- nrow(effects())
    max(top_ns[top_ns < (n_movies/3)])
  })
  
  # Update slider as max changes
  observe({
    cur_top_n <- isolate(input$top_n)
    updateSliderInput(
      session, 
      inputId = "top_n", 
      max = max_top_n(),
      value = min(max_top_n(), cur_top_n))
  })
  
  # Define tooltip used when hovering on scatterplot
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
  
  # Draw Dynamic Effect Plot
  effects_plot <- reactive({
    effects() %>%
      mutate(
        Place = if_else(
          ranking <= input$top_n,
          paste("Top ", input$top_n),
          if_else(
            rev_ranking <= input$top_n,
            paste("Bottom ", input$top_n), "Other"))
      ) %>%
      ggvis(~log10(n), ~effect) %>%
      layer_points(fill = ~Place, size := 50, size.hover := 200, 
                   fillOpacity := .2, fillOpacity.hover := .8,
                   key := ~movieId) %>%
      scale_ordinal("fill", range = c("#73A839", "#868e96", "#C71C22")) %>%
      scale_numeric("y", domain = c(-3.1, 1.6)) %>%
      scale_numeric("x", domain = c(-.2, 4.7)) %>%
      add_axis("x", title = "Number of Ratings (Log. 10 Scale)") %>%
      add_axis("y", title = "Movie Effect") %>%
      add_tooltip(movie_tooltip, "hover") %>%
      set_options(resizable = FALSE, width = "auto")
      
  })
  
  effects_plot %>% bind_shiny("effects")
  
  output$stats <- renderText({
    count <- nrow(selected_effects_d())
    average <- selected_effects_d() %>% 
      summarise(mu_r = mu + sum(effect*n)/sum(n)) %>%
      pull(mu_r)
    paste("<i>Number of Movies</i> = ", format(count, big.mark = ","),
          " | <i>Average Rating</i> = ", format(average, digits = 3),
          sep = "")
  })
  
  # Top 10 movie table
  output$top_movies <- renderTable(
    effects() %>%
      head(n = 10) %>%
      select(Ranking = ranking, 
             Title = title, 
             `Num. Reviews` = n, 
             `Mean Effect` = effect),
    striped = TRUE)
  
  # Bottom 10 movie table
  output$bottom_movies <- renderTable(
    effects() %>%
      tail(n = 10) %>%
      arrange(rev_ranking) %>%
      select(Ranking = rev_ranking, 
             Title = title, 
             `Num. Reviews` = n, 
             `Mean Effect` = effect),
    striped = TRUE)
  
  # Plot of tuning parameter, lambda, vs. RMSE
  tuning_plot <- reactive({
    rmses_tbl %>% 
      mutate(
        highlighted = factor(
          if_else(
            round(lambda,2) == input$lambda, 
            "Yes", "No"), levels = c("No", "Yes"))) %>%
      arrange(highlighted) %>%
      ggvis(~lambda, ~rmse, size = ~highlighted, shape = ~highlighted) %>%
      layer_points() %>%
      scale_ordinal("size", range = c(10,100)) %>%
      scale_ordinal("shape", range = c("dot","cross")) %>%
      add_axis("y", title = "Root Mean Squared Error (RMSE)", title_offset = 70) %>%
      add_axis("x", title = "Lambda") %>%
      ggvis::hide_legend(scales = c("size","shape")) %>%
      set_options(resizable = FALSE, width = "auto")
  })
  
  tuning_plot %>% bind_shiny("tuning")
  
  # Render HTML footer as dynamic UI
  # component, varies based on tab selection
  output$description <- renderUI({
    column(
      offset = .5, width = 11,
      fluidRow(
        if(input$tabs == "Effect Plot"){
          includeHTML("effect_plot.html")
        } else if(input$tabs == "Top 10"){
          includeHTML("top_10.html")
        } else if(input$tabs == "Bottom 10"){
          includeHTML("bottom_10.html")
        } else if(input$tabs == "Parameter Tuning"){
          includeHTML("parameter_tuning.html")
        } 
      )
    )
  })
  
  outputOptions(output, "bottom_movies", suspendWhenHidden = FALSE)
  outputOptions(output, "top_movies", suspendWhenHidden = FALSE)
  
}

shinyApp(ui = ui, server = server)
