#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram

shinyUI(fluidPage(
  # Application title
  titlePanel("Sentence Similarity"),
  
  sidebarLayout(
    # Sidebar with a slider and selection inputs
    sidebarPanel(
      #textInput("selection", "Please enter text:",width = '400px'),
      
      fileInput("file1", "Upload CSV File with at least 2 columns having column name 'Text' and 'Category'",
                multiple = FALSE,
                accept = c("text/csv",
                           "text/comma-separated-values,text/plain",
                           ".csv")),
      
      actionButton("update", "Cick Here!"),
      hr(),
      
      
      sliderInput("max",
                  "Maximum Number of Categories to be Desired:",
                  min = 2,  max = 300,  value = 20),
      
      #numericInput("freq1","Enter Minimum Frequency:",5)
      
      sliderInput("max1",
                  "Maximum Number of Iterations to be Run:",
                  min = 1,  max = 70, value = 20),
      
      selectInput("criteria",
                  "Criteria to be Considered",
                  c("Doc2Vec",'TfIdf','Word2Vec-PretrainedGoogle','Word2Vec-Text8Corpus')),
      
      downloadButton('download',"Download table with refined Category labels")
    ),
    
    # Show Word Cloud
    mainPanel(
      fixedRow(
        column(width = 12,offset =0,style='padding:0px;',h4(verbatimTextOutput("heading1",placeholder = FALSE)))
        
      ),
      tabPanel(
        "Original and Updated % Category Distribution",
        
        fluidRow(
          column(width = 5,offset=0,style='padding:0px;',tableOutput("table3")),
          column(width = 5,offset=0,style='padding:0px;',tableOutput("table2"))
          
        )),
      
      tableOutput("table1")
      
    )
  ),
  tags$style(type="text/css",
             ".shiny-output-error { visibility: hidden; }",
             ".shiny-output-error:before { visibility: hidden; }",
             HTML("
                  #heading1{font-weight:'bold';font-family:'arial';font-size:18px}")
             )
))
