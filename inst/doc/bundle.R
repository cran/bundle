## ---- include = FALSE---------------------------------------------------------
should_eval <- 
  rlang::is_installed("keras") && 
  rlang::is_installed("callr") &&
  rlang::is_installed("xgboost") 

should_eval <- 
  ifelse(should_eval, !is.null(tensorflow::tf_version()), should_eval)

knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = should_eval
)

## ----setup--------------------------------------------------------------------
library(bundle)

## ----setup-exts---------------------------------------------------------------
library(keras)
library(xgboost)

library(callr)

## ----mtcars-fit---------------------------------------------------------------
cars <- mtcars %>%
  as.matrix() %>%
  scale()

x_train <- cars[1:25, 2:ncol(cars)]
y_train <- cars[1:25, 1]

x_test <- cars[26:32, 2:ncol(cars)]
y_test <- cars[26:32, 1]

keras_fit <- 
  keras_model_sequential()  %>%
  layer_dense(units = 1, input_shape = ncol(x_train), activation = 'linear') %>%
  compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam(learning_rate = .01)
  ) 

keras_fit %>%
  fit(
    x = x_train, y = y_train,
    epochs = 100, batch_size = 1,
    verbose = 0
  )

## ----diagram-01, echo = FALSE, fig.alt = "A diagram showing a rectangle, labeled model object, and another rectangle, labeled predictions. The two are connected by an arrow from model object to predictions, with the label predict.", out.width = '100%'----
knitr::include_graphics("figures/diagram_01.png")

## ----predict-example----------------------------------------------------------
predict(keras_fit, x_test)

## ----callr-example------------------------------------------------------------
r(
  function(x) {
    x * 2
  },
  args = list(
    x = 1
  )
)

## ----keras-save---------------------------------------------------------------
temp_file <- tempfile()

saveRDS(keras_fit, file = temp_file)

## ----keras-fresh-rds, linewidth = 60, error = TRUE----------------------------
r(
  function(temp_file, new_data) {
    library(keras)
    
    model_object <- readRDS(file = temp_file)
    
    predict(model_object, new_data)
  },
  args = list(
    temp_file = temp_file,
    new_data = x_test
  )
)

## ----diagram-02, echo = FALSE, fig.alt = "A diagram showing the same pair of rectangles as before, connected by the arrow labeled predict. This time, though, we introduce two boxes labeled reference. These two boxes are connected to the arrow labeled predict with dotted arrows, to show that, most of the time, we don't need to think about including them in our workflow.", out.width = '100%'----
knitr::include_graphics("figures/diagram_02.png")

## ----save-model-tf------------------------------------------------------------
temp_dir <- tempdir()
save_model_tf(keras_fit, filepath = temp_dir)

## ----fresh-keras-fit----------------------------------------------------------
r(
  function(temp_dir, new_data) {
    library(keras)
    
    model_object <- load_model_tf(filepath = temp_dir)
    
    predict(model_object, new_data)
  },
  args = list(
    temp_dir = temp_dir,
    new_data = x_test
  )
)

## ----diagram-03, echo = FALSE, fig.alt = "A diagram showing the same set of rectangles, representing a prediction problem, as before. This version of the diagram adds two boxes, labeled R Session numbe r one, and R session number two. In R session number two, we have a new rectangle labeled standalone model object. In focus is the arrow from the model object, in R Session number one, to the standalone model object in R session number two.", out.width = '100%'----
knitr::include_graphics("figures/diagram_03.png")

## ----diagram-04, echo = FALSE, fig.alt = "A replica of the previous diagram, where the arrow previously connecting the model object in R session one and the standalone model object in R session two is connected by a verb called bundle. The bundle function outputs an object called a bundle.", out.width = '100%'----
knitr::include_graphics("figures/diagram_04.png")

## ----keras-bundle-------------------------------------------------------------
keras_bundle <- bundle(keras_fit)

## ----keras-fresh-bundle-------------------------------------------------------
r(
  function(model_bundle, new_data) {
    library(bundle)
    
    model_object <- unbundle(model_bundle)
 
    predict(model_object, new_data)
  },
  args = list(
    model_bundle = keras_bundle,
    new_data = x_test
  )
)

## ----xgboost-fit--------------------------------------------------------------
xgb_fit <- 
  xgboost(
    data = x_train, 
    label = y_train,
    nrounds = 5
  )

## ----xgboost-bundle-----------------------------------------------------------
xgb_bundle <- bundle(xgb_fit)

## ----xgboost-fresh-bundle-----------------------------------------------------
r(
  function(model_bundle, new_data) {
    library(bundle)
    
    model_object <- unbundle(model_bundle)
    
    predict(model_object, new_data)
  },
  args = list(
    model_bundle = xgb_bundle,
    new_data = x_test
  )
)

