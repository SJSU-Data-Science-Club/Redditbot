#### Basic Network ####
library(keras)
reset_states(model)
model <- keras_model_sequential()

model %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(50), trainable = F) %>%
  layer_dense(units= 20, activation = 'relu') %>%
  layer_dense(units = 12, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')  
)

summary(model)
history <- model %>% fit(
  as.matrix(train_glove), y_train_glove,
  batch_size = 10,
  epochs = 100,
  validation_data = list(as.matrix(test_glove), y_test_glove),
  verbose=1
)
