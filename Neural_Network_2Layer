#### Neural Networks ####
#### June 14, 2015 ####
#### Builds off of NeuralNetwork.R code####

RadialBF <- function(z){
  #The Gaussian Radial Basis Function. Activates regions near the center.
  i <- exp(-z^2)
  return (i)
}

hidden_activation <- function(x,wh){
  #This computes the output of the hidden layer.
  
  #Returns h = e^(-([x]*[wh])^2)
  return (RadialBF(hLinear(x,wh)))
}

output_activation <- function(h,wo){
  #This computes the output of the final layer.
  
  #Returns y = 1/(1 + e^(-h*wo + 1))
  return (sigmoid(h*wo - 1))
}

nn <- function(x, wh, wo){
  #Neural Network Function. Returns Y
  return (output_activation(hidden_activation(x,wh),wo))
}

nn_pred <- function(x, wh, wo) {
  #Returns the prediction of whether in class 1 or 0
  return (round(nn(x,wh,wo)))
}

cost <- function(y,t){
  #The total error in the final layer
  i <- -sum(t*log(y) + (1-t)*log(1-y))
  return (i)
}

cost_for_param <- function(x,wh,wo,t){
  #The total error given specific parameters.
  cost(nn(x,wh,wo),t)
}

gradient_output <- function(y,t){
  #Needed for computing the gradient weight out
  return (y - t)
}

gradient_weight_out <- function(h,grad_output){
  #Computes d/wo, the gradient of the cost with respect to the 
  #weight of the final layer. This is a batch computation, 
  #so sum the results for batch gradient descent.
  return (h*grad_output)
}

gradient_hidden <- function(wo, grad_output){
  #Computes d/wh, the gradient of the cost with respect to the
  #weight of the hidden layer
  return (wo * grad_output)
}

gradient_weight_hidden <- function(x,zh,h,grad_hidden){
  #Computes d/wh, the gradient of the cost with respect to the
  #weight of the hidden layer. This is a batch computation, 
  #so sum the results for batch gradient descent.
  return (x*-2*zh*h*grad_hidden)
}

backprop_update <- function(x, t, wh, wo, learning_rate){
  #Updates network parameters for a single iteration
  zh <- x * wh
  h <- RadialBF(zh)
  y <- output_activation(h,wo)
  grad_out <- gradient_output(y,t)
  d_wo <- learning_rate * gradient_weight_out(h,grad_out)
  grad_hid <- gradient_hidden(wo,grad_out)
  d_wh <- learning_rate * gradient_weight_hidden(x,zh,h,grad_hid)
  
  return(c(wh - sum(d_wh), wo - sum(d_wo)))
}
#source("NeuralNetwork_HiddenLayer.R")
demoNNH <- function(){
  #Runs a demo example on a bimodial distribution.
  
  set.seed(200)
  a <- c(rnorm(100, -2, .5)[1:20],rnorm(100, 2, .5)[1:20]) #Output = 0
  b <- rnorm(100,0,.5)[1:40] #Output = 1 *since in center
  x <- matrix(c(a,b))
  t <- matrix(c(rep(0,40),rep(1,40)))
  qplot(x,fill=as.factor(t))
  #image(x,col=as.factor(t))
  
  wh = 5
  wo = -2
  
  learning_rate = .002
  nb_of_iterations = 10000
  lr_update = learning_rate / nb_of_iterations
  cur_cost <- cost_for_param(x,wh,wo,t)
  print(c("Current cost:",as.character(cur_cost)))
  for (i in 1:nb_of_iterations){
    learning_rate <- learning_rate - lr_update
    cur_cost <- cost_for_param(x,wh,wo,t)
    result <- backprop_update(x, t, wh, wo, learning_rate)
    wh_new <- result[1]
    wo_new <- result[2]
    new_cost <- cost_for_param(x,wh_new,wo_new,t)
    wh <- wh_new
    wo <- wo_new
  }
  print(c("Final cost:",as.character(new_cost)))
  
  #Plots on graph to compute difference between our predictions and actual results
  plot(t-nn_pred(x, wh, wo))
  return (c(wh,wo))
}
