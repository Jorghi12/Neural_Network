#### Neural Networks ####
#### June 18, 2015 ####
#### This Neural Network takes a 2D input space, projects them
#### onto a 3D hidden layer, and classifies them with a 2D
#### softmax output classifier. Bias parameters also included

drawPlot <- function(){
  num_elements = 100
  std_dev = .4
  blues <- cbind(rnorm(num_elements,0,std_dev),rnorm(num_elements,0,std_dev))

  redsRadius <- rnorm(num_elements,2,std_dev)
  redsThetas <- rnorm(num_elements,0,2*pi)
  reds <- cbind(redsRadius*cos(redsThetas),redsRadius*sin(redsThetas))

  #blackRadius <- rnorm(num_elements,4,std_dev)
  #blackThetas <- rnorm(num_elements,0,2*pi)
  #blacks <- cbind(blackRadius*cos(blackThetas),blackRadius*sin(blackThetas))
  
  all_points <- rbind(blues,reds)
  all_colors <- c(rep("blue",100),rep("red",100))
  #all_points <- rbind(blues,reds,blacks)
  #all_colors <- c(rep("blue",100),rep("red",100), rep("black",100))
  
  qplot(all_points[,1],all_points[,2], xlim = c(-3,3), ylim = c(-3,3), colour= I(all_colors))

  
  #blueT <- cbind(matrix( t(rep(1,100))),matrix( t(rep(0,100))))
  #redT <- cbind(matrix( t(rep(0,100))),matrix( t(rep(1,100))))
  
  #targets = rbind(blueT,redT)
  X <- rbind(blues,reds)
  T <- rbind(blueT,redT)
  run_backpropogation(X,T)
  
  
}

#Sigmoid Function
sigmoid <- function(z){
  r <- 1/(1 + exp(-z))
  return (r)
}

#Softmax Function
softmax <- function(Zo){
  i <- exp(Zo) / rowSums(exp(Zo))
  return (i)
}

#Computes hidden layer given training data and hidden weights
hidden_activations <- function(X,Wh,bh){
  #Returns H
  H <- X %*% Wh
  H <- t(t(H) + c(bh))
  H <- sigmoid(H)
  return (H)
}

#Computes outer layer given hidden layer and outer weights
output_activations <- function(H,Wo,bo){
  #Returns Y
  Y <- H %*% Wo
  Y <- t(t(Y) + c(bo))
  Y <- softmax(Y)
  return (Y)
}

#Computers outer layer given hidden layer and inner+outer weights
nn <- function(X,Wh,Wo,bh,bo){
  return (output_activations(hidden_activations(X, Wh, bh), Wo, bo))
}

#Computers outer layer given hidden layer and inner+outer weights
nn <- function(X,params){
  return (output_activations(hidden_activations(X, params[[1]], params[[2]]), params[[3]], params[[4]]))
}

#Computes outer layer predictions
nn_predict <- function(X,Wh,Wo,bh,bo){
  return (round(nn(X,Wh,Wo,bh,bo)))
}

#Computes outer layer predictions
nn_predict <- function(X,params){
  return (round(output_activations(hidden_activations(X, params[[1]], params[[2]]), params[[3]], params[[4]])))
}

#The gradient of the cost function with respect to Zo (the input to the outer layer)
error_output <- function(Y,T){
  Eo <- (Y - T)
  return (Eo)
}

#Computes the gradient of the cost function with respect to the weights out
gradient_weight_out <- function(H, Eo){
  #Batch vectorized
  Jwo <- t(H) %*% Eo
  return (Jwo)
}

#Computes the gradient of the cost function with respect to the bias weights out
gradient_bias_out <- function(Eo){
  #Batch (vectorized requires Jacobian)
  Jbo <- matrix(colSums(Eo),1,2)
  return (Jbo)
}

#Computes the cost
cost <- function(Y,T){
  c <- -sum(T * log(Y))
  return (c)
}

error_hidden <- function(H,Wo,Eo){
  Eh <- H * (1 - H) * (Eo %*% t(Wo))
  return (Eh)
}

gradient_weight_hidden <- function(X, Eh){
  i <- t(X) %*% Eh
  return (i)
}

gradient_bias_hidden <- function(Eh){
  i <- matrix(colSums(Eh),1,3)
  return (i)
}

check_gradient <- function(X,T){
  #Make sure T matrix is initialized correctly or errors will occur!
  bh <- matrix(runif(3, 0.0, 1.0),1,3)
  Wh <- matrix(runif(6, 0.0, 1.0),2,3)
  bo <- matrix(runif(2, 0.0, 1.0),1,2) 
  Wo <- matrix(runif(6, 0.0, 1.0),3,2) #error starts
  #check_gradient(X,T)
  
  H <- hidden_activations(X,Wh,bh)
  Y <- output_activations(H,Wo,bo)
  
  Eo <- error_output(Y,T)
  JWo <- gradient_weight_out(H, Eo)
  Jbo <- gradient_bias_out(Eo)
  
  Eh <- error_hidden(H, Wo, Eo)
  JWh <- gradient_weight_hidden(X, Eh)
  Jbh <- gradient_bias_hidden(Eh)
  
  params <- list(Wh, bh, Wo, bo)
  grad_params <- list(JWh, Jbh, JWo, Jbo)

  eps <- 0.0001
  for (p_idx in 1:length(params)){
    for (row in 1:nrow(params[[p_idx]])){
      for (col in 1:ncol(params[[p_idx]])){
        p_matrix_min <- params[[p_idx]]
        p_matrix_min[row,col] <- p_matrix_min[row,col] - eps
        p_matrix_plus <- params[[p_idx]]
        p_matrix_plus[row,col] <- p_matrix_plus[row,col] + eps
        
        params_min <- params
        params_min[[p_idx]] <- p_matrix_min
        
        params_plus <- params
        params_plus[[p_idx]] <- p_matrix_plus
        
        grad_num <- (cost(nn(X, params_plus), T)-cost(nn(X, params_min), T))/(2*eps)
        
        print(c("backprop grad:",grad_params[[p_idx]][row,col],"numerical grad",grad_num))
    }
  }
  }
}

####################################################################################################
#Multi-layer neural network with a non linear activation function will likely 
#have a NON convex cost function.

#Instead of gradient descent, using a momentum based algorithm. This helps us get
#out of local minimums towards the actual global minimum. Imagine a hill with a lot
#of curves. Gradient descent would get stuck at a local minimum on the hill, but a 
#momentum based approach can be thought of as a ball rolling down said hill.
#The velocity increases as the ball goes down and decreases as it goes up hill,
#thus it's natural for the ball to "break" out of local minimums and reach the absolute
#minimum of the hill

backpop_gradient <- function(X,T,Wh,bh,Wo,bo){
  H <- hidden_activations(X,Wh,bh)
  Y <- output_activations(H,Wo,bo)
  
  Eo <- error_output(Y,T)
  JWo <- gradient_weight_out(H, Eo)
  Jbo <- gradient_bias_out(Eo)
  
  Eh <- error_hidden(H, Wo, Eo)
  JWh <- gradient_weight_hidden(X, Eh)
  Jbh <- gradient_bias_hidden(Eh)
  return (list(JWh,Jbh,JWo,Jbo))
}

update_velocity <- function(X, T, ls_of_params, Vs, momentum_term, learning_rate){
  Js <- backpop_gradient(X, T, ls_of_params[[1]],ls_of_params[[2]],ls_of_params[[3]],ls_of_params[[4]])
  
  for (i in 1:length(Vs)){
    Vs[[i]] <- momentum_term * Vs[[i]] - learning_rate * Js[[i]]
  }
  
  return (Vs)
}

update_params <- function(ls_of_params, Vs){
  newl <- list()
  for (i in 1:length(Vs)){
    newl[[i]] <- ls_of_params[[i]] + Vs[[i]]
  }
  return (newl)
}

run_backpropogation <- function(X,T){
  bh <- matrix(runif(3, 0.0, 1.0),1,3)
  Wh <- matrix(runif(6, 0.0, 1.0),2,3)
  bo <- matrix(runif(2, 0.0, 1.0),1,2) 
  Wo <- matrix(runif(6, 0.0, 1.0),3,2)
  
  learning_rate <- 0.02
  momentum_term <- 0.9
  
  weights <- list(Wh, bh, Wo, bo)
  Vs <- list()
  
  for (i in 1:length(weights)){
    Vs[[i]] <- matrix(0,dim(weights[[i]])[1],dim(weights[[i]])[2])
  }
  
  iterations <- 200
  lr_update <- learning_rate / iterations
  ls_costs <- list()
  
  for (i in 1:iterations){
    current_cost <- cost(nn(X, weights), T)
    ls_costs[[i]] <- current_cost
    Vs <- update_velocity(X, T, weights, Vs, momentum_term, learning_rate)
    weights = update_params(weights, Vs)
  }
  #print(ls_costs)
  C <- list()
  v <- 1
  xA <- list()
  yA <- list()
  colors <- c("#FF0000","#0276FD")
  for (x in seq(from = -3, to = 3, by = .05)){
    for (y in seq(from = -3, to = 3, by = .05)){
      xA[[v]] <- x
      yA[[v]] <- y
      C[[v]] <- colors[nn_predict(c(x,y), weights)+1][1]
      v <- v + 1
    }
  }
  qplot(unlist(xA),unlist(yA),color = unlist(C))
}
