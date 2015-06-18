#### Neural Networks ####
#### June 11, 2015 ####
####Implemented everything from scratch using mathematical knowledge of Supervised Learning####
#Supports single layer binary classification and single layer linear regressions.

sigmoid <- function(z){
  r <- 1/(1 + exp(-z))
  return (r)
}

hLinear <- function(x, params){
  return (x %*% params)
}

hLogistic <- function(x, params){
  h <- hLinear(x,params)
  return (sigmoid(h))
}

computeCostLinear<- function(X,Y,params){
  h <- hLinear(X,params)
  val <- sum((h - Y)**2)/(2*nrow(X))
  return (val)
}

computeCostLogistic <- function(X,Y,params){
  h <- sigmoid(hLinear(X,params))
  val <- sum(-Y*log(h) - (1-Y)*log(1-h))/nrow(X)
  return (val)
}

computeCost <- function(X,Y,params,costFun){
  if (identical(costFun,hLinear))
    computeCostLinear(X,Y,params)
  else if (identical(costFun,hLogistic))
    computeCostLogistic(X,Y,params)
}

batchGradientDescent <- function(X,Y,a,costFun,num_iter){
  ####Make sure the alpha parameter is low enough. IF it's too high then
  ####the SGD will not converge resulting in a NaN.
  num_examps <- nrow(X) #Number of training examples
  X <- cbind(1,X) #Prepend column of 1s to the training examples
  
  num_params <- ncol(X) #Number of features per training example
  parameters <- matrix(rep(0,num_params))
  for (iters in 1:num_iter) { #While no convergence
      temp = rep((Y - costFun(X,parameters)),num_params) * X
      temp = colSums(temp)
      parameters <- parameters + a*(1/num_examps)*temp
  }
  return (parameters)
}

stochasticGradientDescent <- function(X,Y,a, costFun,num_iter){
  ####Make sure the alpha parameter is low enough. IF it's too high then
  ####the SGD will not converge resulting in a NaN.
  
  num_examps <- nrow(X) #Number of training examples
  X <- cbind(1,X) #Prepend column of 1s to the training examples
  num_params <- ncol(X) #Number of features per training example
  parameters <- matrix(rep(0,num_params))
  
  prev_param <- matrix(rep(0,num_params))
  prev_cost <- computeCost(X,Y,parameters,costFun)
  
  t=num_iter/10
  T=num_iter
  c=.05
  for (iters in 1:num_iter) { #While no convergence
    for (i in 1:num_examps) { #Iterate through each example
      #In order to ensure that the parameters converge to a global minimum (rather than oscillate)
      #we make sure to let the learning rate decrease to zero as the algorithm runs
      #Using the (Bold Driver Algorithm) I noticed that there were still convergence
      #problems with SGD... Going to use the "search-then-converge schedule" i.e. Annealing
      #http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.2884&rep=rep1&type=pdf
      T = num_iter*100
      t = iters
      r = a * (1 + (c/a)*(t/T)) /(1 + (c/a)*(t/T) + T*(t^2)/(T^2))
      parameters <- parameters + rep(Y[i] - costFun(X[i,],parameters),num_params) * X[i,]*(r)

      #linearRegressionSGD(X,y,.005,1000)
    }
  }
  return (parameters)
}

initialize <- function(){
  data()
  library(ggplot2) 
}

normalEquation<- function(X,y){
  #Computing linear regression via the Normal Equation is slow for large number of features
  #since we must compute the inverse of X^T*X.
  examps <- dim(X)[1]
  X <- cbind(matrix(t(rep(1,examps))),X) #Prepend column of 1s to the training examples
  theta <- solve((t(X) %*% X)) %*% (t(X) %*% y)
  return (theta)
}

linearRegressionBGD <- function(X,y,a,num_iter){
  return (batchGradientDescent(X,y,a,hLinear,num_iter))
}

linearRegressionSGD <- function(X,y,a,num_iter){
  return (stochasticGradientDescent(X,y,a,hLinear,num_iter))
}


logisticRegressionBGD <- function(X,y,a,num_iter){
  return (batchGradientDescent(X,y,a,hLogistic,num_iter))
}

logisticRegressionSGD <- function(X,y,a,num_iter){
  return (stochasticGradientDescent(X,y,a,hLogistic,num_iter))
}

demoExample <- function(){
  A <- data.frame(mtcars)
  X <- matrix(A[["mpg"]]) #Miles per Gallon
  #X <- matrix(A[["mpg"]], A[["hp"]]),32,2) #Miles per Gallon
  y <- matrix(A[["hp"]]) #Horse PowerX
  theta <- normalEquation(X,y)
  myplot <- qplot(A[["mpg"]], A[["hp"]], ylab = "Horse Power", xlab = "Miles per gallon")
  myline <- geom_abline(intercept= theta[1], slope=theta[2], colour="red")
  myplot + myline
  #plot(data$score.1,data$score.2,col=as.factor(data$label),xlab="Score-1",ylab="Score-2")

  
  #x = seq(30,100, by = 0.01)
  #y = seq(30,100, by = 0.01)
  #z = matrix(a[1]+a[2]*x+a[3]*y)
  #z = outer(x*a[2], y*a[3]) + a[1]
  #image(x,y,z,col=heat.colors(2))
}
