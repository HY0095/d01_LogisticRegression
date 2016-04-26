#*******
#  Credit Risk Scorecards: Development and Implementation using R
#  (c) Duan
#********
#*******************************************************

#-- 01: Data Partition --------------------------
partition <- function(dsin,ratio){
  for(i in 1:nrow(dsin)) {
    if(runif(1) <= ratio) {
      dsin$role[i] = "train"
    } else {
      dsin$role[i] = "test"
    }
  }
  return(dsin)
}

funs<-function(x,y,Beta){
  y<-matrix(as.vector(y),ncol=1)
  x<-matrix(as.vector(x),nrow=nrow(y))
  Beta<-matrix(c(Beta),ncol=1)
  
  pi_value<- pi_fun(x,Beta)
  U<-t(x)%*%(y-pi_value);
  uni_matrix<-matrix(rep(1,nrow(pi_value)),nrow= nrow(pi_value));
  H<-t(x)%*%diag(as.vector(pi_value*( uni_matrix -pi_value)))%*%x
  list(U=U,H=H)                                                  
} 

NewtonRaphson <- function(x,y,weight, max.diff = 1e-8, max.iter = 25 ) {
  iter   = 0
  old_b  = 0
  x      = as.matrix(x)
  nrows  = nrow(x)
  ncols  = ncol(x)
  int_pi = 0.5*as.matrix(rep(1, nrows))
  # x      = cbind(rep(1, nrows), x)
  p       = dim(x)
  repeat {
    W      = diag(as.vector(weight))%*%diag(as.vector(int_pi*(1 - int_pi)))
    covb   = solve(t(x)%*%W%*%x)
    new_b  = old_b + covb%*%t(x)%*%diag(as.vector(weight))%*%(y - int_pi)
    int_pi = 1/(1 + exp( -x%*%new_b))
    iter   = iter +1       
    if( t(new_b - old_b)%*%(new_b - old_b) < max.diff | iter > max.iter) {
      break
    }
    old_b = new_b
  }   
  if (iter > max.iter) {
    cat("Newton-Raphson Algorithm failed to converge !\n")
    return(null)
  } else {
    cat("Newton-Raphson Algorithm Converged !\n")
    std_error = as.matrix(sqrt(diag(covb)))
    corrb     = covb/as.vector(std_error%*%t(std_error))
    wald_chi  = (new_b/std_error)^2
    prob      = 2 * pnorm(-abs(new_b/std_error))
    dfe       = (nrows - 1) - ncols 
    y_fit     = exp(x%*%new_b)/(1 + exp(x%*%new_b))
    LogL      = 2*sum(weight*y*log(y_fit)+ weight*(1-y)*log(1 - y_fit))
    AIC       = -LogL + 2*p[2]
    SC        = -LogL + p[2]*(log(p[1]))       
    return(list(beta = new_b, y_fit = y_fit, dfe = dfe, covb = covb, std_error = std_error, corrb = corrb, wald_chi = wald_chi, prob = prob , AIC = AIC, SC = SC, LogL = -LogL))   
  }      
}


