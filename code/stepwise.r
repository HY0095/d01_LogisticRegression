pi_fun<-function(x,Beta){                    
  Beta<-matrix(as.vector(Beta),ncol=1)   
  x<-matrix(as.vector(x),ncol=nrow(Beta))
  g_fun<-x%*%Beta                        
  exp(g_fun)/(1+exp(g_fun))              
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

Newtons<-function(fun,x,y,ep=1e-8,it_max=100){                                                                                                                                            
  x<-matrix(as.vector(x),nrow=nrow(y))                                                                                      
  Beta=matrix(rep(0,ncol(x)),nrow=ncol(x))                                                                                  
  Index<-0;                                                                                                                 
  k<-1                                                                                                                      
  while(k<=it_max){                                                                                  
    x1<-Beta;obj<-fun(x,y,Beta);                                                                                            
    
    Beta<-Beta+solve(obj$H,obj$U);                                                                                          
    
    objTemp<-fun(x,y,Beta)                                                                                                  
    if(any(is.nan(objTemp$H))){                                                                      
      Beta<-x1                                                                                                              
      print("Warning:The maximum likelihood estimate does not exist.")                                                      
      print(paste("The LOGISTIC procedure continues in spite of the above warning.",                                        
                  "Results shown are based on the last maximum likelihood iteration. Validity of the model fit is questionable."));
      break                                                                                                                 
    }                                                                                                                       
    norm<-sqrt(t((Beta-x1))%*%(Beta-x1))                                                                                    
    if(norm<ep){                                                                                     
      index<-1;break                                                                                                        
    }                                                                                                                       
    k<-k+1                                                                                                                  
  }                                                                                                                         
  obj<-funs(x,y,Beta);                                                                                                        
  
  list(Beta=Beta,it=k,U=obj$U,H=obj$H)                                                                                       
}  

CheckZero<-function(x){                                                                                                                            
  for(i in 1:length(x)){
    if(x[i]==0)                                
      x[i]<-1e-300                           
  }                                          
  x                                            
}  

ModelFitStat<-function(x,y,Beta){                                                                     
  x<-matrix(as.vector(x),nrow=nrow(y))                                                                                                         
  Beta<-matrix(as.vector(Beta),ncol=1)                                                             
  
  uni_matrix<-matrix(rep(1,nrow(y)),nrow= nrow(y));                                                
  LOGL<--2*(t(y)%*%log(pi_fun(x,Beta))+t(uni_matrix-y)%*%log(CheckZero(uni_matrix-pi_fun(x,Beta))))
  
  #print("-----------------")                                                                      
  #print(LOGL)                                                                                     
  
  AIC<-LOGL+2*(ncol(x))                                                                            
  SC<-LOGL+(ncol(x))*log(nrow(x))                                                                  
  list(LOGL=LOGL,AIC=AIC,SC=SC)                                                                    
  
}


GlobalNullTest<-function(x,y,Beta,BetaIntercept){                                                                                                                                              
  y<-matrix(as.vector(y),ncol=1)                                                          
  x<-matrix(as.vector(x),nrow=nrow(y))                                                    
  Beta<-matrix(as.vector(Beta),ncol=1)                                                    
  
  pi_value<- pi_fun(x,Beta)                                                               
  df<-nrow(Beta)-1                                                                        
  ##compute Likelihood ratio                                                              
  
  MF<-ModelFitStat(x,y,Beta)                                                              
  n1<-sum(y[y>0])                                                                         
  n<-nrow(y)                                                                              
  LR<--2*(n1*log(n1)+(n-n1)*log(n-n1)-n*log(n))-MF$LOGL                                   
  LR_p_value<-pchisq(LR,df,lower.tail=FALSE)                                              
  LR_Test<-list(LR=LR,DF=df,LR_p_value=LR_p_value)                                        
  
  ##compute Score                                                                         
  
  BetaIntercept<-matrix(c(as.vector(BetaIntercept),rep(0,ncol(x)-1)),ncol=1)              
  obj<-funs(x,y,BetaIntercept)                                                            
  Score<-t(obj$U)%*%solve(obj$H)%*%obj$U                                                  
  Score_p_value<-pchisq(Score,df,lower.tail=FALSE)                                        
  Score_Test<-list(Score=Score,DF=df,Score_p_value=Score_p_value)                         
  
  ##compute Wald test                                                                     
  obj<-funs(x,y,Beta)                                                                     
  I_Diag<-diag((solve(obj$H)))                                                            
  
  Q<-matrix(c(rep(0,nrow(Beta)-1),as.vector(diag(rep(1,nrow(Beta)-1)))),nrow=nrow(Beta)-1)
  Wald<-t(Q%*%Beta)%*%solve(Q%*%diag(I_Diag)%*%t(Q))%*%(Q%*%Beta)                         
  
  Wald_p_value<-pchisq(Wald,df,lower.tail=FALSE)                                          
  Wald_Test<-list(Wald=Wald,DF=df,Wald_p_value=Wald_p_value)                              
  
  list(LR_Test=LR_Test,Score_Test=Score_Test,Wald_Test=Wald_Test)                         
}       


WhichEqual1<-function(x){                                                                                                           
  a<-NULL                                      
  for(i in 1:length(x)){
    if(x[i]==1){        
      a<-c(a,i)                                
    }                                          
  }                                           
  a                                            
}  


CheckOut<-function(source,check){                                                                   
  for(j in 1:length(source)){ 
    for(k in 1:length(check)){
      if(source[j]==check[k])                        
        source[j]<-0                                 
    }                                                
  }                                                  
  source[source>0]                                   
}       

NegativeCheck<-function(x){                                                                    
  for(i in length(x):1){
    if(x[i]>0)                                 
      break                                    
  }                                          
  i                                            
}     


CycleCheck<-function(x){                                                                       
  NegativeFlg<-NegativeCheck(x)                     
  if(NegativeFlg==length(x)){
    return(FALSE)                                   
  }                                                 
  NegVec<-x[(NegativeFlg+1):length(x)]              
  PosVec<-x[(2*NegativeFlg-length(x)+1):NegativeFlg]
  
  NegVec<-sort(-1*NegVec)                           
  PosVec<-sort(PosVec)                              
  
  if(all((NegVec-PosVec)==0))                       
    return(TRUE)                                   
  
  return(FALSE)                                     
}   

##stepwise                                                                                                                                                                    
Stepwise<-function(x,y,checkin_pvalue=0.3,checkout_pvalue=0.35){                                   
  ##as matrix
  x_name = names(x)                                                                                                              
  x<-matrix(as.vector(as.matrix(x)),nrow=nrow(y))
  x = cbind(rep(1, nrow(y)), x)
  x_name = c("Intercept", x_name)                                                                                     
  ##indication of variable                                                                                                 
  indict<-rep(0,ncol(x)) ##which column enter                                                                              
  ##intercept entered                                                                                                      
  indict[1]<-1                                                                                                             
  Beta<-NULL                                                                                                               
  print("Intercept Entered")                                                                                               
  Result<-Newtons(funs,x[,1],y)                                                                                            
  Beta<-Result$Beta                                                                                                        
  BetaIntercept<-Result$Beta                                                                                               
  uni_matrix<-matrix(rep(1,nrow(y)),nrow= nrow(y));                                                                        
  LOGL<--2*(t(y)%*%log(pi_fun(x[,1],Beta))+t(uni_matrix-y)%*%log(uni_matrix-pi_fun(x[,1],Beta)))                           
  print(paste("-2Log=",LOGL))                                                                                              
  indexVector<-WhichEqual1(indict)                                                                                         
  ##check other variable                                                                                                   
  
  VariableFlg<-NULL                                                                                                        
  Terminate<-FALSE                                                                                                         
  repeat{                                                                                           
    if(Terminate==TRUE){                                                                            
      print("Model building terminates because the last effect entered is removed by the Wald statistic criterion. ")      
      break                                                                                                                
    }                                                                                                                      
    
    pvalue<-rep(1,ncol(x))                                                                                                 
    
    k<-2:length(indict)                                                                                                    
    k<-CheckOut(k,indexVector)                                                                                             
    for(i in k){                                                                                    
      
      obj<-funs(c(x[,indexVector],x[,i]),y,c(as.vector(Beta),0))                                                           
      Score<-t(obj$U)%*%solve(obj$H,obj$U)                                                                                 
      Score_pvalue<-pchisq(Score,1,lower.tail=FALSE)                                                                       
      pvalue[i]<-Score_pvalue                                                                                              
    }                                                                                                                      
    #print("Score pvalue for variable enter")                                                                              
    #print(pvalue)                                                                                                         
    pvalue_min<-min(pvalue)                                                                                                
    if(pvalue_min<checkin_pvalue){                                                                  
      j<-which.min(pvalue)                                                                                                 
      
      print(paste(x_name[j]," entered:"))                                                                                  
      ##set indication of variable                                                                                         
      indict[j]<-1                                                                                                         
      VariableFlg<-c(VariableFlg,j)                                                                                        
      
      indexVector<-WhichEqual1(indict)                                                                                     
      print("indexVector--test")                                                                                           
      print(indexVector)                                                                                                   
      
      Result<-Newtons(funs,x[,indexVector],y)                                                                              
      Beta<-NULL                                                                                                           
      Beta<-Result$Beta                                                                                                    
      
      ##compute model fit statistics                                                                                       
      print("Model Fit Statistics")                                                                                        
      MFStat<-ModelFitStat(x[,indexVector],y,Beta)                                                                         
      print(MFStat)                                                                                                        
      ##test globel null hypothesis:Beta=0                                                                                 
      print("Testing Global Null Hypothese:Beta=0")                                                                        
      GNTest<-GlobalNullTest(x[,indexVector],y,Beta,BetaIntercept)                                                         
      print(GNTest)                                                                                                        
      
      repeat{                                                                                       
        ##compute Wald test in order to remove variable                                                                    
        indexVector<-WhichEqual1(indict)                                                                                   
        obj<-funs(x[,indexVector],y,Beta)                                                                                  
        H_Diag<-sqrt(diag(solve(obj$H)))                                                                                   
        WaldChisq<-(as.vector(Beta)/H_Diag)^2                                                                              
        WaldChisqPvalue<-pchisq(WaldChisq,1,lower.tail=FALSE)                                                              
        WaldChisqTest<-list(WaldChisq=WaldChisq,WaldChisqPvalue=WaldChisqPvalue)                                           
        print("Wald chisq pvalue for variable remove")                                                                     
        print(WaldChisqPvalue)                                                                                             
        
        ##check wald to decide to which variable to be removed                                                             
        pvalue_max<-max(WaldChisqPvalue[2:length(WaldChisqPvalue)])##not check for intercept                               
        
        if(pvalue_max>checkout_pvalue){                                                             
          n<-which.max(WaldChisqPvalue[2:length(WaldChisqPvalue)])##not check for intercept                                
          print(paste(x_name[indexVector[n+1]]," is removed:"))                                                            
          ##set indication of variable                                                                                     
          
          indict[indexVector[n+1]]<-0                                                                                      
          m<- indexVector[n+1]                                                                                             
          VariableFlg<-c(VariableFlg,-m)                                                                                   
          
          ##renew Beta                                                                                                     
          indexVector<-WhichEqual1(indict)                                                                                 
          Result<-Newtons(funs,x[,indexVector],y)                                                                          
          Beta<-NULL                                                                                                       
          Beta<-Result$Beta                                                                                                
          
          if(CycleCheck(VariableFlg)==TRUE){                                                        
            Terminate<-TRUE                                                                                                
            break                                                                                                          
          }                                                                                                                
          
        }                                                                                                                  
        else {                                                                                      
          print(paste("No (additional) effects met the" ,checkout_pvalue,"significance level for removal from the model."))
          break;                                                                                                           
        }                                                                                                                  
      }##repeat end                                                                                                        
    }                                                                                                                      
    else {                                                                                          
      print(paste("No effects met the" ,checkin_pvalue," significance level for entry into the model"))                    
      break                                                                                                                
    }                                                                                                                      
  }##repeat end                                                                                                            
  
  ##                                                                                                                       
  print("Beta")                                                                                                            
  print(Beta)                                                                                                              
  print("Analysis of Maximum Likelihood Estimates")                                                                        
  obj<-funs(x[,indexVector],y,Beta)                                                                                        
  H_Diag<-sqrt(diag(solve(obj$H)))                                                                                         
  WaldChisq<-(as.vector(Beta)/H_Diag)^2                                                                                    
  WaldChisqPvalue<-pchisq(WaldChisq,1,lower.tail=FALSE)                                                                    
  WaldChisqTest<-list(WaldChisq=WaldChisq,WaldChisqPvalue=WaldChisqPvalue)                                                 
  print(WaldChisqTest)                                                                                                     
} 

