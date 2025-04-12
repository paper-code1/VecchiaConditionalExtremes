
#########  comparison of GP methods on satellite drag data

##  data and laGP results are from
## https://bitbucket.org/gramacylab/tpm/src/master/
# To reproduce the results, save the following files
#    from tpm/data/HST to a local subfolder 'HSTdata/':
#   - hstXX.dat for XX in (O,O2,N,N2,He,H)
#     (combine hstHe and hstHe2 into a single file hstHe)
######## This code is from 
## Katzfuss, Matthias, Joseph Guinness, and Earl Lawrence. 
# "Scaled Vecchia approximation for fast computer-model 
#   emulation." SIAM/ASA Journal on Uncertainty 
#   Quantification 10.2 (2022): 537-554.


###  load functions and packages

source('./vecchia_scaled.R')
library(tictoc)


##########    cross-validation experiment   ########

# species=c('He')
species=c('O','O2','N','N2','H','He') #
methods=c('SVecchia','Vecchia')
m.est=30
m.pred=140
n.est=10000  # size of random subset for estimation
n.all=2e6 # 1e6 for He
d=8

### split into 10 subsets for cross-validation
set.seed(999)
folds=10
cv.inds=matrix(sample(1:n.all,n.all),nrow=folds)
n.test=n.all/folds
n.train=n.all-n.test

### initialize output
time=mse=array(dim=c(length(methods),length(species),2,folds))
par.ests=array(dim=c(length(methods),length(species),d+1,folds))

### loop over the 6 chemical species
for(i.s in 1:length(species)){

  print(species[i.s])
  
  ### loop over CV folds
  for(fold in 1:folds){
    
    print(paste0('fold=',fold))
    
    ## Load training and validation data
    train_path <- paste0('./HST/', species[i.s], '/hst', species[i.s], '_fold', fold, '_train.csv')
    val_path <- paste0('./HST/', species[i.s], '/hst', species[i.s], '_fold', fold, '_val.csv')
    
    train_data <- read.csv(train_path, header=FALSE)
    val_data <- read.csv(val_path, header=FALSE)
    
    ## Extract inputs and responses
    y.train <- as.numeric(train_data[,9])
    inputs.train <- as.matrix(train_data[,1:8])
    
    y.test <- as.numeric(val_data[,9])
    inputs.test <- as.matrix(val_data[,1:8])

    ## loop over GP methods
    for(meth in 1:length(methods)){
    
      print(methods[meth])
      
      ## how to scale inputs depends on method
      if(methods[meth]=='SVecchia'){ scale='parms'
      } else if(methods[meth]=='Vecchia'){ scale='ranges'
      }
      
      ## estimate parameters on subset
      tic()
      fit=fit_scaled(y.train,inputs.train,ms=c(m.est),n.est=n.est,scale=scale,print.level = 0)
      temp=toc()
      time[meth,i.s,1,fold]=temp$toc-temp$tic
      par.ests[meth,i.s,,fold]=fit$covparms[1:(d+1)]
    
      ## make prediction
      tic()
      preds=predictions_scaled(fit=fit,locs_pred=inputs.test,m=m.pred,scale=scale)
      temp=toc()
      time[meth,i.s,2,fold]=temp$toc-temp$tic
      
      ## compute rmse
      mse[meth,i.s,1,fold]=mean((preds-y.test)^2)
      mse[meth,i.s,2,fold]=mean((100*(preds - y.test)/y.test)^2)
      
      ## print new results and save
      print(sqrt(mse[meth,i.s,,fold])); print(time[meth,i.s,,fold])
      save(mse,time,par.ests,file=paste0('results/satelliteCV.RData'))
      
    }
  }
  print(paste0("=============MSPE of ",species[i.s],"============="))
  print(paste0("SVecchia: ",mean(mse[1,i.s,2,])))
  print(paste0("CVecchia: ",mean(mse[2,i.s,2,])))
  print("=============================")
}