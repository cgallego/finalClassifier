####################################
## Train Cascade classifier with all datasets for further out-of-bag classifications
####################################

finalClassifier_trainCascade <- function(features, verbose=TRUE) {
  # initialize
  library(caret)
  library(randomForest)
  library(MASS)
  library(mlbench)
  library(pROC)
  
  # create data.frame to hold resamples of cascade
  cascadeCollected <- as.data.frame(setNames(replicate(10, 1, simplify = F), c("caseID", "labels", "pred1", "obs1", "P1", "N1", "pred2",    "obs2",    "P2", "N2")))
  
  #####################################
  # 1) Set partitionsetDi to 100% datasets
  #####################################
  set <-  na.omit(features[c(10:87)])
  setDi <- set[c(1)]
  outcomesetDi  <- setDi$BenignNMaligNAnt
  partitionsetDi <- createDataPartition(y = outcomesetDi, ## the outcome data are needed
                                        p = 1, ## The percentage of data in the training set
                                        list = FALSE) ## The format of the results. 
  
  nTrainsetDi <-  length( setDi[ partitionsetDi ,] )
  
  ## 2) sampled with replacement from the complete cross validation data set 5000 times
  sTrainsetDi <- sample(nrow(set),size=nTrainsetDi, replace=F)
  
  ################################################
  # 2) Select subsets of features correspondingly
  ################################################
  set1_selfeatures <- set[c("BenignNMaligNAnt",
                            "circularity",
                            "Tpeak.inside",
                            "Slope_ini.inside",
                            "SER.inside",
                            "SER.countor",
                            "Kpeak.inside",
                            "iiMin_change_Variance_uptake",
                            "iMax_Variance_uptake",
                            "Kpeak.countor")]
  # generate case IDS to recollect after
  set1_selfeatures$caseID = rownames(set1_selfeatures)
  set1_selfeatures$labels <- set1_selfeatures$BenignNMaligNAnt
  
  Set1trainingSel.boot <- set1_selfeatures[sTrainsetDi ,]
  M<-subset(Set1trainingSel.boot, BenignNMaligNAnt=="massB" | BenignNMaligNAnt=="massM")
  ifelse( M$BenignNMaligNAnt == "massB", "mass", "mass") -> M$BenignNMaligNAnt
  N<-subset(Set1trainingSel.boot, BenignNMaligNAnt=="nonmassB" | BenignNMaligNAnt=="nonmassM")
  ifelse( N$BenignNMaligNAnt == "nonmassB", "nonmass", "nonmass") -> N$BenignNMaligNAnt
  
  Set1trainingSel.boot = data.frame(rbind(M,N))
    
  ################## 
  ### Select subsets of features correspondingly
  set2_selfeatures <- set[c("BenignNMaligNAnt","Kpeak.countor",
                            "beta.countor",
                            "irregularity",
                            "washoutRate.contour",
                            "UptakeRate.contour",
                            "SER.countor",
                            "Tpeak.countor",
                            "A.countor",
                            "min_F_r_i",
                            "maxCr.contour",
                            "circularity",
                            "beta.inside",
                            "UptakeRate.inside",
                            "texture_energy_threeQuaRad",
                            "iAUC1.countor",
                            "texture_ASM_threeQuaRad",
                            "Kpeak.inside",
                            "alpha.inside",
                            "Vr_increasingRate.inside",
                            "maxVr.inside",
                            "texture_ASM_quarterRad",
                            "texture_energy_quarterRad",
                            "maxCr.inside",
                            "max_F_r_i",
                            "A.inside",
                            "texture_energy_zero",
                            "iAUC1.inside",
                            "texture_contrast_quarterRad",
                            "Tpeak.inside",
                            "peakCr.contour",
                            "texture_contrast_halfRad",
                            "Vr_post_1.inside",
                            "alpha.countor",
                            "texture_homogeneity_quarterRad",
                            "texture_contrast_threeQuaRad",
                            "Slope_ini.countor",
                            "texture_ASM_zero",
                            "texture_energy_halfRad",
                            "texture_ASM_halfRad",
                            "texture_contrast_zero",
                            "Vr_decreasingRate.contour",
                            "SER.inside",
                            "mean_F_r_i",
                            "Vr_increasingRate.contour",
                            "var_F_r_i")]
  
  # generate case IDS to recollect after
  set2_selfeatures$caseID = rownames(set2_selfeatures)
  set2_selfeatures$labels <- set2_selfeatures$BenignNMaligNAnt
  
  Set2trainingSel.boot <- set2_selfeatures[sTrainsetDi ,]
  M<-subset(Set2trainingSel.boot, BenignNMaligNAnt=="massB" | BenignNMaligNAnt=="massM")
  ifelse( M$BenignNMaligNAnt == "massB", "NC", "C") -> M$BenignNMaligNAnt    
  Set2trainingSel.boot = data.frame(M)
  
  ################## 
  ### Select subsets of features correspondingly
  set3_selfeatures <- set[c("BenignNMaligNAnt",
                            "Slope_ini.countor",
                            "edge_sharp_std",
                            "UptakeRate.contour",
                            "var_F_r_i",
                            "maxCr.contour",
                            "Tpeak.countor",
                            "alpha.countor",
                            "max_F_r_i",
                            "iiMin_change_Variance_uptake",
                            "max_RGH_mean",
                            "texture_correlation_quarterRad",
                            "max_RGH_var",
                            "texture_contrast_threeQuaRad",
                            "max_RGH_var_k",
                            "washoutRate.contour")]
  
  # generate case IDS to recollect after
  set3_selfeatures$caseID = rownames(set3_selfeatures)
  set3_selfeatures$labels <- set3_selfeatures$BenignNMaligNAnt
  
  Set3trainingSel.boot <- set3_selfeatures[sTrainsetDi ,]
  N<-subset(Set3trainingSel.boot, BenignNMaligNAnt=="nonmassB" | BenignNMaligNAnt=="nonmassM")
  ifelse( N$BenignNMaligNAnt == "nonmassB", "NC", "C") -> N$BenignNMaligNAnt    
  Set3trainingSel.boot = data.frame(N)
  
  #####################################
  # 3) Train Random Forest Classifiers
  #####################################
  bootControl <- trainControl(method = "boot", 
                              number = 10, # number of boostrap iterations
                              savePredictions = TRUE,
                              p = 0.75,
                              classProbs = TRUE,
                              returnResamp = "all",
                              verbose = FALSE,
                              summaryFunction = twoClassSummary)
  
  ########## RF set1
  RFGrid <- expand.grid(.mtry=c(1:9) )
  #set.seed(122)
  set1_RFfit <- train(as.factor(BenignNMaligNAnt) ~ ., data = Set1trainingSel.boot[,1:10],
                      method = "rf",
                      trControl = bootControl,
                      tuneGrid = RFGrid,
                      returnData = TRUE,
                      fitBest = TRUE,
                      verbose=FALSE,
                      metric="ROC")
  if(verbose) print(set1_RFfit$finalModel)
  # Classify Training data to get training error and accuracy
  RF1_train <- confusionMatrix(predict(set1_RFfit , newdata = Set1trainingSel.boot[,1:10]), Set1trainingSel.boot$BenignNMaligNAnt )
  if(verbose) print(RF1_train)
  
  ## compute classifier stage 1 performance    
  RFmodel1 <- list(RFmodel1 = set1_RFfit)
  RFmodel1_predValues <- extractPrediction(RFmodel1, testX= Set1trainingSel.boot[,2:10], testY= Set1trainingSel.boot[,1])
  if(verbose) table(RFmodel1_predValues)
  RFmodel1_probValues <- extractProb(RFmodel1, testX=Set1trainingSel.boot[,2:10], testY=Set1trainingSel.boot[,1])
  RFmodel1_testProbs <- subset(RFmodel1_probValues, dataType=="Test")
  
  # Edit probabilities so that can do pooled ROC
  RFmodels_cascade <- RFmodel1_probValues
  RFmodels_cascade$labels <- RFmodels_cascade$obs
  ifelse( RFmodels_cascade$obs == "mass", "P", "N") -> RFmodels_cascade$obs
  ifelse( RFmodels_cascade$pred == "mass", "P", "N") -> RFmodels_cascade$pred
  colnames(RFmodels_cascade)[1] <- "P"
  colnames(RFmodels_cascade)[2] <- "N"
  
  ## plot
  RFmodel1_ROC <- roc(RFmodel1_testProbs$obs, RFmodel1_testProbs$mass,
                      main="Classifier for mass vs. nonmass",
                      percent=TRUE,
                      ci = TRUE,
                      of = "se", 
                      smooth.method="binormal",
                      sp = seq(0, 100, 10))
  if(verbose) print(RFmodel1_ROC) 
  
  ########### RF set2
  RFGrid <- expand.grid(.mtry=c(1:10,15,20,25,30,35,40) )
  #set.seed(122)
  set2_RFfit <- train(as.factor(BenignNMaligNAnt) ~ ., data = Set2trainingSel.boot[1:46],
                      method = "rf",
                      trControl = bootControl,
                      tuneGrid = RFGrid,
                      returnData = TRUE,
                      fitBest = TRUE,
                      verbose=FALSE,
                      metric="ROC")
  if(verbose) print(set2_RFfit$finalModel)
  
  # Classify Training data to get training error and accuracy
  RF2_train <- confusionMatrix(predict(set2_RFfit , newdata = Set2trainingSel.boot), Set2trainingSel.boot$BenignNMaligNAnt )
  if(verbose) print(RF2_train)
  
  ###### Compute ROC
  RFmodel2 <- list(RFmodel2 = set2_RFfit)
  RFmodel2_predValues <- extractPrediction(RFmodel2, testX= Set2trainingSel.boot[,2:46], testY= Set2trainingSel.boot[,1])
  if(verbose) table(RFmodel2_predValues)
  
  RFmodel2_probValues <- extractProb(RFmodel2, testX=Set2trainingSel.boot[,2:46], testY=Set2trainingSel.boot[,1])
  RFmodel2_testProbs <- subset(RFmodel2_probValues, dataType=="Test")
  
  # Edit probabilities so that can do pooled ROC
  ifelse( RFmodel2_probValues$obs == "C", "P", "N") -> RFmodel2_probValues$obs
  ifelse( RFmodel2_probValues$pred == "C", "P", "N") -> RFmodel2_probValues$pred
  colnames(RFmodel2_probValues)[1] <- "P"
  colnames(RFmodel2_probValues)[2] <- "N"
  
  RFmodel2_ROC <- roc(RFmodel2_testProbs$obs, RFmodel2_testProbs$C,
                      percent=TRUE, 
                      ci = TRUE,
                      of = "se", 
                      smooth.method="binormal",
                      sp = seq(0, 100, 10))
  if(verbose) print(RFmodel2_ROC)
  
  ########### RF set3
  RFGrid <- expand.grid(.mtry=c(1:10) )
  #set.seed(122)
  set3_RFfit <- train(as.factor(BenignNMaligNAnt) ~ ., data = Set3trainingSel.boot[1:16],
                      method = "rf",
                      trControl = bootControl,
                      tuneGrid = RFGrid,
                      returnData = TRUE,
                      fitBest = TRUE,
                      verbose=FALSE,
                      metric="ROC")
  if(verbose)  print(set3_RFfit$finalModel)
  
  # Classify Training data to get training error and accuracy
  RF3_train <- confusionMatrix(predict(set3_RFfit , newdata = Set3trainingSel.boot), Set3trainingSel.boot$BenignNMaligNAnt )
  if(verbose) print(RF3_train)
  
  ###### Compute ROC
  RFmodel3 <- list(RFmodel3 = set3_RFfit)
  RFmodel3_predValues <- extractPrediction(RFmodel3, testX= Set3trainingSel.boot[,2:16], testY= Set3trainingSel.boot[,1])
  if(verbose) table(RFmodel3_predValues)
  
  RFmodel3_probValues <- extractProb(RFmodel3, testX=Set3trainingSel.boot[,2:16], testY=Set3trainingSel.boot[,1])
  RFmodel3_testProbs <- subset(RFmodel3_probValues, dataType=="Test")
  
  # Edit probabilities so that can do pooled ROC
  ifelse( RFmodel3_probValues$obs == "C", "P", "N") -> RFmodel3_probValues$obs
  ifelse( RFmodel3_probValues$pred == "C", "P", "N") -> RFmodel3_probValues$pred
  colnames(RFmodel3_probValues)[1] <- "P"
  colnames(RFmodel3_probValues)[2] <- "N"
  
  RFmodel3_ROC <- roc(RFmodel3_testProbs$obs, RFmodel3_testProbs$C,
                      percent=TRUE, 
                      ci = TRUE,
                      of = "se", 
                      smooth.method="binormal",
                      sp = seq(0, 100, 10))
  if(verbose) {
    print(RFmodel3_ROC)
    plot.roc(RFmodel1_ROC,col="#008600")
    par(new=TRUE)
    plot.roc(RFmodel2_ROC, col="#860000")
    par(new=TRUE)
    plot.roc(RFmodel3_ROC, col="#000086")
    legend("bottomright", legend=c("mass/nonmass", "BM only mass", "BM only nonmass"), col=c("#008600","#860000", "#000086"), lwd=2)
  }
  
  RFcascadeoutput <- list(set1_RFfit = set1_RFfit, set2_RFfit = set2_RFfit, set3_RFfit = set3_RFfit)
  saveRDS(RFcascadeoutput, "RFcascadeoutput.rds")
  
  return(RFcascadeoutput)
}



