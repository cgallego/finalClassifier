finalClassifier_classifyCascade <- function(casefeatures) {
  # initialize
  library(caret)
  library(randomForest)
  library(MASS)
  library(mlbench)
  library(pROC)
  
  # Load the 3 previously trained classifiers to memory
  RFcascade <- readRDS("C:/Users/windows/Documents/repoCode/finalClassifier/RFcascadeoutput.rds")
  print( RFcascade )
  
  ## compute classifier stage 1 
  RFmodel1 <- list(RFmodel1 = RFcascade$set1_RFfit)
  ### Select subsets of features correspondingly
  set1_selfeatures <- casefeatures[c("BenignNMaligNAnt",
                                     "circularity",
                                     "Tpeak.inside",
                                     "Slope_ini.inside",
                                     "SER.inside",
                                     "SER.countor",
                                     "Kpeak.inside",
                                     "iiMin_change_Variance_uptake",
                                     "iMax_Variance_uptake",
                                     "Kpeak.countor")]
  
  print( summary(set1_selfeatures) )  
  
  RFmodel1_predValues <- extractPrediction(RFmodel1, testX= set1_selfeatures[,2:10], testY= set1_selfeatures[,1])
  print( table(RFmodel1_predValues) )
  
  RFmodel1_probValues <- extractProb(RFmodel1, testX=set1_selfeatures[,2:10], testY=set1_selfeatures[,1])
  RFmodel1_testProbs <- subset(RFmodel1_probValues, dataType=="Test")
  print( RFmodel1_testProbs )
  
  return(RFcascade)
}