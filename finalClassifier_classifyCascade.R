finalClassifier_classifyCascade <- function(casefeatures) {
  # initialize
  library(caret)
  library(randomForest)
  library(MASS)
  library(mlbench)
  library(pROC)
  
  # Load the 3 previously trained classifiers to memory
  RFcascade <- readRDS("C:/Users/windows/Documents/repoCode/finalClassifier/RFcascadeoutput.rds")

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
  

  ## compute classifier stage 1 
  RFmodel1 <- list(RFmodel1 = RFcascade$set1_RFfit)
  RFmodel1_predValues <- extractPrediction(RFmodel1, testX= set1_selfeatures[,2:10], testY= set1_selfeatures[,1])
  print( table(RFmodel1_predValues) )
  
  RFmodel1_probValues <- extractProb(RFmodel1, testX=set1_selfeatures[,2:10], testY=set1_selfeatures[,1])
  MNm_probValues <- subset(RFmodel1_probValues, dataType=="Test")
  
  # instantiate the 2nd stage RF classifiers
  RFmodelM <- list(RFmodelM = RFcascade$set2_RFfit)
  RFmodelNonM <- list(RFmodelNonM = RFcascade$set3_RFfit)
  
  #subseting to fed into 2nd stage
  M.boot <- subset(MNm_probValues, pred=="mass")
  NonM.boot<- subset(MNm_probValues, pred=="nonmass")
  
  ### Select subsets of features correspondingly
  if( !empty(M.boot) ){
    set2_selfeatures <- casefeatures[c("BenignNMaligNAnt",
                                       "Kpeak.countor",
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

    # relabel down the cascade
    if( casefeatures$labels  == "massB") set2_selfeatures$BenignNMaligNAnt = "NC"
    if( casefeatures$labels == "massM") set2_selfeatures$BenignNMaligNAnt = "C"
    if( casefeatures$labels == "nonmassB") set2_selfeatures$BenignNMaligNAnt = "NC"
    if( casefeatures$labels == "nonmassM") set2_selfeatures$BenignNMaligNAnt = "C"
    print(set2_selfeatures)
    
    probValuesM <- extractProb(RFmodelM, testX=set2_selfeatures[,2:46], testY=set2_selfeatures[,1])
    k_probValuesM <- subset(probValuesM, dataType=="Test")
    
    casestest = as.data.frame(setNames(replicate(10, 1, simplify = F), c("caseID", "labels", "pred1", "obs1", "P1", "N1", "pred2","obs2", "P2", "N2")))
    casestest$caseID = as.character( casefeatures[c("id")] )
    casestest$labels = as.character( casefeatures[c("labels")] )
    casestest$pred1 = MNm_probValues$pred
    casestest$obs1 = MNm_probValues$obs
    casestest$P1 = MNm_probValues$mass
    casestest$N1 = MNm_probValues$nonmass
    casestest$pred2 = k_probValuesM$pred
    casestest$obs2 = k_probValuesM$obs
    casestest$P2 = k_probValuesM$C
    casestest$N2 = k_probValuesM$NC
    print(casestest)
    
  }
  if( !empty(NonM.boot) ){
    set3_selfeatures <- casefeatures[c("BenignNMaligNAnt",
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
    
    # relabel down the cascade
    if( casefeatures$labels  == "massB") set3_selfeatures$BenignNMaligNAnt = "NC"
    if( casefeatures$labels == "massM") set3_selfeatures$BenignNMaligNAnt = "C"
    if( casefeatures$labels == "nonmassB") set3_selfeatures$BenignNMaligNAnt = "NC"
    if( casefeatures$labels == "nonmassM") set3_selfeatures$BenignNMaligNAnt = "C"
    print(set3_selfeatures)
    
    probValuesNonM <- extractProb(RFmodelNonM, testX=set3_selfeatures[,2:16], testY=set3_selfeatures[,1])
    k_probValuesNonM <- subset(probValuesNonM, dataType=="Test")
    
    casestest = as.data.frame(setNames(replicate(10, 1, simplify = F), c("caseID", "labels", "pred1", "obs1", "P1", "N1", "pred2","obs2", "P2", "N2")))
    casestest$caseID = as.character( casefeatures[c("id")] )
    casestest$labels = as.character( casefeatures[c("labels")] )
    casestest$pred1 = MNm_probValues$pred
    casestest$obs1 = MNm_probValues$obs
    casestest$P1 = MNm_probValues$mass
    casestest$N1 = MNm_probValues$nonmass
    casestest$pred2 = k_probValuesNonM$pred
    casestest$obs2 = k_probValuesNonM$obs
    casestest$P2 = k_probValuesNonM$C
    casestest$N2 = k_probValuesNonM$NC
    print(casestest)
    
  }
  print(RFcascade)
    
  return(casestest)
}