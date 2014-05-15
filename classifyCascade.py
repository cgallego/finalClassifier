# -*- coding: utf-8 -*-
"""
Create class for passing a new case's features to the cascade classifier and obtain a Prediction

Created on Tue May 13 12:33:11 2014

@ author (C) Cristina Gallego, University of Toronto, 2014
"""

import os, os.path
import sys
import string
from sys import argv, stderr, exit
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

import pandas as pd
import pylab    
 
# convertion packages
import pandas.rpy.common as com
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as R
from rpy2.robjects import globalenv

#!/usr/bin/env python
class classifyCascade(object):
    """
    USAGE:
    =============
    classifier = classifyCascade()
    """
    def __init__(self): 
        """ initialize """           
        # use cell picker for interacting with the image orthogonal views.
        self.rpycasesFrame = []
        self.RFcascade_probs = []
        self.veredict = []
        
    def __call__(self):       
        """ Turn Class into a callable object """
        classifyCascade()
        
    def parse_classes(self, cascadeprobs):
        """ Take prob outputs of cascade classifiers and contrast it with labels to produce output accuracy"""
        caselabel = cascadeprobs['labels'].iloc[0]
        stage1label = caselabel[:-1] # mass vs. nonmass
        stage2label = caselabel[-1]  # B vs. M
        if stage2label == 'B': stage2label = 'NC'
        if stage2label == 'M': stage2label = 'C'
        
        hit = []
        miss = []        
        # proccess 2 possible correct classifications (e.g hit)
        if (cascadeprobs['pred1'].iloc[0] == stage1label):
            hit.append('stage1')
        else:
            miss.append('stage1')
            
        # proccess 2 possible correct classifications (e.g hit)
        if (cascadeprobs['pred2'].iloc[0] == stage2label):
            hit.append('stage2')
        else:
            miss.append('stage2')
        
        #procees correct results
        if( hit == ['stage1', 'stage2']):
            self.veredict = True
            self.caseoutcome = "P_stage1_P_stage2"
        if( hit == ['stage2'] and miss == ['stage1'] ):
            self.veredict = True
            self.caseoutcome = "N_stage1_P_stage2"
        
        #procees incorrect results
        if( miss == ['stage1', 'stage2']):
            self.veredict = False
            self.caseoutcome = "N_stage1_N_stage2"
        if( hit == ['stage1'] and miss == ['stage2'] ):
            self.veredict = False
            self.caseoutcome = "P_stage1_N_stage2"
    
        return(self.veredict, self.caseoutcome)            
            
        
    def case_classifyCascade(self):
        """ A individual case classification function"""
        ########### To R for classification
        os.chdir("Z:\Cristina\MassNonmass\codeProject\codeBase\extractFeatures\casesDatabase")        
        cF = pd.read_csv('casesFrames_toclasify.csv')
        
        cF['finding.mri_mass_yn'] = cF['finding.mri_mass_yn'].astype('int32')
        cF['finding.mri_nonmass_yn'] = cF['finding.mri_nonmass_yn'].astype('int32')
        cF['finding.mri_foci_yn'] = cF['finding.mri_foci_yn'].astype('int32')
        cF['finding.mri_foci_yn'] = cF['finding.mri_foci_yn'].astype('int32')
        cF['is_insitu'] = cF['is_insitu'].astype('int32')
        cF['is_invasive'] = cF['is_invasive'].astype('int32')
                
        self.rpycasesFrame = com.convert_to_r_dataframe(cF)
        base = importr('base')
        base.source('Z:/Cristina/MassNonmass/codeProject/codeBase/finalClassifier/finalClassifier_classifyCascade.R')
        
        RFcascade = globalenv['finalClassifier_classifyCascade'](self.rpycasesFrame)
        
        self.RFcascade_probs = com.convert_robj(RFcascade)
        print "\n========================"
        print self.RFcascade_probs
        
        # proccess possible outcome
        [veredict, caseoutcome] = self.parse_classes(self.RFcascade_probs)
        print "\n========================\nCascade classification result:"
        print veredict
        print caseoutcome
        
        return
