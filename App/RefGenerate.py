#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 03:57:20 2022

@author: ubuntu
"""

from multiprocessing import Pool
import matplotlib.pyplot as plt
import os,glob,sys
import re
import config
sys.path.insert(0, '../Lib')
import HEVCInterfaceEncoder

outputFilePath = config.outputFilePath
mdcConfigPath = config.mdcConfigPath
sequencePath = config.sequencePath
resultPath = config.resultPath
libPath = config.libPath
QPPath = config.QPPath

QPstep = config.QPstep
QP_min = config.QP_min

QPMstep = config.QPMstep
QPM_min = config.QPM_min
QPM_max = config.QPM_max

def worker(obj):
    obj.runHEVCEncoderNoVerbose()

listFileRef = []
listEncCfgorgFileName = config.listEncCfgorgFileName
listSeqCfgFileName = config.listSeqCfgFileName

for i in range (QP_min,QPM_max,QPstep):
    listFileRef.append([resultPath+"str_org/str_org_%d.hevc"%(i),resultPath+"rec_org/rec_org_%d.yuv"%(i),"",i])
listEncoderHevcOrg = [HEVCInterfaceEncoder.HEVCEncoder(mdcConfigPath+listEncCfgorgFileName[0],
                                                      sequencePath+listSeqCfgFileName[0],
                                                      i[0],i[1],i[2],i[3]) for i in listFileRef]
with Pool() as p:
    p.map(worker,listEncoderHevcOrg)
