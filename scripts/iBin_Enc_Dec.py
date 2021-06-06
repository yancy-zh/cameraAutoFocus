# -*- coding: utf-8 -*-
"""
Created on Wed May 04 12:29:24 2016

@author: deinnisg
"""
import cv2
import math as mth
import numpy as np

def DoDCTTrans_noshift(img,blksize):
    iHeight, iWidth = img.shape[:2]
    img2 = np.empty(shape=(iHeight, iWidth))
    img=img.astype('float32')
    #img=img-128
    for startY in range(0, iHeight, blksize):
        for startX in range(0, iWidth, blksize):
            block = img[startY:startY+blksize, startX:startX+blksize]
            blockf = np.float32(block)
            dst = cv2.dct(blockf)
            img2[startY:startY+blksize, startX:startX+8] = dst
    return img2
    
def DoDCTTrans(img,blksize):
    iHeight, iWidth = img.shape[:2]
    img2 = np.empty(shape=(iHeight, iWidth))
    img=img.astype('float32')
    img=img-128
    for startY in range(0, iHeight, blksize):
        for startX in range(0, iWidth, blksize):
            block = img[startY:startY+blksize, startX:startX+blksize]
            blockf = np.float32(block)
            dst = cv2.dct(blockf)
            img2[startY:startY+blksize, startX:startX+8] = dst
    return img2


def DoDCTTrans_and_Quantize(img,blksize,qtable):
    iHeight, iWidth = img.shape[:2]
    img2 = np.empty(shape=(iHeight, iWidth))
    img=img.astype('float32')
    img=img-128
    for startY in range(0, iHeight, blksize):
        for startX in range(0, iWidth, blksize):
            block = img[startY:startY+blksize, startX:startX+blksize]
            blockf = np.float32(block)
            dst = cv2.dct(blockf)
            img2[startY:startY+blksize, startX:startX+8] = np.floor(np.divide(dst, qtable)+0.5)
    return img2

def DoInvDCTTrans_and_DeQuantize(img,blksize,qtable):
    iHeight, iWidth = img.shape[:2]
    img2 = np.empty(shape=(iHeight, iWidth))
    for startY in range(0, iHeight, blksize):
        for startX in range(0, iWidth, blksize):
            block = img[startY:startY+blksize, startX:startX+blksize]
            blockf = np.float32(block)
            #dst=np.floor(np.multiply(blockf, qtable)+0.5)
            dst=np.multiply(blockf, qtable)
            dst = cv2.idct(dst)
            img2[startY:startY+blksize, startX:startX+blksize] = dst
    img2=img2.astype('float32')+128    
    return img2 

def DoZigZagScan(trans_img,blksize,pattern):
    iHeight, iWidth = trans_img.shape[:2]
    imgout=np.empty(shape=(np.power(blksize,2),(iHeight*iWidth)/np.power(blksize,2)))
    i=0    
    for startY in range(0, iHeight, blksize):
      for startX in range(0, iWidth, blksize):
         block = trans_img[startY:startY+blksize, startX:startX+blksize]
         imgout[:,i]=block[np.unravel_index(pattern,block.shape)]
         i+=1
    return imgout
    
def DoDCencoding(DCcoeffs):
    d_shift_right=np.hstack((0,DCcoeffs.T))
    d_shift_left=np.hstack((DCcoeffs.T,0))
    diff=d_shift_left-d_shift_right
    #diff=1
    return diff
    
def CalcPSNR(orig,decomp):
    #Additional testing with normalization to 255 for maxval
    iHeight, iWidth = orig.shape[:2]
    se=cv2.sumElems(np.power((orig-decomp),2))
    mse=np.float(se[0])/np.float((iHeight*iWidth))
    maxval=np.max(orig)
    psnr=10.0*mth.log10(float(maxval*maxval)/float(mse))
    return psnr
    
def CalcPSNRbaseline(orig,decomp):
    #Additional testing with normalization to 255 for maxval
    import matplotlib.pyplot as plt
    
#    plt.imshow(orig, cmap=plt.cm.gray)
#    plt.imshow(decomp, cmap=plt.cm.gray)
    iHeight, iWidth = orig.shape[:2]
    se=cv2.sumElems(np.power((orig-decomp),2))
    mse=np.float(se[0])/np.float((iHeight*iWidth))
    maxval=255#np.max(orig)
    psnr=10.0*mth.log10(float(maxval*maxval)/float(mse))
    return psnr
    
def CalcPSNRbaselineBlock(orig,decomp,blksize):
    #Additional testing with normalization to 255 for maxval
    iHeight, iWidth = orig.shape[:2]
    psnr = np.empty(shape=(iHeight/blksize, iWidth/blksize))
    maxval=255#np.max(orig)
    for startY in range(0, iHeight, blksize):
        for startX in range(0, iWidth, blksize):
            blockorig = np.float32(orig[startY:startY+blksize, startX:startX+blksize])
            blockdecomp = np.float32(decomp[startY:startY+blksize, startX:startX+blksize])
            se=cv2.sumElems(np.power((blockorig-blockdecomp),2))
            mse=np.float(se[0])/np.float((iHeight*iWidth)/np.power(blksize,2))
            psnr[np.int(np.float(startY)/blksize),np.int(np.float(startX)/blksize)]=10.0*mth.log10(float(maxval*maxval)/float(mse))
    return psnr