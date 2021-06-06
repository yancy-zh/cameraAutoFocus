"""
Created on June 01 2016

@author: Yao Zhang
"""
import numpy as np
import cv2
import iBin_Enc_Dec as ed
import scipy.stats as ss
import os as osc
import xlsxwriter
import imp
from matplotlib import pyplot as plt
mad= imp.load_source("melitta_annotation_display", "C:\Users\deinyazh\Documents\ImageProcessing\Melitta\\melitta_annotation_display.py")
pltU=imp.load_source("plotModule", "C:\Users\deinyazh\Documents\ImageProcessing\\plotModule.py")
#MinDCTValue; /* Minimum DCT value to take into account, typically 8 */
MinDCTValue=8
#MaxHistValue;/* Histogram relative frequency to reach, typically 0.1 */
MaxHistValue=0.1
#img = cv2.imread(readPath + f, cv2.IMREAD_UNCHANGED)
#/* Constants for measure weighting */
Weight=np.array([8,7,6,5,4,3,2,1,
                 7,8,7,6,5,4,3,2,
                 6,7,8,7,6,5,4,3,
                 5,6,7,8,7,6,5,4,
                 4,5,6,7,8,7,6,5,
                 3,4,5,6,7,8,7,6,
                 2,3,4,5,6,7,8,7,
                 1,2,3,4,5,6,7,8])

std_luminance_quant_tbl = np.array([ [16,  11,  10,  16,  24,  40,  51,  61],  
                                     [12,  12,  14,  19,  26,  58,  60,  55],  
                                     [14,  13,  16,  24,  40,  57,  69,  56],  
                                     [14,  17,  22,  29,  51,  87,  80,  62],  
                                     [18,  22,  37,  56,  68, 109, 103,  77],  
                                     [24,  35,  55,  64,  81, 104, 113,  92],  
                                     [49,  64,  78,  87, 103, 121, 120, 101],  
                                     [72,  92,  95,  98, 112, 100, 103,  99]])
                                     
# Scan reorder type for coefficients according to frequency 
scan_order_zz=np.array([0,1,8,16,9,2,3,10,
                        17,24,32,25,18,11,4,5,
                        12,19,26,33,40,48,41,34,
                        27,20,13,6,7,14,21,28,
                        35,42,49,56,57,50,43,36,
                        29,22,15,23,30,37,44,51,
                        58,59,52,45,38,31,39,46,
                        53,60,61,54,47,55,62,63])
scan_order_row=np.array([0,1,2,3,4,5,6,7,
                         8,9,10,11,12,13,14,15,
                         16,17,18,19,20,21,22,23,
                         24,25,26,27,28,29,30,31,
                         32,33,34,35,36,37,38,39,
                         40,41,42,43,44,45,46,47,
                         48,49,50,51,52,53,54,55,
                         56,57,58,59,60,61,62,63])
L_kernel=np.array([[1.0,4.0,1.0],[4.0,-20.0,4.0],[1.0,4.0,1.0]])/6.0
Robert_operator_Horizontal=np.array([0,0,0, 0,1,0, -1,0,0]).reshape(3,3)
Robert_operator_Vertical=np.array([0,0,0,0,1,0,0,0,-1]).reshape(3,3)
Prewitt_operator_Horizontal=np.array([-1,0,1, -1,0,1, -1,0,1]).reshape(3,3)
Prewitt_operator_Vertical=np.array([-1,-1,-1, 0,0,0, 1,1,1]).reshape(3,3)
Horizontal_Sobel_3=np.array([1,2,1,-2,-4,-2,1,2,1]).reshape(3,3)
Vertical_Sobel_3=np.array([1,-2,1, 2,-4,2, 1,-2,1]).reshape(3,3)
# Total weight for normalization - should transform the value to account for 640x480 image

#/* Variables for computation */
DCTnonzeroHist=np.zeros((64,1)) #/* histogram, Initialization of histogram to zero */
blur=0 # /* blur measure */
blksize=8

Files_Path=r"C:\Users\deinyazh\Documents\ImageProcessing\iBin\autoFocusDesign\sampleImg_autofocus_JSD5011\\"
#Files_Path=r"C:\Users\deinyazh\Documents\ImageProcessing\iBin\autoFocusDesign\imgDatasetForAutofocus\allImages"
#Files_Path=r'C:\Users\deinyazh\Documents\ImageProcessing\iBin\autoFocusDesign\TestImages\\'
Files_Path_Results=r'C:\Users\deinyazh\Documents\ImageProcessing\iBin\autoFocusDesign\Test_results\\'
workbook = xlsxwriter.Workbook(Files_Path_Results+'Metrics_JSD5011.xlsx')
worksheet = workbook.add_worksheet()
Header=(['Image Blur degree','Kurtosis-DC','Kurtosis-NoDC','DCTzeros','Mean of Image', 'Variance of Image','Relative Variance', 'Tenengrad-T=0, Sobel',
         'Tenengrad-T=0, Scharr','L_kernel Laplace abs sum-T=0','L_kernel Laplace mean-T=0','L_kernel Laplace abs mean-T=1','L_kernel Laplacian variance - T=0',
         'L_kernel Laplacian abs variance - T=1','Relative L_kernel Laplace dev','Relative L_kernel Laplace abs dev','OPENCV Laplace absolute sum-T=0',
         'OPENCV Laplace mean-T=0','OPENCV Laplace abs mean-T=1','OPENCV Laplacian variance - T=0','OPENCV Laplacian abs variance - T=1','OPENCV Laplace Relative dev',
         'OPENCV abs Laplace Relative dev','SMD','Median variance','Entropy','Contrast', "squared horizontal gradient", "squared vertical gradient", "Robert gradient",
         "Prewitt gradient", "Sobel kernel size 5", "Range of histogram", "Mendelsohn and Mayall", "Vollath F4", "Vollath F5", "Auto-Correlation", "PSNR"])

# Write Header to 
rowx = 0
len=np.size(Header)
# Iterate over the data and write it out row by row.
for col in range(0,np.size(Header)):
    worksheet.write(rowx, col, Header[col])
    col += 1
rowx=1
onlyfiles = [f for f in osc.listdir(Files_Path) if osc.path.isfile(osc.path.join(Files_Path, f))]
#get number of files
#dataFp=np.zeros(size(onlyfiles), dtype=np.float)
#dataFx=np.zeros(size(onlyfiles), dtype=np.int)
dataMetric={}
for f in onlyfiles:
    #Filename constructor
    fname=Files_Path+f
    #Extract Blur Level encoded in last one or two characters of filename
    num=int(f[6:-4])
#    num=int(f[13:-4])
    #Read image    
    img=cv2.imread(fname,0) # read grayscale 
    #Equalize
    #img=cv2.equalizeHist(img)
    #Select stripe
    img=img[220:244,:]
#    img[230:254,:-1]
    #img=img[:,:-1]#
    #Display stripe
    print('the filename is:' + np.str(num)) 
#    mad.disp_image(img)
    #Write stripe to disk    
#    cv2.imwrite('C:\Users\deinyazh\Documents\ImageProcessing\iBin\autoFocusDesign\stripe_focus\\'+np.str(num)+'.tif',img)  
    #Calculate stripe dimensions
    iHeight,iWidth=img.shape
    #Calculate number of blocks of size blocksize in image
    TotalWeight=(iHeight*iWidth)/np.power(blksize,2)
    
    #Caclulate Tenegrad modified (1) and Tenengrad (2) metrics
    xgrad1=cv2.Scharr(img,cv2.CV_64F,1,0) #sobel x
    ygrad1=cv2.Scharr(img,cv2.CV_64F,1,0) #sobel x
    xgrad2=cv2.Sobel(img,cv2.CV_64F,1,0)
    ygrad2=cv2.Sobel(img,cv2.CV_64F,1,0)
    Sim1=(np.power(xgrad1,2)+np.power(ygrad1,2))/16
    Sim2=(np.power(xgrad2,2)+np.power(ygrad2,2))/16 
    
    #draw histogram
    hst,bins = np.histogram(img.ravel(),range(256))
#    plt.bar(bins[:-1], hst, width=1, color='r')

    #Calculate Laplace modified (1) and Laplace OPENCV (2)
    Lim1=cv2.filter2D(img,cv2.CV_64F,L_kernel)
    Lim2=cv2.Laplacian(img,cv2.CV_64F)
    
    #Calculate Sum of Modulus Difference
    SMD1=np.sum(np.abs(img[:,1:]-img[:,0:-1]))
    
    SMD2=np.sum(np.abs(img[0:-1,:]-img[1:,:]))
    #Calculate sum of squared gradient
    SG_horizontal=np.sum(np.power(img[:,1:]-img[:,0:-1], 2))
    SG_horizontal=SG_horizontal/(iHeight*(iWidth-1))
    SG_vertical=np.sum(np.power(img[0:-1,:]-img[1:,:], 2))
    SG_vertical=SG_vertical/((iHeight-1)*iWidth)
    #Apply Robert filter    
    S_Robert_x=cv2.filter2D(img,cv2.CV_64F,Robert_operator_Horizontal)
    S_Robert_y=cv2.filter2D(img,cv2.CV_64F,Robert_operator_Vertical)
    G_Robert=(np.power(S_Robert_x,2)+np.power(S_Robert_y,2))/16
    
    #Apply Prewitt
    S_Prewitt_x=cv2.filter2D(img,cv2.CV_64F, Prewitt_operator_Horizontal)
    S_Prewitt_y=cv2.filter2D(img,cv2.CV_64F, Prewitt_operator_Vertical)
    G_Prewitt=(np.power(S_Prewitt_x,2)+np.power(S_Prewitt_y,2))/16
    
    #Apply Sobel filter kernel size 5
    sobel5_x=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    #(src, dst, xorder, yorder, apertureSize=3) 
    sobel5_y=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    S_sobel5=(np.power(sobel5_x,2)+np.power(sobel5_y,2))/16
    SG_kernel5=np.sum(S_sobel5)/(iHeight*iWidth)
    #Calculate median variance
    compval=cv2.medianBlur(img,3)
    valMed=np.abs(img.astype('float32')-compval.astype('float32'))
    
    #Calculate DCT zeros
    DCTcoeffs = ed.DoDCTTrans(img,blksize)
    img_recover = ed.DoInvDCTTrans_and_DeQuantize(DCTcoeffs,blksize,200*std_luminance_quant_tbl)
    #DCTcoeffs = ed.DoDCTTrans_noshift(img,blksize)
    #DCTcoeffs = ed.DoDCTTrans_and_Quantize(img,blksize,3*std_luminance_quant_tbl)
    DCTCBlock= ed.DoZigZagScan(DCTcoeffs,blksize,scan_order_row) # get result after zig zag scan
    height,width=DCTCBlock.shape
    #/* Compute Histogram *//* Add to histogram if coefficient is big enough */
    for col in range (0,width):
        for row in range(0,height):
            if (np.abs(DCTCBlock[row,col]) > MinDCTValue):
                DCTnonzeroHist[row]+=1;

    for k in range (0,64): 
        #/* add the corresponding weight for all coefficients with a sufficient number of occurences */
        if (DCTnonzeroHist[k] < MaxHistValue*DCTnonzeroHist[0]):
                blur += Weight[k]   
    
    blur /= np.float32(TotalWeight);

    #Kurtosis Calculation All coeffs
    kurt=np.mean(ss.kurtosis(DCTCBlock[0:,:],axis=0, fisher=False, bias=False))
    #Kurtosis Calculation No DC components
    kurtNoDC=np.mean(ss.kurtosis(DCTCBlock[1:,:],axis=0, fisher=False, bias=False))
    #Variance of intensity values
    val_var=np.var(img)
    #mean of image
    val_mean=np.mean(img)
    #Sum-of-Modulus-Difference
    SMD=(SMD1.astype('float32')+SMD2.astype('float32'))/(iHeight*iWidth)
    #Variance of Median
    valMed=valMed.var()
    #Tenengrad Scharr
    S1=np.sum(Sim1)/(iHeight*iWidth)
    #Tenengrad Sobel
    S2=np.sum(Sim2)/(iHeight*iWidth)
    dataMetric[num]=S2
    #Laplacian with external kernel
    L1=np.sum(np.abs(Lim1))/(iHeight*iWidth)
    #Laplacian with external kernel variance
    L1v=Lim1.var()
    #abs Laplacian with external kernel variance
    L1av=np.abs(Lim1).var()
    #Laplacian with external kernel mean
    L1m=Lim1.mean()
    #abs Laplacian with external kernel mean
    L1am=np.abs(Lim1).mean()
    #Laplacian with internal OPENCV kernel
    L2=np.sum(np.abs(Lim2))/(iHeight*iWidth)
    #laplacian with intenral OPENCV kernel variance
    L2v=Lim2.var()
    #abs laplacian with intenral OPENCV kernel variance
    L2av=np.abs(Lim2).var()
    #laplacian with intenral OPENCV kernel mean
    L2m=Lim2.mean()
    #abs laplacian with intenral OPENCV kernel mean
    L2am=np.abs(Lim2).mean()
    # Entropy
    max_freq=np.amax(hst)
    E=ss.entropy(hst/float(max_freq),qk=None, base=2.0)
    #Mason and Green focus measure method

    #Modified Contrast function
    Contr=np.power((np.float32(img.max())-np.float32(img.mean())),2)/np.power(np.float32(img.max())+np.float32(img.mean()),2)
    #Robert
    RG=np.sum(G_Robert)/(iHeight*iWidth)
    #Prewitt
    RP=np.sum(G_Prewitt)/(iHeight*iWidth)
    #range of histogram
    Range_Hist=max_freq-np.amin(hst)
    #Mendelsohn and Mayall
    mean_int=int(L1am)
    qty=255-mean_int+1
    k=range(qty)+np.ones(qty, dtype=np.int)*mean_int
    hk=hst[mean_int-1:]
    MenMay=sum(k*hk)
    #Vollath correlation F4
    Voll_F4=np.sum(img[0:-2,:]*img[1:-1, :])-np.sum(img[0:-3, :]*img[2:-1, :])
    #Vollath correlation F5
    Voll_F5=np.sum(img[0:-2,:]*img[1:-1, :])-(iHeight*iWidth)*val_mean
    #Standard autocorrelation
    k=2
    mat_1=img[:-1, :-(k+1)]-np.ones((iHeight-1, iWidth-(k+1)), dtype=np.float)*val_mean
    mat_2=img[:-1, k:-1]-np.ones((iHeight-1, iWidth-(k+1)), dtype=np.float)*val_mean
    Auto_Correlation=(iHeight*iWidth-k)*val_var-np.sum(mat_1*mat_2)
    
    #PSNR    
#    PSNR=ed.CalcPSNRbaseline(img,img_recover)
    
    worksheet.write(rowx, 0, num)
    worksheet.write(rowx, 1, kurt)
    worksheet.write(rowx, 2, kurtNoDC)
    worksheet.write(rowx, 3, blur)
    worksheet.write(rowx, 4, val_mean)
    worksheet.write(rowx, 5, val_var)
    worksheet.write(rowx, 6, 100*val_var/val_mean)
    worksheet.write(rowx, 7, S2)
    worksheet.write(rowx, 8, S1)
    worksheet.write(rowx, 9, L1)
    worksheet.write(rowx, 10, L1m)
    worksheet.write(rowx, 11, L1am)
    worksheet.write(rowx, 12, L1v)
    worksheet.write(rowx, 13, L1av)
    worksheet.write(rowx, 14, 100*L1v/L1m)
    worksheet.write(rowx, 15, 100*L1av/L1am)
    worksheet.write(rowx, 16, L2)
    worksheet.write(rowx, 17, L2m)
    worksheet.write(rowx, 18, L2am)
    worksheet.write(rowx, 19, L2v)
    worksheet.write(rowx, 20, L2av)
    worksheet.write(rowx, 21, 100*L2v/L2m)
    worksheet.write(rowx, 22, 100*L2av/L2am)
    worksheet.write(rowx, 23, SMD)
    worksheet.write(rowx, 24, valMed)
    worksheet.write(rowx, 25, E)
    worksheet.write(rowx, 26, Contr)
    worksheet.write(rowx, 27, SG_horizontal)
    worksheet.write(rowx, 28, SG_vertical)
    worksheet.write(rowx, 29, RG)
    worksheet.write(rowx, 30, RP)
    worksheet.write(rowx, 31, SG_kernel5)
    worksheet.write(rowx, 32, Range_Hist)
    worksheet.write(rowx, 33, MenMay)
    worksheet.write(rowx, 34, Voll_F4)
    worksheet.write(rowx, 35, Voll_F5)
    worksheet.write(rowx, 36, Auto_Correlation)
#    worksheet.write(rowx, 37, 1/PSNR)
    rowx+=1
workbook.close()   
# save the data from one metric (T)
print( dataMetric)
