# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 10:46:46 2016
Kehtarnavaz and Oh rule-based search algorithm for finding a maximum of a focus measure phi(p) over a set of focus positions p in {a, a+1, ..., b}
@author: deinyazh
"""
import numpy as np
import scipy.interpolate as inter
import xlsxwriter
import imp
metricValues=imp.load_source("read_xls_values", r"C:\Users\deinyazh\Documents\ImageProcessing\iBin\autoFocusDesign\\Read_metric_from_Excel.py")
plot=imp.load_source("pltWithLabel", r"C:\Users\deinyazh\Documents\ImageProcessing\\plotModule.py")
def generate_interpolated_data(positions,Focus_measure_values):       
    # Create a spline interpolant
    s1 = inter.InterpolatedUnivariateSpline (positions, Focus_measure_values)
    return s1

#Position - Focus measure set from Excel
#automatically import metric in excel
xls_name=r'C:\Users\deinyazh\Documents\ImageProcessing\iBin\autoFocusDesign\Test_results\\Metrics_JSD5011.xlsx'

# Assign position and metric columns
position_col=1
metric_col=31 #Tenengrad operator col 9, Sobel 5 col 32, PNSR col 38, prewitt 31
positions,fv = metricValues.read_xls_values(xls_name,position_col,metric_col)
#normalization of sharpness function
fmin=min(fv)
fmax=max(fv)
fvNorm=(fv-fmin)/(fmax-fmin)
fvNormMax=max(fvNorm)
#positions=np.array([0,90,180,270,360,450,540,630,720,810,900,945,990,1035,1080,1170,1260,1350])
#fv=np.array([39.04172607,39.50105387,42.75124349,48.49127767,55.40788859,69.2102002,87.00944946,113.3474386,
#             203.4344499,464.9581812,1339.59995,927.6038464,513.6947021,322.0089832,214.8561247,90.24122559,
#             55.84075602,43.88562052])
# Fucntion call for building the model
model=generate_interpolated_data(positions,fvNorm)

#sampleValues=np.zeros(90)
#i=0
#for x in range(0, 1350, 15):
#    sampleValues[i] = model(x)
#    i+=1
#plot.pltWithLabel(range(0, 1350, 15), sampleValues, "camera position", "fv normalized")

#InitialRange=range(60, 180, 10)
InitialRange=range(0, 360, 45)

#InitialRange=[140]
FineRange=range(1, 50, 1) 
#FineRange=[5, 15]
#MidRange=range(1, 50, 1)
Files_Path_Results=r'C:\Users\deinyazh\Documents\ImageProcessing\iBin\autoFocusDesign\Test_results\\'
workbook = xlsxwriter.Workbook(Files_Path_Results+'BestFocusJSD5011.xlsx')
worksheet = workbook.add_worksheet()
Header=(["Initial Step Size", "Fine Step Size", "Number of Iterations", "Max Focus Pos", "Calc Metric"])
for col in range(0,np.size(Header)):
    worksheet.write(0, col, Header[col])
    col += 1
row=1
flagBreak=False
for Initial in InitialRange:
    if flagBreak==True:
        break
    for Fine in FineRange:
        k=0
        down=0
        Fcurr=0
        Fmax=0
        Coarse=180 #set to 210 for Tenengrad, sobel 5
        Mid=90 # set to 45 for Tenengrad, sobel 5
#        Initial=140
        interval=range(538, 1350)
        p=interval[0] # set to interval[9] to start from camera position 720
        pmax=0
        counter1=counter2=counter3=0
        # add the rule for initial focus is close to Fmax
        if abs(model(p)-fvNormMax)/fvNormMax > 0.0013:
            while p <= interval[-1]:
                counter1+=1
                Fprev=Fcurr
                Fcurr=model(p)
                if k<=5:
                    stepSize=Initial
                else:
                    counter2+=1
                    if Fcurr<= 0.25*Fmax:
                        stepSize=Coarse
                        down=0
                    else:
                        counter3+=1
                        DiffFocus=Fcurr-Fprev
                        if DiffFocus>0.25*Fprev:
                            stepSize=Fine
                            down=0
                        elif stepSize==Fine and DiffFocus>0:
                            down=0
                        elif DiffFocus < 0:
                            if stepSize==Fine:
                                down+=1
                            if down==3:
                                stepSize=Mid
                                down=0
                        else:
                            stepSize=Mid
                            down=0
                if Fcurr>Fmax:
                    Fmax=Fcurr
                    pmax=p
                k+=1
                p+=stepSize
        else:
            pmax=p
            Fmax=model(pmax)
            flagBreak=True
            break
        worksheet.write(row, 0, Initial)
        worksheet.write(row, 1, Fine)
        worksheet.write(row, 2, counter1)
        worksheet.write(row, 3, pmax)
        worksheet.write(row, 4, Fmax)
        row+=1
workbook.close()
#print("counter1=%d", counter1)
#print("counter2=%d", counter2)    
print "counter1=%d" % counter1   

                