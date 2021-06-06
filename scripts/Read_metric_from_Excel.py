# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 10:46:46 2016
Kehtarnavaz and Oh rule-based search algorithm for finding a maximum of a focus measure phi(p) over a set of focus positions p in {a, a+1, ..., b}
@author: deinyazh
"""
import numpy as np
import scipy.interpolate as inter
import xlsxwriter
import xlrd
from operator import itemgetter

def generate_interpolated_data(positions,Focus_measure_values):       
    # Create a spline interpolant
    s1 = inter.InterpolatedUnivariateSpline (positions, Focus_measure_values)
    return s1

    
def read_xls_values(xls_name,positions_column,metric_column):
    
    #xls_name: Excel sheet containing the metrics
    #positions_column: column of recorded positions to take images wrt Excel columns (start at 1)
    #metric_column: column containing the metric to be evaluated wrt best focus position (start at 1)
    
    positions_column=positions_column-1
    metric_column=metric_column-1    
    book = xlrd.open_workbook(xls_name)
    
    #Get Sheet with metrics
    sh = book.sheet_by_index(0)

    #Assume row zero has a title so start from row 1    
    row=1
       
    positions=np.zeros(((sh.nrows-1,1)))
    fv=np.zeros(((sh.nrows-1,1)))  
    while row<=sh.nrows-1:
        positions[row-1,0]=sh.cell_value(row,positions_column)
        fv[row-1,0]=sh.cell_value(row,metric_column)
        print "numer of row: %d, value of fv: %f" %(row,fv[row-1])
        row+=1

    #Take care of possibly unsorted Excel cells so that positions are always in *increasing* order       
    index=np.argsort(positions,axis=0)
    positions=positions[index[:,0],:]
    fv=fv[index[:,0],:]

    return positions,fv

#------------------------------------------------------------------------------------------------#
#Example

#Position - Focus measure set from Excel file
#Get Excel file with Metrics

#xls_name=r'C:\Users\deinyazh\Documents\ImageProcessing\iBin\autoFocusDesign\Test_results\\Metrics_TestImages0708.xlsx'
#
## Assign position and metric columns
#position_col=1
#metric_col=37 #Tenengrad operator - Sobel
#
#positions,fv = read_xls_values(xls_name,position_col,metric_col)
