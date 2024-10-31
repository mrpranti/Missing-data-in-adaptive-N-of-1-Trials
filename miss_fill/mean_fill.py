'''
All the missing value filling methods with mean
'''

## libraries

import pandas as pd
import numpy as np

def mean_fill(dt, method = 'global'): # dt is the data Frame and method global represent the way to mean filling
    # getting the first time cycle point 
    dt_fill = dt.copy()
    miss_dt = dt_fill[dt_fill['outcome'].isna()]
    ts = miss_dt.iloc[0]['t']

    if ts == 0 or method =='global':
        dt_fill.fillna({'outcome': dt_fill['outcome'].mean()}, inplace=True)
    
        
    elif  method == 'individual': ## filling missing value with individual mean
        for row in miss_dt.itertuples():
            index = row.Index
            patient_id = row.patient_id
            fill_value = dt_fill[dt_fill['patient_id'] == patient_id]['outcome'].mean()
            dt_fill.loc[index,'outcome'] = fill_value
    
    return dt_fill

def tr_mean_fill(dt, method = 'global'): # dt is the data Frame and method global represent the way to mean filling
    # getting the first time cycle point 
    dt_fill = dt.copy()
    miss_dt = dt_fill[dt_fill['outcome'].isna()]
    
    for row in miss_dt.itertuples():
        index = row.Index
        patient_id = row.patient_id
        treatment = row.treatment
        t = row.t
        global_fill_value = dt_fill[dt_fill['treatment']== treatment]['outcome'].mean()
        
        ## what if treatment doesn't have a global mean??
         # for now, I am replacing the value with normal mean value
        if (pd.isna(global_fill_value)) & (method == 'global'):
            global_fill_value = dt_fill['outcome'].mean()
           
          # for individual, replacing with individual mean  (possible to also have choice of global and individual here?)
        elif (pd.isna(global_fill_value)) & (method == 'individual'):
            global_fill_value = dt_fill[dt_fill['patient_id'] == patient_id]['outcome'].mean()
            
        if t == 0 or method =='global':
            dt_fill.loc[index,'outcome'] = global_fill_value  
        elif  method == 'individual':
            ## filling missing value with individual's treatment mean
            fill_value = dt_fill[(dt_fill['patient_id'] == patient_id) & (dt_fill['treatment'] == treatment)]['outcome'].mean() 
            ## when Individual doesn't have particular treatmen in the previous time period
            if pd.isna(fill_value): 
                dt_fill.loc[index,'outcome'] = global_fill_value ## replacing with global treatment mean value   
            else:
                dt_fill.loc[index,'outcome'] = fill_value
    
    return dt_fill