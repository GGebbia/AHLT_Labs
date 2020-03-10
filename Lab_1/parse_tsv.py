#!/usr/bin/env python
import pandas as pd

#file:
database = 'names_3star.tsv'

#read 
tsv_read = pd.read_csv(database, sep='\t')


drugs = tsv_read.loc[tsv_read['TYPE'] == 'INN', 'NAME']
    
with open('drug_names_dB.txt', 'a') as output:
    for drug in drugs:
        try:
            output.write(drug + '\n')
        except:
            pass
        
