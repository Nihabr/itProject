import csvTutorial as csvt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jupyter as jp
from subprocess import check_output

#%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

def setPDdisplay():
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

setPDdisplay()


data = csvt.readCSVfile('train.csv')
# csvTutorial.printRows(data,10,15)
data = csvt.removeStopwords(data[:10001])
# csvt.printRows(data,10,15)
# print(csvTutorial.baseLine(data[1:], 5))
# header = data[0]
csvt.writeCSVfile(data, 'functionTestFile.csv')



df = pd.read_csv('functionTestFile.csv').fillna('')
df.head(10)


df ['wordShareRatio'] = df.apply(csvt.normalized_word_share, axis = 1)
df ['wordShareRatio2'] = df.apply(csvt.word_match_share, axis = 1)
df.head(10)
