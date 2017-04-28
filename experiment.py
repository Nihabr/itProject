import csvTutorial as csvt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STOPWORDS CODE
    # data = csvt.readCSVfile('train.csv')
    # csvTutorial.printRows(data,10,15)
    # data = csvt.removeStopwords(data[:10001])
    # csvt.printRows(data,10,15)
    # print(csvTutorial.baseLine(data[1:], 5))
    # header = data[0]
    # csvt.writeCSVfile(data, 'functionTestFile.csv')

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_info_rows', -1)

df = pd.read_csv('train.csv').fillna('')
df.info()

sum(df[:20].is_duplicate)

csvt.baseLine(df, 'is_duplicate')
df.head(30)

# Use to apply various features
# df ['wordShareRatio'] = df.apply(csvt.normalized_word_share, axis = 1)
# df ['wordShareRatio2'] = df.apply(csvt.word_match_share, axis = 1)
# df ['AvgQlength'] = df.apply(csvt.getQlength, axis = 1)
# df ['nounShareRatio'] = df.apply(csvt.proper_noun_share, axis = 1)
# df = df[df.AvgQlength > 150]
# df = df[df.wordShareRatio < 0.1]
# df.head(10)

# Calculates ratio of is duplicate for df
# print(df.is_duplicate.value_counts()[1] / df.is_duplicate.value_counts()[0])

# Simple scatterPlot
# colors = np.where(df.is_duplicate > 0, 'r', 'b')
# plt.scatter(df.AvgQlength, df.nounShareRatio, c=colors, s = (20, 20))
# plt.show()

# Experimentation with chunking
