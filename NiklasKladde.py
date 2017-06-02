import MainParameters as params
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import cross_val_score as cval
from IPython.display import display, HTML

#FOR CONFUSION MATRIX
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

#FOR CROSSFOLD VALIDATION / ROC GRAPH
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

#FOR TIME CALCULATIONS
import time
import datetime

# DEFINITIONS
def init_pd_df (dataset):
    if (dataset == 'train'):
        df = pd.read_csv(params.TRAINPATH)[:params.ROWREADINGLIMIT + 1].fillna('')
        pd.set_option('display.max_colwidth', -1)
        pd.set_option('display.max_info_rows', -1)
    if (dataset == 'test'):
        df = pd.read_csv(params.TESTPATH).fillna('')
        pd.set_option('display.max_colwidth', -1)
        pd.set_option('display.max_info_rows', -1)
    return df

def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split()))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split()))

    return (1.0 * len(w1 & w2)/(len(w1) + len(w2)))*2

def posTagger (row):
    tagList = []
    q1 = nltk.pos_tag(row['question1'].strip().split())
    q2 = nltk.pos_tag(row['question2'].strip().split())
    tagList.append(q1)
    tagList.append(q2)
    # print (tagList[:10])
    return tagList

def nounShare (row):
    tagList = posTagger(row)
    counter = 0
    q1Nouns = []
    q2Nouns = []
    '''
    SEARCHES FOR FOLLOWING TAGS:
        NN	 = Noun, singular or mass
        NNS	 = Noun, plural
        NNP	 = Proper noun, singular
        NNPS = Proper noun, plural
    '''
    for idx, tag in enumerate(tagList[0]):
        if (tag[1] in {'NN', 'NNP', 'NNS', 'NNPS'}):
            q1Nouns.append(tag)
    for idx, tag in enumerate(tagList[1]):
        if (tag[1] in {'NN', 'NNP', 'NNS', 'NNPS'}):
            q2Nouns.append(tag)

    q1Nouns = set(q1Nouns)
    q2Nouns = set(q2Nouns)
    if (bool(q1Nouns) == False or bool(q2Nouns) == False):
        return 0
    else:
        return  (1.0 * len(q1Nouns & q2Nouns)/(len(q1Nouns) + len(q2Nouns)))*2

def verbShare (row):
    tagList = posTagger(row)
    counter = 0
    q1Verbs = []
    q2Verbs = []
    '''
    SEARCHES FOR FOLLOWING TAGS:
        VB	= Verb, base form
        VBD	= Verb, past tense
        VBG	= Verb, gerund or present participle
        VBN	= Verb, past participle
        VBP	= Verb, non-3rd person singular present
        VBZ	= Verb, 3rd person singular present
    '''
    for idx, tag in enumerate(tagList[0]):
        if (tag[1] in {'V','VBD','VBG','VBN','VBP','VBZ'}):
            q1Verbs.append(tag)
    for idx, tag in enumerate(tagList[1]):
        if (tag[1] in {'V','VBD','VBG','VBN','VBP','VBZ'}):
            q2Verbs.append(tag)

    q1Verbs = set(q1Verbs)
    q2Verbs = set(q2Verbs)
    if (bool(q1Verbs) == False or bool(q2Verbs) == False):
        return 0
    else:
        return  (1.0 * len(q1Verbs & q2Verbs)/(len(q1Verbs) + len(q2Verbs)))*2
    # Won't work. Lists need to be 'sets', as in the normalized_word_share method
    # print ('Nounshare: ' + ratio)

def getAvgQlength (row):
    return (len(row['question1']) + len(row['question2']))/2

def cvalMeanScores (cvVal, data, target, model):
    scores = cval(model, data, target, cv = cvVal['Folds'])
    return scores.mean()

def cvalScores (cvVal, data, target, model):
    scores = cval(model, data, target, cv = cvVal)
    r = {'Mean': scores.mean(), 'stdDeviance': scores.std()}
    return r

def QLengthDiff (row):
    return (abs(len(row['question1']) + len(row['question2'])))

def showFalsePositives (predicted, actual):
    pred = pd.DataFrame(data = predicted[:], index = actual.index, columns = ['predicted'])
    temp = pd.concat([actual, pred.predicted], axis = 1)
    #df.loc[df['column_name'] == some_value]
    temp = temp.loc[temp['is_duplicate'] == 0]
    temp = temp.loc[temp['predicted'] == 1]
    return temp

def showTrueNegatives (predicted, actual):
    pred = pd.DataFrame(data = predicted[:], index = actual.index, columns = ['predicted'])
    temp = pd.concat([actual, pred.predicted], axis = 1)
    #df.loc[df['column_name'] == some_value]
    temp = temp.loc[temp['is_duplicate'] == 1]
    temp = temp.loc[temp['predicted'] == 0]
    return temp

def applyAttributes (df, batchlen):
    totalTimeStart = time.clock()
    datalength = len(df)
    batches =  datalength / batchlen
    print('Data rowcount: %s \nNumber of batches: %s' % (datalength, batches))
    print('Applying Attributes...')
    currentbatch = 0
    result = pd.DataFrame()
    while (currentbatch < batches):
        currentbatch += 1
        print('--- BATCH %i OF %s ---' % (currentbatch, batches))
        start = time.clock()

        if (currentbatch < batches):
            print('BATCH INTERVAL: [' + str((currentbatch-1)*batchlen) + ' - ' + str(currentbatch*batchlen) + ']')
            tempdf = df.iloc[(currentbatch-1)*batchlen:currentbatch*batchlen]
        else:
            print('BATCH INTERVAL: ' + str((currentbatch-1)*batchlen) + ' - ' + str(int(df.tail(1).get('id'))) + ']')
            tempdf = df.iloc[(currentbatch-1)*batchlen : int(df.tail(1).get('id'))]

        tempdf ['WordShare'] = tempdf.apply(normalized_word_share, axis = 1)
        print('WordShare applied for batch %i' % (currentbatch))
        tempdf ['NounShare'] = tempdf.apply(nounShare, axis = 1)
        print('NounShare applied for batch %i' % (currentbatch))
        tempdf ['VerbShare'] = tempdf.apply(verbShare, axis = 1)
        print('VerbShare applied for batch %i' % (currentbatch))
        tempdf ['AvgQLength'] = tempdf.apply(getAvgQlength, axis = 1)
        print('AvgQLength applied for batch %i' % (currentbatch))
        tempdf ['QLenDiff'] = tempdf.apply(QLengthDiff, axis = 1)
        print('QLenDiff applied for batch %i' % (currentbatch))
        end = time.clock()
        result = pd.concat([result, tempdf], axis = 0)
        print ('Batch time: ' + str(datetime.timedelta(seconds =(end - start))))
        print ('Current total time: ' + str(datetime.timedelta(seconds =(end - totalTimeStart))))
        print('--- END OF BATCH ---\n')

    totalTimeEnd = time.clock()
    print ('TOTAL TIME: ' + str(datetime.timedelta(seconds =(totalTimeEnd - totalTimeStart))))
    return result

# CODE START
# Initiate a dataframe based on our init_pd_df function
df = init_pd_df('train')

# Engineer and apply attributes to dataframe (using our functions)
df = applyAttributes(df, 5000)
# Relevant useful snippets:
# df = df[df.wordShareRatio > 0.1]
# df.iloc[[13]]
# df.describe()
# df.info()

# VISUALIZE
colors = np.where(df.is_duplicate > 0, 'r', 'b')
plt.scatter(df['wordShareRatio'], df['NounShare Ratio'], c=colors, s = (20, 20))
plt.show()

# ----------------------------------------------------------------------------
# CREATE RANDOMFOREST MODEL
    '''
    From: https://chrisalbon.com/machine-learning/random_forest_classifier_example_scikit.html
    '''
# SPLIT INTO TRAINING AND TEST DATA
# Splits test into 25%, and train to 75%
rowSplit = int(params.ROWREADINGLIMIT * 0.90)
train, test = df[:rowSplit], df[rowSplit:]

#TRAIN RANDOMFOREST
features = train.columns[6:]
trainTarget = train['is_duplicate']
model = rfc(n_jobs = -1)
model.fit(train[features], trainTarget)

# RANDOMFOREST PREDICTIONS
#Actual test dataset
test = init_pd_df('test')
test = applyAttributes(test, 5000)
test.info()
predictions = model.predict(test[features])

# ----------------------------------------------------------------------------
#RANDOMFOREST CONFUSION MATRIX
pd.crosstab(test['is_duplicate'], predictions, rownames=['Actual is_dup'], colnames=['Predicted is_dup'])
# DISPLAY FALSE POSITIVES FROM ABOVE MATRIX (PREDICTED DUPLICATE, BUT NOT ACTUALLY DUPLICATE)
showFalsePositives(predictions, test).head(10)

# DISPLAY TRUE NEGATIVES FROM ABOVE MATRIX (NOT PREDICTED DUPLICATE, BUT ACTUALLY DUPLICATE)
showTrueNegatives(predictions, test).head(10)

#FEATURE IMPORTANCE
fList = list(zip(train[features], model.feature_importances_))
feat = pd.DataFrame(fList, columns = ('Feature', 'Importance'))
feat.head(5)
y_pos = np.arange(len(feat['Feature']))
plt.bar(y_pos, feat['Importance'], align = 'center', alpha = 0.5)
plt.suptitle('Feature Importance')
plt.xticks(y_pos, feat['Feature'])
plt.ylabel('Importance')
plt.show()

# ----------------------------------------------------------------------------
# CROSS VALIDATION WITH ROC AUC SCORE
    '''
    From: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    '''
# Specify data
X = df[features]
y = df['is_duplicate']

# X, y = X[y != 2], y[y != 2]

n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
print(X)
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)
# classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
for (train, test), color in zip(cv.split(X, y), colors):
    probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

mean_tpr /= cv.get_n_splits(X, y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# ----------------------------------------------------------------------------
