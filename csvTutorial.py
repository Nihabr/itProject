# Import packages
import csv
import sys
import itertools as iterT
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

# Define some variables we will need for later
# Variables that are declared inside a function will not be usable outside of
# said function. I believe the same rule applies for declared variables in loops.
# In python, you can create variables without specifying the data type,
# and also reassign the variable a new value of a different data type.
dataHeader = ''
trainData = ''

# Create a function called main. In order to run the code in this script
# We will call on the main() function in the bottom of the code.
def printRows (data, firstrowID = None, lastrowID = None):
    """
    Parameters: (list data, int firstrowID = None, int lastrowID = None)

    Description: Prints rowID, Question 1, Question 2, and is_dublicate from
    train.csv in a reader friendly manner. Will start from firstrowID and return
    the amount of rows specified by lastrowID. If no lastrowID is given, will
    return 100 rows.

    Example:
    >>>printRows(0, 5)
    RowID: 0
    Question 1: What is the step by step guide to invest in share market in india?
    Question 2: What is the step by step guide to invest in share market?
    Is duplicate: No

    RowID: 1
    Question 1: What is the story of Kohinoor (Koh-i-Noor) Diamond?
    Question 2: What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?
    Is duplicate: No

    RowID: 2
    Question 1: How can I increase the speed of my internet connection while using a VPN?
    Question 2: How can Internet speed be increased by hacking through DNS?
    Is duplicate: No
    RowID: 3
    Question 1: Why am I mentally very lonely? How can I solve it?
    Question 2: Find the remainder when [math]23^{24}[/math] is divided by 24,23?
    Is duplicate: No

    RowID: 4
    Question 1: Which one dissolve in water quikly sugar, salt, methane and carbon di oxide?
    Question 2: Which fish would survive in salt water?
    Is duplicate: No
    """
    # The above provides a description of the function that can be accessed
    # in other coding circumstances.
    # Notice how the code within the function contains indents. These indents
    # provide code heirarchy. All code on this indentation level
    # will pertain to this function.

    # We want to save the first row of the data as a header, so that
    # we can use this for any models we may want to build. Typically,
    # the first row of csv files will descripe the setup of the csv rows.
    dataHeader = data[0]

    if(firstrowID == None or firstrowID <= 0):
        firstrowID = 1
    if(lastrowID == None or lastrowID <= firstrowID):
        lastrowID = firstrowID + 101
    else:
        lastrowID += 1

    # By using a for loop we can iterate over every row in the reader object
    # And define actions that should be done them.
    # I chose to declare every row as 'row', but this name could be whatever.
    for row in data[firstrowID:lastrowID]:
        tempString =    'RowID: '       + row[0] + '\n'
        tempString +=   'Question 1: '  + row[3] + '\n'
        tempString +=   'Question 2: '  + row[4] + '\n'
        tempString +=   'Is duplicate: '
        if (int(row[5]) == 1):
            tempString += 'Yes' + '\n'
        else:
            tempString += 'No'  + '\n'

        # See the next function, uprint
        uprint(tempString)

# This is a print function that allows us to circumvent issues with annoying characters
def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)

def baseLine (list, columnindex):
    baseline = sum(int(row[columnindex]) for row in list)/len(list)
    return baseline

def readCSVfile (filename):
    data = []

    # Using a 'with' loop, we will open the 'train.csv' file under the name
    # trainData. The 'r' determines how the machine reads the code, and is more
    # technical. I believe this has changed from python 2.7 -> 3.5.2.
    # Code will not exit this while loop (AFAIK), before the trainData variable
    # is closed. However, once closed, trainData cannot be accessed, and the
    # data must be saved in another variable (AFAIK).
    with open (filename, 'r', encoding='utf8') as trainOpen:
        # With the train file open, we will generate a reader object using
        # the trainOpen variable. We store this object as trainReader, and
        # tell it to delimit using the comma sign.
        trainReader = csv.reader(trainOpen, delimiter = ',')
        data = list(trainReader)

        trainOpen.close()

    return data

def writeCSVfile (file, filename, path = ''):
    '''
    Parameters: file list, filename str
    file will be written to the specified path, with the filename
    '''
    with open (path + filename, 'w', encoding='utf8', newline='') as newfile:
        writer = csv.writer(newfile, delimiter = ',')
        writer.writerows(file)

        newfile.close()
    print('File ' + filename + ' has been written to path: ' + 'C:/Users/nihab/workspace/itProject/' + path)

def removeStopwords (data):
    stop = set(stopwords.words('english'))
    # STILL NEED TO APPLY THIS
    stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    for row in data:
        # Treat question 1
        temp = [word.lower() for word in wordpunct_tokenize(row[3]) if word.lower() not in stop]
        row[3] = ' '.join(temp)
        # Treat question 2
        temp = [word.lower() for word in wordpunct_tokenize(row[4]) if word.lower() not in stop]
        row[4] = ' '.join(temp)

    return data

# Source: https://www.kaggle.com/philschmidt/quora-question-pairs/quora-interactive-eda-model-selection
def normalized_word_share(row):
    '''
    Comment by Niklas:
    This function is rather complicated - as such I have provided explanation to help understand what happens:
        set() = "A set is an unordered collection of items. Every element is unique (no duplicates) and must be immutable (which cannot be changed)."
         Source: https://www.programiz.com/python-programming/set
        map() = "Map applies a function to all the items in an input_list." Blueprint: map(function_to_apply, list_of_inputs). Source / documentation: http://book.pythontips.com/en/latest/map_filter.html
        lambda = imbedded function. Like def (function name):, but not global?
        lower(): sets all character to lowercase
        strip(): removes whitespace / punctuation from words
        split(): splits a string into substrings, in this case using space as a splitter. This means the string is seperated into words since all words are seperated by space.
        len(): returns the length of input
    '''
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    print('len(w1): ' + str(len(w1)))
    print('len(w2): ' + str(len(w2)))
    print('len(w1 & w2): ' + str(len(w1 & w2)))

    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        q1words[word] = 1
    for word in str(row['question2']).lower().split():
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R
