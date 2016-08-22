import csv
import numpy
#create test and training data

with open('review_train.csv') as f:
    feature_train=[tuple(line) for line in csv.reader(f)]
feature_train = numpy.array(feature_train)    

#print feature_train

with open('label_train.csv') as f:
    label_train = [tuple(line) for line in csv.reader(f)]
label_train = numpy.array(label_train)    

#print label_train



#from sklearn.cross_validation import train_test_split
#feature_train, feature_test, label_train, label_test = train_test_split(feature_train, label_train, test_size= 0.1, random_state=42)

'''

print feature_train
print label_train
print feature_test
print label_test

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
tokens = feature_train
stemmed = []
print tokens[5][1]
print stemmer.stem(tokens[5][1])

stemmed = []
for item in tokens:
    item[1] = stemmed.append(stemmer.stem(item[1]))
print tokens
'''

tokens = feature_train
comment = []
for item in tokens:
    comment.append(item[1])
    #print item[1]

#tokens = feature_test
#comment_test = []
#for item in tokens:
#    comment_test.append(item[1])

#print comment

#Removing punctuation
import re

# Lowercase, then replace any non-letter, space, or digit character in the headlines.
new_comment = [re.sub(r'[^\w\s\d]','',h.lower()) for h in comment]
# Replace sequences of whitespace with a space character.
new_comment = [re.sub("\s+", " ", h) for h in new_comment]
#print comment
#print new_comment
#unique_words = list(set(" ".join(new_comment).split(" ")))
# We've reduced the number of columns in the matrix a bit.
#print(make_matrix(new_headlines, unique_words))

#new_comment_test = [re.sub(r'[^\w\s\d]','',h.lower()) for h in comment_test]
# Replace sequences of whitespace with a space character.
#new_comment_test = [re.sub("\s+", " ", h) for h in new_comment_test]




#tokenizing
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words = 'english')
X_train_counts = count_vect.fit_transform(new_comment)
#print X_train_counts.shape
#print count_vect.get_feature_names()
X_train_counts = X_train_counts.toarray()

#tfidf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print(X_train_tfidf.todense())
#print X_train_tfidf.shape

#chisquared test

#set binary


col = label_train
new_col = []
for item in col:
	if item == 'Good':
	   #print item
	   new_col.append(1);
	else:
	   new_col.append(0);
#print new_col	      


#k Best
full_matrix = X_train_tfidf
col = new_col
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
selector = SelectKBest(chi2, k='all')
selector.fit(full_matrix, col)
top_words = selector.get_support().nonzero()
chi_matrix = full_matrix[:,top_words[0]]
#print X_train_tfidf.shape
#print chi_matrix.shape


#metafunctions

#take number of capital letters for instance
tokens = feature_train
count_upper = []
for item in tokens:
    #comment.append(item[1])
    #text = numpy.array(item[1])
    text = item[1]
    #print text+"\n"
    upsum = 0
    for char in text:
      if char.isupper():
      	#print " yes+ "
      	upsum = upsum + 1
    #print upsum  	
    count_upper.append(upsum)
    #print "\n"

count_upper = numpy.array(count_upper)
#print count_upper
#print numpy.array(X_train_tfidf)


#take no of user comments
tokens = feature_train
count_number = []
for item in tokens:
    #comment.append(item[1])
    #text = numpy.array(item[1])
    text = item[2]
    #print text+"\n"
    #print upsum    
    count_number.append(text)
    #print "\n"

count_number = numpy.array(count_number)
#print count_number

#take no of user comments
tokens = feature_train
count_like = []
for item in tokens:
    #comment.append(item[1])
    #text = numpy.array(item[1])
    text = item[3]
    #print text+"\n"
    #print upsum    
    count_like.append(text)
    #print "\n"

count_like = numpy.array(count_like)
#print count_like




import numpy as np
#features =scipy.sparse.hstack([count_upper, X_train_tfidf.todense()])
#features = numpy.hstack([count_upper, X_train_tfidf.todense()])
total_data = np.column_stack([chi_matrix.toarray(), count_upper, count_like, count_number])
#print total_data


#Classifier
#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB().fit(total_data, new_col)
from sklearn import svm
clf = svm.SVC()
clf.fit(total_data,new_col)

#Predict class of Test data set

from sklearn.cross_validation import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(total_data, new_col, test_size= 0.1, random_state=42)

#X_new_counts = count_vect.transform(new_comment_test)
#X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#print feature_train
#print feature_test
import sys
from time import time
t1 = time()
predicted = clf.predict(feature_test)
print "predicting time:", round(time()-t1, 3), "s"
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(label_test,predicted)
print accuracy
#print feature_test
print predicted
print label_test

