'''
Created on May 4, 2015

@author: Noah
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from MultinomialNB import article_text_generator, get_article_text
from sklearn import svm
from sklearn.pipeline import Pipeline
import numpy


name_to_index = { "conservative":0.0, "liberal":1.0}
# used for loading training data into classifier
print "Loading corpus"
corpus = numpy.load("corpus.npy")
target = numpy.load("training_target.npy")

# training data loading from file
print "Loading test article data"
test_data = open("test_articles.csv")
lines = [line.strip() for line in test_data.readlines()]
testList = []
testActualCategories = []

counter = 0
print "Analyzing test data"
for line in lines:
    actual_category, url = line.split(",")
    testList.append(url)
    testActualCategories.append(actual_category)
    #target.append(category_index)
    counter = counter + 1
    #update_progress(counter, len(lines))

correct = 0.0
numTests = float(len(testList))

classifier_type = raw_input("Choose classifier type: MultinomialNB (1), SVM (2): ")
if classifier_type == "1":

    clf = Pipeline([('vect', CountVectorizer(min_df=1)), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    clf=  clf.fit(corpus, target)
    for url, actual_category in zip(testList, testActualCategories):
        new_text = get_article_text(url)
        predicted = clf.predict([new_text])
        if predicted == name_to_index[actual_category]:
            correct += 1.0
    
    print "Multinomial accuracy: {}".format(str(100.0*correct/numTests)[:4])
elif classifier_type == "2":
    clf = Pipeline([('vect', CountVectorizer(min_df=1)), ('tfidf', TfidfTransformer()), ('clf', svm.SVC())])
    clf=  clf.fit(corpus, target)
    for url, actual_category in zip(testList, testActualCategories):
        new_text = get_article_text(url)
        predicted = clf.predict([new_text])
        if predicted == name_to_index[actual_category]:
            correct += 1.0
            
    print "SVM accuracy: {}".format(str(100.0*correct/numTests)[:4])
else:
    print "Invalid entry"