'''
Created on May 3, 2015

@author: Noah
'''

from MultinomialNB import article_text_generator
import numpy

def update_progress(current, max):
    num_out_of_ten = int(10.0 * float(current)/max)
    prog_str = "[{0}{1}] {2}%\r".format('#'*num_out_of_ten, ' '*(10- num_out_of_ten), str(100.0*float(current)/max)[:4])
    print prog_str
    
    
print "Generating corpus..."
# generate corpus
corpus = []
target = []

training_file = open("training_articles.csv")
lines = [line.strip() for line in training_file.readlines()]

counter = 0
for category_index, article_text in article_text_generator(lines):
    corpus.append(article_text)
    target.append(category_index)
    counter = counter + 1
    update_progress(counter, len(lines))
    
numpy.save("corpus", corpus)
numpy.save("training_target", target)