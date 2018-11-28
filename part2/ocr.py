#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
###############################################################################
# CS B551 Fall 2018, Assignment #3 - Part 2: Optical Character Recognition (OCR)
#
# Completed by:
# Derrick Eckardt
# derrick@iu.edu
# (Based on skeleton code by D. Crandall)
#
# Completed on November 25, 2018
#
# For the assignment details, please visit:
#
# https://github.iu.edu/cs-b551-fa2018/derrick-a3/blob/master/a3.pdf
#
################################################################################
################################################################################
# Part 1 - Part of Speech Tagging
################################################################################
################################################################################
#
# 
# 
# 
################################################################################
# Emission Probabilities
################################################################################
# 
# Wow.  This was by far one of the most difficult items that i dealt with.
# 
# If you look into my code, you will see lots of pO_of_L commented out of my
# code.  These represent the many different emission probabilities that I tried.
# The biggest issue I faced with that is that many of the methods preferred
# absolute pixel matches.  For the punctation and blank spaces, those would have
# very high hit ratios, despite, it being mostly not useful information.

from PIL import Image, ImageDraw, ImageFont

# This article helped figure out how to call different folder
# https://www.reddit.com/r/learnpython/comments/3pzo9a/import_class_from_another_python_file/
import sys
sys.path.append('../part1/')

from pos_scorer import Score    
from pos_solver import *
from operator import itemgetter
from collections import Counter
from pprint import pprint
from time import sleep

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # print im.size
    # print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# Have to add in blank spaces between words and before and after sentences since those
# are present in real life, and needed in this example.
def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        example = ""
        for word, state in zip(line.split()[0::2], line.split()[1::2]):
            if state != ".":
                example += word + " "
            else:
                example = example[:-1] + word + " "
        exemplars.extend([(example+"")])
    return exemplars


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
training_text = read_data(train_txt_fname)
# print training_text

## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!

# Train letters
# p(l), p(l2 | l1), p(L1)
pL_count = Counter()
pL1_count = Counter()
pL2_pL1_count = {}
for letter in train_letters:
    pL2_pL1_count[letter] = Counter()
    
for line in training_text:
    last_letter = ""
    for letter in line:
        pL_count[letter] += 1
        if last_letter != "":
            if last_letter in pL2_pL1_count.keys():
                pL2_pL1_count[last_letter][letter] += 1
            else:
                pL2_pL1_count[last_letter] = Counter()
                pL2_pL1_count[last_letter][letter] += 1
        last_letter = letter
    pL1_count[line[0]] += 1

total_char = float(sum(pL_count.values()))
total_pL1 = float(sum(pL1_count.values()))
# print pL1_count

# For unknown characters    
smoother = 1 / total_char
pL1_smoother = 1 / total_pL1


# Simple method
# Compare each test letter to the train letter.  The one with the highest percentage of similar
# points is the winner
def simple(train_letters, test_letters):
    simple_text = ""
    # cycle through each test lette
    
    
    for test in test_letters[:]:
        test_scores = []
        # Sort through each letter in the train set, a
        test_string = "".join(test)
        t = 0.05# tuning parameter, as suggested by instruction and @590 in Piazza
        for train in train_letters:
            train_string = "".join(train_letters[train])
            m = sum([1.0 if train_dot == test_dot and train_dot == "*" else 0.05 if train_dot == test_dot and train_dot == " " else 0.0 for train_dot, test_dot in zip(train_string, test_string)]) # dots in common, as suggested by instruction and @590 in Piazza
            N = float(len(test_string)) #float(sum([1 if train_dot == "*" else 0 for train_dot in train_string]))  # Number of dots, as suggested by instruction and @590 in Piazza
            # pO_of_L = ((1-t)**m) * (t**(N-m))
            pO_of_L = m / (N*10.0)
            
            # print pO_of_L
            # pO_of_L = sum([1 if train_dot == test_dot else 0 for train_dot, test_dot in zip(train_string, test_string)]) / float(len(train_string))
            pL = (pL_count[train] + smoother) / (total_char + smoother)
            test_scores.extend([[train, log(pO_of_L)]]) #pL
        # print sorted(test_scores, key=itemgetter(1), reverse = True)
        simple_text += sorted(test_scores, key=itemgetter(1), reverse = True)[0][0]

    return simple_text


def viterbi(train_letters,test_letters):
    viterbi_model = []

    # first letter in image
    test_string = "".join(test_letters[0])
    t = 0.20 # tuning parameter, as suggested by instruction and @590 and @615 in Piazza
    # N = float(len(test_string)) # Number of dots, as suggested by instruction and @590 in Piazza
    viterbi_temp = []
    total_pO_of_L = 0
    for train in train_letters:
        # sublist to the form of [position_no, value, path, pos], only with the words that appear
        # one added to both in the event it is a new word or a word being used in a new form.
        train_string = "".join(train_letters[train])
        m = sum([10.0 if train_dot == test_dot and train_dot == "*" else 1.25 if train_dot == test_dot and train_dot == " " else 0.0 for train_dot, test_dot in zip(train_string, test_string)])# dots in common, as suggested by instruction and @590 in Piazza
        j = sum([1.0 if train_dot == test_dot else 0.0 for train_dot, test_dot in zip(train_string, test_string)]) # dots in common, as suggested by instruction and @590 in Piazza
        N = float(len(test_string)) #float(sum([1 if train_dot == "*" else 0 for train_dot in train_string]))  # Number of dots, as suggested by instruction and @590 in Piazza
        pO_of_L = (m / (N*10.0))#**(N-j)
        # pO_of_L = ((1-t)**j) * (t**(N-j))
        
        # print j, N - j
        # pO_of_L = ((j / N) ** j) * (((N-j)/N))**(N-j)
        
        # print pO_of_L, log(pO_of_L) *175
        
        viterbi_model.extend([[0,222*log(pO_of_L), train, train]])  # + log((pL1_count[train]+pL1_smoother)/float(total_pL1+pL1_smoother))
    # print sorted(viterbi_model,key=itemgetter(1), reverse=True)

    # Rest of text
    for test in test_letters[1:]:
        test_string = "".join(test)
        viterbi_maxes =[]
        # t = 0.05 # tuning parameter, as suggested by instruction and @590 in Piazza
        # N = float(len(test_string)) # Number of dots, as suggested by instruction and @590 in Piazza
        for train in train_letters:
            # sublist to the form of [position_no, value, path, letter], only with the words that appear
            # one added to both in the event it is a new word or a word being used in a new form.
            viterbi_temp = []
            train_string = "".join(train_letters[train])
            m = sum([10.0 if train_dot == test_dot and train_dot == "*" else 1.25 if train_dot == test_dot and train_dot == " " else 0.0 for train_dot, test_dot in zip(train_string, test_string)]) # dots in common, as suggested by instruction and @590 in Piazza
            j = sum([1.0 if train_dot == test_dot else 0.0 for train_dot, test_dot in zip(train_string, test_string)]) # dots in common, as suggested by instruction and @590 in Piazza

            N = float(len(test_string)) #float(sum([1 if train_dot == "*" else 0 for train_dot in train_string]))  # Number of dots, as suggested by instruction and @590 in Piazza
            # print m," ", N, " ", ((1-t)**m) * (t**(N-m)), " ", log(((1-t)**m) * (t**(N-m)))
            # pO_of_L = ((1-t)**m) * (t**(N-m)) #m / (N*10.0)
            pO_of_L = (m / (N*10.0))#**(N-j)
            # pO_of_L = ((1-t)**m) * (t**(N-j))
            # print j, N - j
            # pO_of_L = ((j / N) ** j) * (((N-j)/N))**(N-j)
            # print pO_of_L, log(pO_of_L)
            # pO_of_L = ((1-t)**j) * (t**(N-j))


            for n, value, path, last_letter in viterbi_model:
                new_value = log((pL2_pL1_count[last_letter][train]+smoother) / float(sum(pL2_pL1_count[last_letter].values())+(total_char*smoother))) # *
                viterbi_temp.extend([[n+1, 222*log(pO_of_L)+value+new_value, path+train, train]])
            viterbi_max = sorted(viterbi_temp, key=itemgetter(1), reverse = True)[0]
            viterbi_maxes.extend([viterbi_max])
        viterbi_model = viterbi_maxes * 1
        # print sorted(viterbi_model,key=itemgetter(1), reverse=True)


    # backtrack now
    likely_path = sorted(viterbi_model,key=itemgetter(1), reverse=True)[0][2]
    
    # return likely_path
    return likely_path

print "Simple:  "+simple(train_letters, test_letters)
print "Viterbi: "+viterbi(train_letters, test_letters)
print "Final Answer:"
print viterbi(train_letters, test_letters)