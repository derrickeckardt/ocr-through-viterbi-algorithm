#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
###################################
# CS B551 Fall 2018, Assignment #3
#
# Completed by:
# Derrick Eckardt
# derrick@iu.edu
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

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
                example += " "+ word
            else:
                example += word
        exemplars.extend([(example+" ")])
    return exemplars


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
training_text = read_data(train_txt_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!

# Train letters
# p(l), p(l2 | l1)
pL_count = Counter()
for line in training_text:
    for letter in line:
        pL_count[letter] += 1

total_char = float(sum(pL_count.values()))

# For unknown characters    
smoother = 1 / total_char


# Simple method
# Compare each test letter to the train letter.  The one with the highest percentage of similar
# points is the winner
def simple(train_letters, test_letters):
    # cycle through each test letter
    simple_text = ""
    for test in test_letters:
        test_scores = []
        # Sort through each letter in the train set, a
        test_string = "".join(test)
        for train in train_letters:
            train_string = "".join(train_letters[train])
            t = 0.3 # tuning parameter
            N = float(len(train_string)) # Number of bits
            m = sum([1 if train_dot == test_dot else 0 for train_dot, test_dot in zip(train_string, test_string)]) # bits in common
            pO_of_L = ((1-t)**m) * (t**(N-m))
            # pO_of_L = sum([1 if train_dot == test_dot else 0 for train_dot, test_dot in zip(train_string, test_string)]) / float(len(train_string))
            pL = (pL_count[train] + smoother) / (total_char + smoother)
            test_scores.extend([[train, pO_of_L * pL]])
        # print pprint(sorted(test_scores, key=itemgetter(1), reverse = True))
        simple_text += sorted(test_scores, key=itemgetter(1), reverse = True)[0][0]

    return simple_text

print "Simple: "+simple(train_letters, test_letters)

