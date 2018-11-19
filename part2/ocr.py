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

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print im.size
    print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print "\n".join([ r for r in train_letters['p'] ])
import pprint
# print pprint.pprint(train_letters['P'])
# print len(train_letters['p'])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print "\n".join([ r for r in test_letters[2] ])
# for each in test_letters:
# print pprint.pprint(test_letters[2])
print "".join(test_letters[0][6:8])
print "".join(train_letters['T'][6:8])

# for letter in test_letters:
#     for train in train_letters:
#         for row in letter:
#             for square in row:
#                 print square

