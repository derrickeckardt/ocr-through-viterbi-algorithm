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
# Part 2 - Optical Character Recognition (OCR)
################################################################################
################################################################################
#
# This was really fun.  This is something that I've seen in real life, and now I
# understand what it trying to do.  While our version was simplified, I have
# a deeper appreciation for what actually needs to happen in order to complete
# this kind of work.
#
# My program takes about 10 to 12 seconds to run per test case.  For the 20 new
# test cases that Prof. Crandall mentioned in Piazza @641, I would expect my
# program to tkae about 4 minutes or less to complete.
# 
################################################################################
# Defining the HMM
################################################################################
#
# The HMM I defined for this example, is similar to the one that was used in
# Figure 1a of the assignment.  Drawn with text it would be:
#
# L1 -> O1, L1 -> L2, L2 -> O2, L2 -> L3.....Ln -> On, Ln-1 -> Ln
#
# Where L is the L letter state, and O is the observation that our program sees.
# Then when the data is trained, I had to find P(Li) and P(Li+1| Li) from a
# training text data set (I used bc.train from part 1).  Later I would have to 
# estimate P(Oi | Li) based on the training letter images. See below for more on
# the details of that struggle.
#
################################################################################
# Reusing Code - Simple and Viterbi
################################################################################
# 
# For my two main algorithms, it was really nice to be able to reuse my code
# from Part 1.  I decided to bring the code in directly to it (instead of an
# import), because the functions had to be changed for the OCR cases, and for
# simplicity, it made sense to have them there.  Otherwise, I would have to
# modify the other code so that it worked for everything, which would be
# significatly more difficult in the long run.  If I was a programmer that used
# viterbi everyday, I would find a way to make a more all-purpose algorithm, 
# that took set inputs.  For this purpose, the benefits would not be reaped in
# time.
#
# The biggest decision on adapting my Part 1 functions was how to handle the 
# emission probabilities, which ended up being a serious issue, since I could
# not find emission probabilities that functioned terribly well.  I would get
# lines of just blanks or punctuation for my results, until I really figured out
# how to address emission probabilities. (see next)
# 
###############################################################################
# Emission Probabilities
################################################################################
# 
# Wow.  This was by far one of the most difficult items that i dealt with.  I 
# probably spent about 10 hours trying to get my emission probabilities to
# work in a reasonable fashion.
# 
# If you look into my code, you will see lots of pO_of_L commented out of my
# code.  These represent the many different emission probabilities that I tried.
# The biggest issue I faced with that is that many of the methods preferred
# absolute pixel matches, such as the one suggested in the assignment prompt.
# For the punctation and blank spaces, those would have very high hit ratios,
# despite, it being mostly not useful information. Results were akin like:
#
#  "                        "  or "!!!!!!!!!!!!!!!!!" or "?    '       !"
#
# Those don't remotely resemble the correct text.  So, I needed something new.
# Ultimately, a Piazza post, @615, gave me a method that made sense on how to 
# handle it, without using the dampening factor.  In restrospect, I did have a
# dampening factor, it just looks different.  For the viterbi, What I did is had
# matching  "*" pixels  worth 10, while matching a " " would be worth 1.25, and
# noisy pixels would be worth 0.  Then, I would devide them by the total number 
# of pixels times 10, and then multiply the log of that by 250.  All the numbers 
# were found by trial error of raising and lowering them. 10, 1.25, 0, 10, and 
# 250 were the values that seemed to get the most consistent results.  For
# the simple model, "*" is worth 1, a blank spot is worth 0.05, 0 for noise, 
# still divide by number of pixels times 10, and then do not multiply the log by
# anything (or multiply by 1).  This was more trial and error, and reinforced
#
# In the future, I would improve on this by having a script that tests all
# of those parameters to find which one.  Given another week, I would do that
# next.  Of course, that runs the risk of overfitting, however, I would expect
# to get a range of values that would provide an idea of what will get me in the
# top range of what to expect.
#
################################################################################
# Final Answer
################################################################################
# 
# Ultimately, I had my algorithm use the viterbi algorithm as the final answer
# this was function that it was more accurate for the noisy test cases, which
# gave the same ones too many problems.  I did find that the sample one seemed
# to work better on the clearer examples, and in two instances, it was actually
# better than viterbi
# 
################################################################################
# Use a Better and/or Legal Dictionary -- Opportunity for Improvement
################################################################################
# 
# The next way to improve this would be to take an-in-depth dictionary, and 
# compare provisional results against it, and throw out the options  (or downgrade
# tehir score) that use words that do not exist in a robust dictionary.  That
# would clean up additional mistakes.
#
# Since this was for legal documents, a training text document for legal text
# would also be helpful, as it would pick up the pecular traits of the images,
# such as latin words, lots of punctations, and abbreviations.



from PIL import Image, ImageDraw, ImageFont

import sys
from math import log
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
        
        viterbi_model.extend([[0,250*log(pO_of_L), train, train]])  # + log((pL1_count[train]+pL1_smoother)/float(total_pL1+pL1_smoother))

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
                viterbi_temp.extend([[n+1, 250*log(pO_of_L)+value+new_value, path+train, train]])
            viterbi_max = sorted(viterbi_temp, key=itemgetter(1), reverse = True)[0]
            viterbi_maxes.extend([viterbi_max])
        viterbi_model = viterbi_maxes * 1

    # backtrack now
    likely_path = sorted(viterbi_model,key=itemgetter(1), reverse=True)[0][2]
    
    # return likely_path
    return likely_path

# Output Results
print "Simple:  "+simple(train_letters, test_letters)
viterbi_result = viterbi(train_letters, test_letters)
print "Viterbi: "+viterbi_result
print "Final Answer:"
print viterbi_result