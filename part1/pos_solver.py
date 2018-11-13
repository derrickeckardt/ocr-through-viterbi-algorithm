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

import random
from math import log
from collections import defaultdict, Counter
from operator import itemgetter
from pprint import pprint
from time import sleep


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

# Things to do to make it better
# Turn the counters into decimals values to make reading the formulas easier instead of calculating them each time.
# For viterbi, the smoothing favors x for foreign words for unknown words... 

class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def __init__(self):
        self.p_s1 = Counter()
        self.p_si = Counter()
        self.p_wi = Counter()
        self.unique_words = 0
        self.unique_lines = 0
        self.p_si1_si = {} # Previous state
        self.p_si2_si1_si = {} # Two states agp
        self.p_wi_si = {}
        self.pos = ["adj","adv","adp","conj","det","noun","num","pron","prt","verb","x","."]
        for pos in self.pos:
             self.p_wi_si[pos] = Counter()
             self.p_si1_si[pos] = Counter()
             self.p_si2_si1_si[pos] = {}
             for part in self.pos:
                 self.p_si2_si1_si[pos][part] = Counter()
                 
    
    def posterior(self, model, sentence, label):
        if model == "Simple":
            interim = 0
            for word,part in zip(sentence,label):
                # Added plus one to numerator and denominator to smooth for unknown words
                interim += log(((self.p_wi_si[part][word]+1) / float(self.p_si[part]+1)) * ((self.p_si[part] + 1)/float(self.unique_words+1)))
            return interim
        elif model == "HMM":
            # First word in the sentence is the same as the simple model
            interim = Solver.posterior(self,"Simple",[sentence[0]],[label[0]])
            # rest of elements
            if len(sentence) > 1:
                for word, part, i in zip(sentence[1:],label[1:], range(1,len(sentence))):
                    interim += log(((self.p_wi_si[part][word]+1) / float(self.p_si[part]+1))*((self.p_si1_si[label[i-1]][part] + 1)/float(self.p_si[label[i-1]]+1)))
            return interim
        elif model == "Complex":
            # First two words in the sentence are the same as the hmm model
            interim = Solver.posterior(self,"HMM",sentence[0:2],label[0:2])
            if len(sentence) > 2:
                for word, part, i in zip(sentence[2:],label[2:], range(2,len(sentence))):
                    interim += log(((self.p_wi_si[part][word]+1) / float(self.p_si[part]+1))*((self.p_si2_si1_si[label[i-2]][label[i-1]][part] + 1)/float(sum(self.p_si2_si1_si[label[i-2]][label[i-1]].values())+1)))
            return interim
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        # Estimate the probabilities of P(S1), P(Si+1|Si), P(Wi|Si)
        # List of different parts of speech
        for each in data:
            self.p_s1[each[1][0]] += 1
            last_pos = ""
            two_ago_pos =""
            for word, part in zip(each[0],each[1]):
                self.p_si[part] += 1
                self.p_wi[word] += 1
                if word in self.p_wi_si[part]:
                    self.p_wi_si[part][word] += 1
                else:
                    self.p_wi_si[part][word] = 1
                if two_ago_pos != "":
                    if part in self.p_si2_si1_si[two_ago_pos][last_pos]:
                        self.p_si2_si1_si[two_ago_pos][last_pos][part] += 1
                    else:
                        self.p_si2_si1_si[two_ago_pos][last_pos][part] = 1
                if last_pos != "":
                    if part in self.p_si1_si[last_pos]:
                        self.p_si1_si[last_pos][part] += 1
                    else:
                        self.p_si1_si[last_pos][part] = 1
                    two_ago_pos = last_pos 
                last_pos = part

        # Totals for use later
        self.unique_words = sum(self.p_si.values())
        self.unique_lines = sum(self.p_s1.values())
        # print self.p_s1
        # print self.p_si1_si
        # print self.p_si2_si1_si
        # return self.p_s1, self.p_si1_si, self.p_wi_si
    
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        simplified_model = []
        for word in sentence:
            pos_values = []
            for pos in self.pos:
                interim = (self.p_wi_si[pos][word] / float(self.p_si[pos])) * ((self.p_si[pos])/float(self.unique_words))
                pos_values.extend([[interim, pos]])
            pos_values = sorted(pos_values, key=itemgetter(0), reverse=True)
            # Note uses most popular pos in case word is not in training corpus
            most_likely = pos_values[0][1] if pos_values[0][0] != 0 else self.p_si.most_common(1)[0][0]
            simplified_model.extend([most_likely])
        return simplified_model 

    def complex_mcmc(self, sentence):
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        viterbi_model = []
        # first word in sentence
        for pos in self.pos:
            # sublist to the form of [position_no, value, path, pos], only with the words that appear
            # one added to both in the event it is a new word or a word being used in a new form.
            viterbi_model.extend([[0,(self.p_s1[pos]+1)/float(self.unique_lines+1)*(self.p_wi_si[pos][sentence[0]]+1)/float(self.p_si[pos]+1), pos, pos]])
        
        # Rest of sentence 
        if len(sentence) > 1:
            for word in sentence[1:]:  # Remove 2 later on once its running
                viterbi_maxes =[]
                for pos in self.pos:
                    viterbi_temp = []
                    for n, value, path, last_pos in viterbi_model:
                        new_value = value*(self.p_si1_si[last_pos][pos]+1)/float(self.p_si[last_pos]+1)*(self.p_wi_si[pos][word]+1)/float(self.p_si[pos]+1)
                        viterbi_temp.extend([[n+1, new_value, path+" "+pos, pos]])
                    viterbi_max = sorted(viterbi_temp, key=itemgetter(1), reverse = True)[0]
                    viterbi_maxes.extend([viterbi_max])
                viterbi_model = viterbi_maxes * 1

        # backtrack now
        likely_path = sorted(viterbi_model,key=itemgetter(1), reverse=True)[0][2].split()

        # print len(viterbi_temp)
        # print sorted(viterbi_maxes, key=itemgetter(3), reverse = True)
        # print viterbi_max

    
        return likely_path
    
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

