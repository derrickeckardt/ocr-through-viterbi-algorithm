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
import math
from collections import defaultdict, Counter
from operator import itemgetter
from pprint import pprint
from time import sleep


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

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
        self.p_si2_si = {} # Two states agp
        self.p_wi_si = {}
        self.pos = ["adj","adv","adp","conj","det","noun","num","pron","prt","verb","x","."]
        for pos in self.pos:
             self.p_wi_si[pos] = Counter()
             self.p_si1_si[pos] = Counter()
             self.p_si2_si[pos] = Counter()
    
    def posterior(self, model, sentence, label):
        if model == "Simple":
            interim = 0
            for word,part in zip(sentence,label):
# Need to devide by denominator
                word_interim = (self.p_wi_si[part][word] / float(self.p_si[part])) * ((self.p_si[part])/float(self.unique_words))
# Double check the use of math.log(1/float(self,unique words)
                interim += math.log(word_interim) if word_interim != 0 else math.log(1/float(self.unique_words)) 
            return interim
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return -999
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
                    if part in self.p_si2_si[last_pos]:
                        self.p_si2_si[last_pos][part] += 1
                    else:
                        self.p_si2_si[last_pos][part] = 1
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
        # print self.p_si2_si
        # return self.p_s1, self.p_si1_si, self.p_wi_si
    
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        print sentence
        simplified_model = []
        for word in sentence:
            pos_values = []
            for pos in self.pos:
# Need to divide by denominator
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
            # sublist to the form of [position_no, p(si), came_from_pos, pos], only with the words that appear
            if self.p_wi_si[pos][sentence[0]] != 0:
                viterbi_model.extend([[0,self.p_s1[pos]/float(self.unique_lines)*self.p_wi_si[pos][sentence[0]]/float(self.p_si[pos]), "self", pos]])
        # Double check in case there are no values, will generate with each pos as if there was exactly one occurance
        if len(viterbi_model) == 0:
            for pos in self.pos:
                viterbi_model.extend([[0,self.p_s1[pos]/float(self.unique_lines)*1/self.p_si[pos], "self", pos]])
            
        # second word to last word in sentence
        viterbi_temp = viterbi_model*1
        for word, i in zip(sentence[1:],range(1,len(sentence))):
            viterbi_last = viterbi_temp * 1
            viterbi_temp = []
            for n, value, predecessor, old_pos in viterbi_last:
                for pos in self.pos:
                    if self.p_wi_si[pos][word] !=0 and self.p_si1_si[old_pos][pos] !=0:
                        #print n, value, predecessor, old_pos
                        #sleep(5)
                        viterbi_temp.extend([[i,self.p_si1_si[old_pos][pos]/float(self.p_si[old_pos])*self.p_wi_si[pos][word]/float(self.p_si[pos]), old_pos, pos]])
                if len(viterbi_temp) == 0:
                    if self.p_si1_si[pos][word] !=0:
                        viterbi_temp.extend([[i,1/float(self.p_si[old_pos])*self.p_wi_si[pos][word]/float(self.p_si[pos]), old_pos, pos]])
                    else:
                        viterbi_temp.extend([[i,1/float(self.p_si[old_pos])*1/float(self.p_si[pos]), old_pos, pos]])
                        
            viterbi_model.extend(viterbi_temp)
            print viterbi_temp
            sleep(5)
        
        
        # backtrack now
        if len(viterbi_model) > 40:
            print viterbi_model[-40:]
        else:
            print viterbi_model
            
        return [ "noun" ] * len(sentence)


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

