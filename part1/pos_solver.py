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
from collections import Counter

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
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
        pos = ["ADJ","ADV","ADP","CONJ","DET","NOUN","NUM","PRON","PRT","VERB","X","."]
        p_s1 = Counter()
        p_s2_s1 = Counter()
        p_wi_si = Counter()
        for i, each in zip(range(len(data)),data):
            if i == 0:
                print each
                print each[0]
                print each[1]
                print each[1][0]
            p_s1[each[1][0]] += 1
            last_pos = ""
            for word, part in zip(each[0],each[1]):
                p_wi_si[part+"--"+word] += 1
                if last_pos != "":
                    p_s2_s1[part+"--"+last_pos] += 1
                last_pos = part

        print p_s1
        print sum(p_s1.values())
        # print p_wi_si
        print sum(p_wi_si.values())
        print p_s2_s1

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        return [ "noun" ] * len(sentence)

    def complex_mcmc(self, sentence):
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
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

