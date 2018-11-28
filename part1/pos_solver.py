###################################
# CS B551 Fall 2018, Assignment #3
#
# Completed by:
# Derrick Eckardt
# derrick@iu.edu
# (Based on skeleton code by D. Crandall)
#
# Completed on November 25, 2018
#
################################################################################
################################################################################
# Part 1 - Part of Speech Tagging
################################################################################
################################################################################
#
# Overall, here are the results of the various means of doing part of speech
# tagging.  One thing to note is that I ran the MCMC at various amounts of Gibbs
# samples, which are noted in the last column.  In addition, some of those were
# set to ignore a certain amount of gibbs samples (aka burn-in) for determining
# the answer.
#
# ==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct: 
#   0. Ground truth:      100.00%              100.00%
#          1. Simple:       93.92%               47.45%
#             2. HMM:       94.36%               50.75%
#         3. Complex:       93.50%               46.75%  Gibbs = 3
#         3. Complex:       94.02%               48.35%  Gibbs = 5
#         3. Complex:       94.52%               51.15%  Gibbs = 10
#         3. Complex:       94.48%               51.15%  Gibbs = 25
#         3. Complex:       94.55%               51.45%  Gibbs = 50
#         3. Complex:       94.65%               52.05%  Gibbs = 100 (current)
#         3. Complex:       94.67%               52.10%  Gibbs = 100, burn =50
#         3. Complex:       94.63%               52.15%  Gibbs = 500
#         3. Complex:       94.68%               52.35%  Gibbs = 500, burn = 250
#         3. Complex:       94.67%               52.25%  Gibbs = 1000
#         3. Complex:       94.67%               52.30%  Gibbs = 1000 burn = 500
#         3. Complex:       94.65%               52.30%  Gibbs = 2000

# Ultimately, I went with 100 Gibbs samples as my default MCMC setting, since at
# that level it looked like there was no noticable improvement in performance. 
# As computer scientists, we have to trade-off performance against resources,
# this is just one of many.  I'll get back to the rest of the MCMC decision
# later, I just wanted to put the results up-front.
#
################################################################################
# Training the Model
################################################################################
#
# Loading the training data was probably one of the easier parts of the process,
# but it was not without it's decisions.  While loading the data in, I had to
# arrange the information so that I could easily and quickly get that 
# information, with a dictionary.  But, dictionaries break with unknown keys.
# One of the most useful things for this is that since there were millions of
# training words and millions more that could be in a test set, I needed to 
# compensate for ones that were not in the training model, and i did not want to
# have to check to see if I had previously seen it.  Luckily, python does this
# will with Counter(), which is a version of a dictionary for counts, which if
# asked for an unknown key, will return 0, instead of a keyError and break my
# code.
# 
# Once I settled on that, I put the needed variables inside of my def __init__
# or my def train().  This resulted in finding the variables of P(Wi | Si),
# P(S1), P(Si), P (Si+1 | S), P(Si+2| Si+1, Si)
#
# One item to also note is that I added a constant c in there that would be used
# for smoothing when unknown words or words used in with a new POS.  While
# initially set to 1, it later gets changed to 1 / total words.
#
################################################################################
# Calculating Posteriors and Simple Model
################################################################################
#
# The nice thing about this, was once I figured out how to find the posteriors,
# the simple model was more or else completed, since the model shown in Figure
# 1b represents the simpliest bayes net we could have, where the posteriors is
# is what we would be looking for in each word.
#
# The first thing we run into here is the issue with words in the test set that
# our model has no previous knowledge of.  In order to account for this, I added
# a constant c that added a tiny amount to the numerator and tiny amount to the
# denominator.  This allows for our model to consider a word being used in all
# 12 different Parts of Speech, without breaking our program.
#
# Once the values for the posteriors are calculated for the word with each of
# the 12 parts of speech, the most likely word is simply the word with highest
# value.  Then
#
# One trade-off made on the simple model was whether I should use P(S1) or P(Si)
# for the first word.  Since the simple model does not imply any knowledge
# of the other words, and word order really does not matter, and I wanted my
# observation (w) to be able to pull from as large a dataset as possible, so I 
# went with P (Si) for the simple model.  However, for Viterbi and the MCMC, 
# where the first word multiplies into other words, it was more important to use
# P(S1) since word order matters more there.
#
# I also, then set-up the posterior models for the Viterbi and Complex, which
# involved multiplying in other factors.  Since those values have already been
# calculated during training, it was not that difficult. The hardest part was 
# making sure it managed to account for the first, second, and second-to-last, 
# and last words of sentence, since they didn't always have the same factors
# weighed in, which is more complex than the 
#
# I was proud that I recognized that the first word of the HMM and Complex
# models were the same as a one-word Simple, and the second word of the complex
# was the same as a two-word HMM.  That allowed for some code simplification.
#
# At this level, i noticed a limitation of the design decision to use Counter(),
# which I was glowing about before.  Since I liked having Counter() since it
# handled unknown keys well, it also meant that I couldn't have float numbers in
# my dictionary.  Which meant I had to calculate the probabilities of each
# value on the fly, resulting in additional calculations, which likely slow down
# my code.  I still think this is better since the way it handles unknown keys;
# however, i would welcome finding another method that allowed me to simplify
# my code
#
################################################################################
# Hidden Markov Model solved via Viterbi Model
################################################################################
#
# When, I finished implementing it, I was amazed at the simplicity and how
# efficiently it worked.  As an alum of the USC Viterbi School of Engineering,
# I was incredibly proud to be able to finally get an understanding of it.
#
# 
#
################################################################################
# Complex Model solved via Markov Chain Monte Carlo
################################################################################
#
#
##
# Calculating Emissions
#
# Using Viterbi
#
# MCMC iterations - One of the most 
#


#


import random
from math import log
from collections import defaultdict, Counter
from operator import itemgetter
from random import random

# Things to do to make it better
# 1. Turn the counters into decimals values to make reading the formulas easier instead of calculating them each time.
# 2. Make sure s1 is being used as the first one in simple, viterbi, and mcmc
# 4. Enable Voting ability?

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
        self.c = 1 # used for smoothing, with training, will change it to 1/ total words.
        # when I used one, it tended to favor x words which were unknown foreign words. which would then impact other words, causing others to be misclassified.
        self.j = 1 # for gibbs
                 
    
    def posterior(self, model, sentence, label):
        if model == "Simple" or "Voting":
            interim = 0
            for word,part in zip(sentence,label):
                # Added constance c to numerator and denominator to smooth for unknown words
                interim += log(((self.p_wi_si[part][word]+self.c) / float(self.p_si[part]+self.c)) * ((self.p_si[part] + self.c)/float(self.unique_words+self.c)))
            return interim
        elif model == "HMM":
            # First word in the sentence is the same as the simple model
            interim = Solver.posterior(self,"Simple",[sentence[0]],[label[0]])
            # rest of elements
            if len(sentence) > 1:
                for word, part, i in zip(sentence[1:],label[1:], range(1,len(sentence))):
                    interim += log(((self.p_wi_si[part][word]+self.c) / float(self.p_si[part]+self.c))*((self.p_si1_si[label[i-1]][part] + 1)/float(self.p_si[label[i-1]]+self.c)))
            return interim
        elif model == "Complex":
            # First two words in the sentence are the same as the hmm model
            interim = Solver.posterior(self,"HMM",sentence[0:2],label[0:2])
            if len(sentence) > 2:
                for word, part, i in zip(sentence[2:],label[2:], range(2,len(sentence))):
                    interim += log(((self.p_wi_si[part][word]+self.c) / float(self.p_si[part]+self.c))*((self.p_si2_si1_si[label[i-2]][label[i-1]][part] + 1)/float(sum(self.p_si2_si1_si[label[i-2]][label[i-1]].values())+self.c)))
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
        self.c = 1 / float(self.unique_words)
        
    
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

    def coin_flip(self, ratios):
        flip = random()
        check = 0
        for pos, value in ratios:
            check += value
            if flip < check:
                return pos
                
    def complex_mcmc(self, sentence):
        # gibbs sampling
        gibbs_samples = {}
        for i in range(len(sentence)):
            gibbs_samples[i] = Counter()
        
        ns =[ "noun" ] * len(sentence)
        for g in range(100):
            for i in range(len(sentence)):
                ratios =[]
                running_total = 0.00
                for pos in self.pos:
                    # First Five are for len(sentence) >= 3
                    pos_value = (self.p_wi_si[pos][sentence[i]] + self.c) / float(self.p_si[pos]+self.c)
                    if i >= 2 and i < len(sentence) -2 and len(sentence) >2 :
                        pos_value *= (
                            (self.p_si2_si1_si[ns[i-2]][ns[i-1]][pos]+self.c) / float(sum(self.p_si2_si1_si[ns[i-2]][ns[i-1]].values())+self.c)*
                            (self.p_si2_si1_si[ns[i-1]][pos][ns[i+1]]+self.c) / float(sum(self.p_si2_si1_si[ns[i-1]][pos].values())+self.c)*
                            (self.p_si2_si1_si[pos][ns[i+1]][ns[i+2]]+self.c) / float(sum(self.p_si2_si1_si[pos][ns[i+1]].values())+self.c)
                            )
                    elif i == 0 and i < len(sentence) - 2 and len(sentence) >2 :
                        pos_value *= (
                            (self.p_s1[pos] + self.c) / float(self.unique_lines +self.c) * 
                            (self.p_si1_si[pos][ns[i+1]] + self.c) / float(self.p_si[pos] + self.c) *
                            (self.p_si2_si1_si[pos][ns[i+1]][ns[i+2]]+self.c) / float(sum(self.p_si2_si1_si[pos][ns[i+1]].values())+self.c)
                            )
                    elif i == 1 and i < len(sentence) - 2 and len(sentence) >2:
                        pos_value *= (
                            (self.p_si1_si[ns[i-1]][pos] + self.c) / float(self.p_si[ns[i-1]] + self.c) *
                            (self.p_si2_si1_si[ns[i-1]][pos][ns[i+1]]+self.c) / float(sum(self.p_si2_si1_si[ns[i-1]][pos].values())+self.c)*
                            (self.p_si2_si1_si[pos][ns[i+1]][ns[i+1]]+self.c) / float(sum(self.p_si2_si1_si[pos][ns[i+1]].values())+self.c)
                            )
                    elif i == len(sentence) - 2 and len(sentence) >2: # (second to last word)
                        pos_value *= (
                            (self.p_si2_si1_si[ns[i-2]][ns[i-1]][pos]+self.c) / float(sum(self.p_si2_si1_si[ns[i-2]][ns[i-1]].values())+self.c)*
                            (self.p_si2_si1_si[ns[i-1]][pos][ns[i+1]]+self.c) / float(sum(self.p_si2_si1_si[ns[i-1]][pos].values())+self.c)
                            )
                    elif i == len(sentence) - 1 and len(sentence) >2: # (last word)
                        pos_value *= (
                            (self.p_si2_si1_si[ns[i-2]][ns[i-1]][pos]+self.c) / float(sum(self.p_si2_si1_si[ns[i-2]][ns[i-1]].values())+self.c)
                            )
                    elif i == 0 and len(sentence) == 2:
                        pos_value *= (
                            (self.p_s1[pos] + self.c) / float(self.unique_lines +self.c) * 
                            (self.p_si1_si[pos][ns[i+1]] + self.c) / float(self.p_si[pos] + self.c)
                            )
                    elif i == 1 and len(sentence) == 2: # last word
                        pos_value *= (
                            (self.p_si1_si[ns[i-1]][pos] + self.c) / float(self.p_si[ns[i-1]] + self.c)
                            )
                    # For one word sentences, it's already the part that was included                        
                    # inetntionally used this one, since if there is a one-word sentence, i figured it could be anything.  more of a quirk of the data than anything.
                    ratios.extend([[pos, pos_value]])
                    running_total += pos_value
                ratios = sorted([[each, value/running_total] for each, value in ratios], key=itemgetter(1), reverse = True)

                # Flip the coin and assign new point of speech
                ns[i] = self.coin_flip(ratios)

                if g >= 0: # For burn-in, disregard the first n amount.  currently at 0, meaning we use all data points
                    gibbs_samples[i][ns[i]] += 1
            
        return [gibbs_samples[i].most_common(1)[0][0] for i in range(len(sentence))]

    def hmm_viterbi(self, sentence):
        viterbi_model = []
        # first word in sentence
        for pos in self.pos:
            # sublist to the form of [position_no, value, path, pos], only with the words that appear
            # one added to both in the event it is a new word or a word being used in a new form.
            viterbi_model.extend([[0,(self.p_s1[pos]+self.c)/float(self.unique_lines+self.c)*(self.p_wi_si[pos][sentence[0]]+self.c)/float(self.p_si[pos]+self.c), pos, pos]])
        
        # Rest of sentence 
        if len(sentence) > 1:
            for word in sentence[1:]:
                viterbi_maxes =[]
                for pos in self.pos:
                    viterbi_temp = []
                    for n, value, path, last_pos in viterbi_model:
                        new_value = value*(self.p_si1_si[last_pos][pos]+self.c)/float(self.p_si[last_pos]+self.c)*(self.p_wi_si[pos][word]+self.c)/float(self.p_si[pos]+self.c)
                        viterbi_temp.extend([[n+1, new_value, path+" "+pos, pos]])
                    viterbi_max = sorted(viterbi_temp, key=itemgetter(1), reverse = True)[0]
                    viterbi_maxes.extend([viterbi_max])
                viterbi_model = viterbi_maxes * 1

        # backtrack now
        likely_path = sorted(viterbi_model,key=itemgetter(1), reverse=True)[0][2].split()
 
        return likely_path
        
    def voting(self,sentence):
        votes = [self.simplified(sentence), self.complex_mcmc(sentence), self.hmm_viterbi(sentence)]
        output_votes = []
        for i in range(len(votes[0])):
            vote_counter = Counter([votes[j][i] for j in range(len(votes))])
            output_votes.extend([vote_counter.most_common(1)[0][0]])
        return output_votes
    
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
        elif model == "Voting":
            return self.voting(sentence)
        else:
            print("Unknown algo!")

