import time
import argparse
import pickle as pkl
import numpy as np
from tqdm import tqdm
from os import path
from collections import defaultdict


def zero():
    return 0


def epsilon():
    return 0.00000001


class HMMTagger(object):

    def __init__(self, train_path, load_vars=False, beam_search_n=None):
        self.initial_count = [0, defaultdict()]
        self.transition_count = defaultdict()
        self.emission_count = defaultdict()
        self.initial_prob = defaultdict(epsilon)
        self.transition_prob = defaultdict()
        self.emission_prob = defaultdict()
        self.states_list = []
        self.emissions_list = []
        self.beam_search_n = beam_search_n

        if load_vars:
            self.load_vars()
        else:
            self.sentences = self.load_file(train_path)
            self.get_counts()
            self.get_probabilities()
            self.save_vars()

    def load_file(self, file_name):
        with open(file_name, "r") as f:
            return f.readlines()

    def load_variable(self, var_name):
        var_path = '../data/variables/' + var_name + '.p'
        if path.exists(var_path):
            with open(var_path, 'rb') as f:
                return pkl.load(f)
        else:
            return None
    
    def save_variable(self, var, var_name):
        var_path = '../data/variables/' + var_name + '.p'
        with open(var_path, 'wb') as f:
            pkl.dump(var, f)
    
    def save_vars(self):
        self.save_variable(self.initial_count, 'initial_count')
        self.save_variable(self.transition_count, 'transition_count')
        self.save_variable(self.emission_count, 'emission_count')
        self.save_variable(self.initial_prob, 'initial_prob')
        self.save_variable(self.transition_prob, 'transition_prob')
        self.save_variable(self.emission_prob, 'emission_prob')
        self.save_variable(self.states_list, 'states_list')
        self.save_variable(self.emissions_list, 'emissions_list')
    
    def load_vars(self):
        self.initial_prob = self.load_variable('initial_prob')
        self.transition_prob = self.load_variable('transition_prob')
        self.emission_prob = self.load_variable('emission_prob')
        self.states_list = self.load_variable('states_list')
        self.emissions_list = self.load_variable('emissions_list')
    
    def normalize_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.split(' ')
        if sentence[-1] == '\n':
            sentence.pop()
        for i, word in enumerate(sentence):
            if len(word) > 1 and word[0] == '\'':
                sentence[i] = word[1:]
        return sentence
    
    def get_counts(self):
        print('Counting words and tags...')
        for sentence in tqdm(self.sentences):
            i = 0
            previous_tag = None
            for word_tag in self.normalize_sentence(sentence):
                # Get word and tag pair
                word, tag = tuple(word_tag.rsplit('/', 1))

                # Get state list
                if tag not in self.states_list:
                    self.states_list.append(tag)
                
                # Get emission list
                if word not in self.emissions_list:
                    self.emissions_list.append(word)
                
                # Count transition
                if i == 0:
                    self.initial_count[0] += 1
                    if tag in self.initial_count[1]:
                        self.initial_count[1][tag] += 1
                    else:
                        self.initial_count[1][tag] = 1
                else:
                    if previous_tag in self.transition_count:
                        self.transition_count[previous_tag][0] += 1
                        if tag in self.transition_count[previous_tag][1]:
                            self.transition_count[previous_tag][1][tag] += 1
                        else:
                            self.transition_count[previous_tag][1][tag] = 1
                    else:
                        self.transition_count[previous_tag] = \
                            [1, defaultdict(zero)]
                        self.transition_count[previous_tag][1][tag] = 1
                previous_tag = tag
                i += 1
                
                # Count emission
                if tag in self.emission_count:
                    self.emission_count[tag][0] += 1
                    if word in self.emission_count[tag][1]:
                        self.emission_count[tag][1][word] += 1
                    else:
                        self.emission_count[tag][1][word] = 1
                else:
                    self.emission_count[tag] = [1, defaultdict(zero)]
                    self.emission_count[tag][1][word] = 1
                    
    def get_probabilities(self):
        print('Getting initial probabilities...')
        n_init = self.initial_count[0]
        for tag, tag_count in tqdm(self.initial_count[1].items()):
            self.initial_prob[tag] = tag_count / n_init
        
        print('Getting transition probabilities...')
        for tag, (n_tag, next_tag_count) in tqdm(self.transition_count.items()):
            for next_tag, count in next_tag_count.items():
                if tag not in self.transition_prob:
                    self.transition_prob[tag] = defaultdict(epsilon)
                self.transition_prob[tag][next_tag] = count / n_tag

        print('Getting emission probabilities...')
        for tag, (n_tag, word_count) in tqdm(self.emission_count.items()):
            for word, count in word_count.items():
                if tag not in self.emission_prob:
                    self.emission_prob[tag] = defaultdict(epsilon)
                self.emission_prob[tag][word] = count / n_tag
                    
    def check_word(self, word, word_t):
        if word == '\'':
            word = word_t
        assert word == word_t, (word, word_t)
        return word, word_t

    def viterbi(self, emissions):
        # Init t_0
        v_path = [{}]
        for s in self.states_list:
            v_path[0][s] = {
                'prob': self.initial_prob[s] * \
                    self.emission_prob[s][emissions[0]],
                'pre_state': None
            }

        # Forward: calculate the viterbi path
        for t, e in enumerate(emissions[1:]):
            dict_t = {}
            
            # Current state
            for s in self.states_list:
                prob_state = 0.
                pre_state = self.states_list[0]
                
                # Previous state
                for pre_s in self.states_list:
                    prob = v_path[t][pre_s]['prob'] * \
                        self.transition_prob[pre_s][s]
                        
                    # Track the max prob and the previous state
                    if prob > prob_state :
                        prob_state = prob
                        pre_state = pre_s
                        
                # Times the emission prob
                prob_state *= self.emission_prob[s][e]
                dict_t[s] = {'prob': prob_state, 'pre_state': pre_state}
                
            # Add current state to viterbi path
            v_path.append(dict_t)

        # Choose the final state
        state_selected = np.random.choice(list(v_path[-1].keys()))
        max_prob = v_path[-1][state_selected]['prob']
        for s, value in v_path[-1].items():
            if value['prob'] > max_prob:
                max_prob = value['prob']
                state_selected = s

        # Backward: trace back the state path
        back_path = [state_selected]
        for v_t in v_path[::-1]:
            state_selected = v_t[state_selected]['pre_state']
            back_path.append(state_selected)

        # The most possible state path
        states_path = back_path[-2::-1]

        return states_path, max_prob

    def viterbi_beam(self, emissions):
        # Init t_0
        v_path = [{}]
        beam_prob = []
        beam_st = []
        for s in self.states_list:
            prob_state = self.initial_prob[s] * \
                self.emission_prob[s][emissions[0]]
            v_path[0][s] = {'prob': prob_state, 'pre_state': None}
            beam_st.append(s)
            beam_prob.append(prob_state)
        # Generate beam states for next round
        beam_st = np.array(beam_st)
        beam_prob = np.array(beam_prob)
        beam_states = \
            beam_st[beam_prob.argsort()[-self.beam_search_n:]]
                
        # Forward: calculate the viterbi path
        for t, e in enumerate(emissions[1:]):
            dict_t = {}
            beam_prob = []
            beam_st = []
            
            # Current state
            for s in self.states_list:
                prob_state = 0.
                pre_state = self.states_list[0]
                
                # Previous state
                for pre_s in beam_states:
                    prob = v_path[t][pre_s]['prob'] * \
                        self.transition_prob[pre_s][s]
                        
                    # Track the max prob and the previous state
                    if prob > prob_state :
                        prob_state = prob
                        pre_state = pre_s
                        
                # Times the emission prob
                prob_state *= self.emission_prob[s][e]
                dict_t[s] = {'prob': prob_state, 'pre_state': pre_state}
                
                # Calculate beam
                beam_st.append(s)
                beam_prob.append(prob_state)
                
            # Add current state to viterbi path
            v_path.append(dict_t)
            
            # Generate beam states for next round
            beam_st = np.array(beam_st)
            beam_prob = np.array(beam_prob)
            beam_states = \
                beam_st[beam_prob.argsort()[-self.beam_search_n:]]
                
            # print(dict_t)
            # print(beam_prob)
            # print(beam_states)

        # Choose the final state
        state_selected = np.random.choice(list(v_path[-1].keys()))
        max_prob = v_path[-1][state_selected]['prob']
        for s, value in v_path[-1].items():
            if value['prob'] > max_prob:
                max_prob = value['prob']
                state_selected = s

        # Backward: trace back the state path
        back_path = [state_selected]
        for v_t in v_path[::-1]:
            state_selected = v_t[state_selected]['pre_state']
            back_path.append(state_selected)

        # The most possible state path
        states_path = back_path[-2::-1]

        return states_path, max_prob
    
    def main(self, test_path, test_tagged_path, test_num=None):
        start_time = time.time()
        accuracy = []
        if test_num:
            test_sentences = self.load_file(test_path)[:test_num]
            test_tagged_sentences = self.load_file(test_tagged_path)[:test_num]
        else:
            test_sentences = self.load_file(test_path)
            test_tagged_sentences = self.load_file(test_tagged_path)
            
        print('Testing...')
        for sentence, tagged_sentence in \
                tqdm(zip(test_sentences, test_tagged_sentences),
                     total=len(test_sentences)):
                    
            # Normalize sentence
            words_test = self.normalize_sentence(sentence)
            tagged_words_test = self.normalize_sentence(tagged_sentence)
            assert len(words_test) == len(tagged_words_test)
            
            # Get emissions and true states for a sentence
            emissions = []
            states_true = []
            for word, word_tag in zip(words_test, tagged_words_test):
                word_t, tag = word_tag.rsplit('/', 1)
                word, word_t = self.check_word(word, word_t)
                emissions.append(word)
                states_true.append(tag)
            
            # Get predicted states
            if self.beam_search_n:
                states_pred, _ = self.viterbi_beam(emissions)
            else:
                states_pred, _ = self.viterbi(emissions)
            
            # Count accuracy
            for tag_pred, tag_true in zip(states_pred, states_true):
                accuracy.append(1 if tag_pred == tag_true else 0)
        
        # Compute the final accuracy
        accuracy = np.array(accuracy).mean()
        print('Accuracy: {:.4f}%'.format(accuracy * 100))
        print('Runtime: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--beam_size', type=int, metavar='')
    args = parser.parse_args()

    HMMTagger(
        train_path='../data/brown.train.tagged.txt',
        load_vars=True,
        beam_search_n=args.beam_size
        ).main(
        test_path='../data/brown.test.raw.txt',
        test_tagged_path='../data/brown.test.tagged.txt'
        )
