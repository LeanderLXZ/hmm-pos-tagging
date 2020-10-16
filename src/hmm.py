import numpy as np
from collections import defaultdict


class HMMTagger(object):

    def __init__(self, train_path):
        
        self.EPSILON = 0.00000001

        self.initial_count = [0, defaultdict()]
        self.transition_count = defaultdict()
        self.emission_count = defaultdict()
        self.initial_prob = defaultdict(lambda: self.EPSILON)
        self.transition_prob = defaultdict(lambda: self.EPSILON)
        self.emission_prob = defaultdict(lambda: self.EPSILON)
        self.state_list = []
        self.emission_list = []

        self.sentences = self.load_file(train_path)
        self.get_counts()
        self.get_probabilities()

    def load_file(self, file_name):
        with open(file_name, "r") as f:
            return f.readlines()
    
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
        
        for sentence in self.sentences:
            i = 0
            previous_tag = None
            for word_tag in self.normalize_sentence(sentence):
                # Get word and tag pair
                word, tag = tuple(word_tag.rsplit('/', 1))

                # Get state list
                if tag not in self.state_list:
                    self.state_list.append(tag)
                
                # Get emission list
                if word not in self.emission_list:
                    self.emission_list.append(word)
                
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
                        if tag in self.transition_count[previous_tag]:
                            self.transition_count[previous_tag][1][tag] += 1
                        else:
                            self.transition_count[previous_tag][1][tag] = 1
                    else:
                        self.transition_count[previous_tag] = \
                            [1, defaultdict(lambda: 0)]
                        self.transition_count[previous_tag][1][tag] = 1
                previous_tag = tag
                i += 1
                
                # Count emission
                if tag in self.emission_count:
                    self.emission_count[tag][0] += 1
                    if word in self.emission_count[tag]:
                        self.emission_count[tag][1][word] += 1
                    else:
                        self.emission_count[tag][1][word] = 1
                else:
                    self.emission_count[tag] = [1, defaultdict(lambda: 0)]
                    self.emission_count[tag][1][word] = 1
                    
    def get_probabilities(self):
        n_init = self.initial_count[0]
        for tag, tag_count in self.initial_count[1].items():
            self.initial_prob[tag] = tag_count / n_init
        
        for tag, (n_tag, next_tag_count) in self.transition_count.items():
            for next_tag, count in next_tag_count.items():
                self.transition_prob[tag][next_tag] = count / n_tag

        for tag, (n_tag, word_count) in self.emission_count.items():
            for word, count in word_count.items():
                self.emission_prob[tag][word] = count / n_tag

    def print_path(self, v_path, emissions):
        # Print a table of steps from dictionary
        yield 't:' + ' '.join('{:>7d}'.format(i) for i in range(len(v_path)))
        yield 'e:' + ' '.join('{:>7d}'.format(i) for i in emissions)
        for s in v_path[0]:
            yield "{}:  ".format(s) + " ".join(
                '{:.5f}'.format(v[s]['prob']) for v in v_path)
   
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

        # print('-' * 70)
        # print('Viterbi Path:')
        # for line in self.print_path(v_path, emissions):
        #     print(line)

        # Choose the final state
        max_prob = 0.
        state_selected = None
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
        state_path = back_path[-2::-1]

        return state_path, max_prob

    def main(self, test_path, test_tagged_path):
        accuracy = []
        test_sentences = self.load_file(test_path)
        test_tagged_sentences = self.load_file(test_tagged_path)
        for sentence, tagged_sentence in \
                zip(test_sentences, test_tagged_sentences):
            words_test = self.normalize_sentence(sentence)
            tagged_words_test = self.normalize_sentence(tagged_sentence)
            assert len(words_test) == len(tagged_words_test)
            for word, word_tag in zip(words_test, tagged_words_test):
                word_t, tag = word_tag.rsplit('/', 1)
                word, word_t = self.check_word(word, word_t)
                tag_pred = self.word_tag_prob[word][0][0]
                accuracy.append(1 if tag == tag_pred else 0)
        accuracy = np.array(accuracy).mean()
        print(accuracy)


if __name__ == '__main__':

    HMM('../inputs/dice_4.txt').run()
