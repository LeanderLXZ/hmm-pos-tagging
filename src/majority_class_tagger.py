import time
import argparse
import numpy as np
from copy import deepcopy
from collections import defaultdict


class MajorityClassTagger(object):
    
    def __init__(self, train_path, apply_rules=False) -> None:
        
        self.apply_rules = apply_rules
        self.word_tag_count = defaultdict(lambda: 'nn')
        self.word_tag_prob = defaultdict(lambda: 'nn')
        self.sentences = self.load_file(train_path)
        self.get_word_tag_count()
        self.get_word_tag_prob()
    
    def load_file(self, file_name):
        with open(file_name, "r") as f:
            return f.readlines()
    
    def normalize_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.split(' ')
        if sentence[-1] == '\n':
            sentence.pop()
        if self.apply_rules:
            for i, word in enumerate(sentence):
                if len(word) > 1 and word[0] == '\'':
                    sentence[i] = word[1:]
        return sentence
    
    def get_word_tag_count(self):
        for sentence in self.sentences:
            for word_tag in self.normalize_sentence(sentence):
                word, tag = tuple(word_tag.rsplit('/', 1))
                if word in self.word_tag_count:
                    self.word_tag_count[word][0] += 1
                    if tag in self.word_tag_count[word][1]:
                        self.word_tag_count[word][1][tag] += 1
                    else:
                        self.word_tag_count[word][1][tag] = 1
                else:
                    self.word_tag_count[word] = [1, defaultdict(lambda: 0)]
                    self.word_tag_count[word][1][tag] = 1

    def get_word_tag_prob(self):
        word_tag_prob = deepcopy(self.word_tag_count)
        for word, (word_count, tag_count) in self.word_tag_count.items():
            max_prob = ['', 0]
            for tag, count in tag_count.items():
                prob_tag = count / word_count
                if prob_tag > max_prob[1]:
                    max_prob = [tag, prob_tag] 
                word_tag_prob[word][1][tag] = prob_tag
            word_tag_prob[word][0] = tuple(max_prob)
        self.word_tag_prob = word_tag_prob

    def main(self, test_path, test_tagged_path):
        start_time = time.time()
        accuracy = []
        test_sentences = self.load_file(test_path)
        test_tagged_sentences = self.load_file(test_tagged_path)
        
        error_counter = {}
        
        for sentence, tagged_sentence in \
                zip(test_sentences, test_tagged_sentences):
            # Normalize sentence
            words_test = self.normalize_sentence(sentence)
            tagged_words_test = self.normalize_sentence(tagged_sentence)
            assert len(words_test) == len(tagged_words_test)
            
            # Testing
            pre_word = None
            i = 0
            for word, word_tag in zip(words_test, tagged_words_test):
                word_t, tag_true = word_tag.rsplit('/', 1)
                
                # Get predicted tag
                tag_pred = self.word_tag_prob[word][0][0]
                
                if self.apply_rules:
                    if word == 'to':
                        word_next = words_test[i+1]
                        tag_next = self.word_tag_prob[word_next][0][0]
                        if 'v' not in tag_next and tag_next != 'be':
                            tag_pred = 'in'
                    
                    if word == 'it':
                        word_pre = words_test[i-1]
                        if 'v' in self.word_tag_prob[word_pre][0][0]:
                            tag_pred = 'ppo'
                            
                    if word == 'that':
                        word_pre = words_test[i-1]
                        if 'in' in self.word_tag_prob[word_pre][0][0]:
                            tag_pred = 'dt'
                    
                    if word == 'her' :
                        if i + 1 < len (words_test):
                            word_next = words_test[i+1]
                            tag_next = self.word_tag_prob[word_next][0][0]
                            if 'n' not in tag_next:
                                tag_pred = 'ppo'
                        else:
                            tag_pred = 'ppo'
                    
                    if word == 'as':
                        word_next = words_test[i+1]
                        tag_next = self.word_tag_prob[word_next][0][0]
                        if tag_next in ['ap', 'jj']:
                            tag_pred = 'ql'
                    
                    if word == 'you':
                        word_pre = words_test[i-1]
                        if 'v' in self.word_tag_prob[word_pre][0][0]:
                            tag_pred = 'ppo'

                    if word == 'so':
                        word_next = words_test[i+1]
                        tag_next = self.word_tag_prob[word_next][0][0]
                        if tag_next not in ['ap', 'jj']:
                            tag_pred = 'rb'

                # Count accuracy
                accuracy.append(1 if tag_pred == tag_true else 0)
                
                i += 1
                
        # Compute the final accuracy
        accuracy = np.array(accuracy).mean()
        print('Accuracy: {:.4f}%'.format(accuracy * 100))
        print('Runtime: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rules', action="store_true")
    args = parser.parse_args()
    apply_rules_ = True if args.rules else False

    MajorityClassTagger(
        train_path='../data/brown.train.tagged.txt',
        apply_rules=apply_rules_
        ).main(
        test_path='../data/brown.test.raw.txt',
        test_tagged_path='../data/brown.test.tagged.txt'
        )
    