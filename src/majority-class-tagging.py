import numpy as np
from copy import deepcopy
from collections import defaultdict


class PoSTagger(object):
    
    def __init__(self, train_path) -> None:
        
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
    
    def check_word(self, word, word_t):
        if word == '\'':
            word = word_t
        assert word == word_t, (word, word_t)
        return word, word_t

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

    PoSTagger('../data/brown.train.tagged.txt').main(
        test_path='../data/brown.test.raw.txt',
        test_tagged_path='../data/brown.test.tagged.txt'
    )
    