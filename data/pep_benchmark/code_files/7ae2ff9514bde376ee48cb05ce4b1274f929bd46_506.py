import pandas as pd
import random
from collections import Counter

PROB_MASK_NON_KEYWORDS = 0.9

PROB_KEEP_KEYWORDS = 0.8
PROB_REPLACE_KEYWORDS = 0.15
PROB_MASK_KEYWORDS = 0.05

class MaskCorpus():

    def __init__(self,
                 corpus,
                 idx,
                 keep_tags,
                 keep_words):

        self.keep_tags=keep_tags
        self.keep_words=keep_words
        self.corpus=corpus
        self.idx=idx
        self.tagger=self._default_tagger()
        self.vocab=self.build_vocab(corpus)

    def build_vocab(self, corpus):
        vocab = Counter()
        for sentence in corpus:
            vocab += Counter(sentence.split())
        return list(vocab)

    def _default_tagger(self):
        try:
            import spacy_udpipe
        except ImportError:
            raise('You are missing pos tagger, try `pip install spacy_udpipe`')
        spacy_udpipe.download('en')
        return spacy_udpipe.load('en')

    def _mask_non_keywords(self, word):
        return '[MASK]' if random.random()<PROB_MASK_NON_KEYWORDS else word

    def _mask_keywords(self, word):
        choice = random.random()
        if choice < PROB_KEEP_KEYWORDS:
            return word
        elif PROB_KEEP_KEYWORDS<=choice and choice<PROB_REPLACE_KEYWORDS+PROB_KEEP_KEYWORDS:
            return random.sample(self.vocab, 1)[0]
        else:
            return '[MASK]'

    def generate_corpus(self):
        masked_corpus = []
        for id, sentence in zip(self.idx, self.corpus):
            _sentence = self.tagger(sentence.lower())
            masked_sentence = []
            original_sentence = []
            for token in _sentence:
                if (token.text not in self.keep_words) and \
                    (token.pos_ not in self.keep_tags):
                    masked_sentence.append(self._mask_non_keywords(token.text))
                elif token.text in self.keep_words:
                    masked_sentence.append(token.text)
                else:
                    masked_sentence.append(self._mask_keywords(token.text))
                original_sentence.append(token.text)
            masked_corpus.append((id, ' '.join(masked_sentence), ' '.join(original_sentence)))
        return masked_corpus
