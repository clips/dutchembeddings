import json
import numpy as np
import scipy.sparse

from gensim import matutils
from gensim.models import Word2Vec
from gensim.models.word2vec import Vocab
from io import open


class SPPMIModel(Word2Vec):
    """Class which allows for addressing an SPPMI matrix like a word2vec model."""

    def __init__(self, pathtomapping, pathtovectors, pathtocounts="", initkeys=()):
        """
        SPPMI model equivalent to a gensim word2vec model.

        :param pathtomapping:
        :param pathtovectors:
        :param pathtocounts:
        :param initkeys:
        :return:
        """

        super(SPPMIModel, self).__init__()
        self.word2index = json.load(open(pathtomapping))
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.word_vectors = self._load_sparse(pathtovectors)

        self.vocab = {}

        self.fast_table = {k: {} for k in initkeys}

        if pathtocounts:

            counts = json.load(open(pathtocounts))

            for w, idx in self.word2index.items():
                v = Vocab(count=counts[w], index=idx)
                self.vocab[w] = v

    @staticmethod
    def _load_sparse(path):
        """
        Snippet from:

            http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format

        :param path: the path to the sparse matrix in CSR format.
        :return: scipy.sparse.csr_matrix
        """

        loader = np.load(path)
        return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

    def similarity(self, w1, w2):
        """
        Calculates the similarity between two words as the dot product of the vectors.
        If OOV words are queried, this will return 0.
        Uses intermediate caches to speed up queries.

        :param w1: the first word
        :param w2: the second word
        :return: a similarity score, where higher is more similar.
        """

        try:
            return self.fast_table[w1][w2]
        except KeyError:
            try:
                self.fast_table[w1][w2] = self.word_vectors[self.word2index[w1]].dot(self.word_vectors[self.word2index[w2]].T)[0, 0]
            except KeyError:
                self.fast_table[w1][w2] = 0

        return self.fast_table[w1][w2]

    def most_similar(self, positive=[], negative=[], topn=10, restrict_vocab=False):
        """
        Gets the most similar words to a set of words.

        :param positive: The positive words to consider
        :param negative: The negative words to consider
        :param topn: How many words to return.
        :param restrict_vocab: Unused, for compatibility with w2v.
        :return: the n most similar words to the queried words.
        """

        if isinstance(positive, str) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, (str, np.ndarray)) else word
            for word in positive]
        negative = [
            (word, -1.0) if isinstance(word, (str, np.ndarray)) else word
            for word in negative]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, np.ndarray):
                mean.append(weight * word)
            elif word in self.word2index:
                word_index = self.word2index[word]
                mean.append(weight * self.word_vectors[word_index])
                all_words.add(word_index)
            else:
                raise KeyError("word '%s' not in vocabulary" % word)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        if scipy.sparse.issparse(self.word_vectors):
            mean = scipy.sparse.vstack(mean)
        else:
            mean = np.array(mean)
        mean = matutils.unitvec(mean.mean(axis=0)).astype(self.word_vectors.dtype)

        dists = self.word_vectors.dot(mean.T).flatten()
        if not topn:
            return dists
        best = np.argsort(dists)[::-1][:topn + len(all_words)]

        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]

        return result[:topn]