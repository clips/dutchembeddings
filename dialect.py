# -*- coding: utf-8 -*-
import logging

import operator
import re
import json

import numpy as np

from gensim.models import Word2Vec
from collections import defaultdict, OrderedDict

# Same prepocessing used in the corpus. For consistency.
COW_preprocess = re.compile(r"(&.*?;)|((?<=\s)([a-tv-z]|\[.*\]|[^a-zA-Z\s]+?))(?=\s|\b$)")
removepunct = re.compile(r"[^\w\s'-]")


class DialectDetector(object):
    """Detection of regional variants of language based on word2vec or a dictionary"""

    def __init__(self, dictionary, model):
        """
        Initialization of regions + countries.
        """

        self._regions = (u"antwerpen", u"oost-vlaanderen", u"west-vlaanderen",
                         u"groningen", u"friesland", u"drenthe", u"overijssel", u"flevoland",
                         u"gelderland", u"utrecht", u"noord-holland", u"zuid-holland", u"zeeland", u"noord-brabant",
                         u"limburg", u"vlaams-brabant")

        self._countries = (u"belgië", u"nederland")

        self._model = model
        self._dictionaries = dictionary

    @property
    def regions(self):
        """
        The list of regions.

        :return: a list of regions.
        """

        return self._regions

    @property
    def countries(self):
        """
        The list of countries.

        :return: a list of countries.
        """

        return self._countries

    def featurize_labels(self, labels, include_countries=False):
        """
        Featurizes a list of labels to a matrix of one-hot encoded binary vectors.

        :param labels: A list of labels
            Ex. ['antwerpen', 'limburg', 'friesland', ..., 'friesland']
        :param include_countries: Whether to include the countries in the binary vector.
        Including the countries is necessary when using them as distractors, but these are
        never used for evaluation.
        :return: A matrix of vectors.
        """

        locales = self.regions if not include_countries else self.regions + self.countries
        y = [[1 if x == locales.index(l) else 0 for x in range(len(locales))] for l in labels]

        return np.array(y)

    def accuracy(self, y_pred, y, mrr):
        """
        Calculates the accuracy score of a system as compared to set of gold standard labels.

        :param y_pred: A numpy array of vectors containing MRR scores.
            Ex. [[0, 0,    0.5,  1, 0, 0.33, 0, ..., 0],
                 [0, 1,    0.25, 0, 0, 0,    0, ..., 0],
                 [0, 0.25, 0,    0, 0, 0,    0, ..., 1],
                 [...                                 ],
                 [...                            ,0.11]]

        :param y: A list of one-hot encoded vectors, representing the true labels.
            Ex. [[0, 0, 0, 1, 0, 0, 0, ..., 0],
                 [0, 1, 0, 0, 0, 0, 0, ..., 0],
                 [0, 0, 0, 0, 0, 0, 0, ..., 1],
                 [...                        ],
                 [...                      ,0]]

        :param mrr: Whether to use the Mean Reciprocal Rank. If this is false, it has the effect of removing
        any scores which are not exactly 0, in effect only taking into account the items which were ranked first.
        :return: A tuple of accuracy scores per individual label, and the mean of those scores.

            Ex. ({'antwerpen': 0.22, 'limburg': 0.12, ...}, 0.143)
        """

        y = np.array(y)

        if not mrr:
            y_pred[y_pred < 1.0] = 0.0

        result = y_pred * y
        mean = np.sum(result) / np.sum(y)

        return {k: v for k, v in zip(self._regions, np.sum(result, axis=0) / np.sum(y, axis=0))}, mean

    def run_dictionary(self, data, y_true, mrr):
        """
        Runs the dialect detection procedure using a dictionary.

        :param data: The sentences on which to run.
        :param y_true: The gold standard labels.
        :param mrr: Whether to use Mean Reciprocal Rank or Accuracy.
        :return: An accuracy score or the MRR score, depending on the MRR parameter.
        """

        y_pred = self._calc_sentences_w2v(data, self._regions, use_dict=True)
        return self.accuracy(y_pred=y_pred, y=y_true, mrr=mrr)

    def run_model(self, data, y_true, mrr, use_filter=False):
        """
        Runs the dialect detection using a model.

        :param data: The sentences on which to run.
        :param y_true: The gold standard labels, encoded as a matrix of one-hot encoded vectors.
        :param mrr: Whether to use Mean Reciprocal Rank or Accuracy.
        :return: An accuracy score or the MRR score, depending on the MRR parameter.
        """

        if use_filter:
            y_pred = self._calc_sentences_w2v(data, self.regions+self.countries, use_filter=True)
        else:
            y_pred = self._calc_sentences_w2v(data, self.regions, use_filter=False)
        return self.accuracy(y_pred=y_pred, y=y_true, mrr=mrr)

    def _calc_sentences_w2v(self, sentences, locales, use_filter=False, use_dict=False):
        """
        Calculates, for each sentence, for each word in that sentence, the closest neighbour in the list of locales.
        The locale that is most often chosen is then returned as the most likely region for this sentence.

        :param sentences: list of sentences
        :param locales: list of locales to be compared to the words in the sentences.
        :param use_filter: whether to use the countries as a filter.
        :return: a list of labels, one for each sentence.
        """

        labels = []

        for s in sentences:

            if use_dict:
                scores = self._sentence_to_locale_dict(s, locales)
            else:
                scores = self._sentence_to_locale_w2v(s, locales)

                if use_filter:
                    for c in self._countries:
                        try:
                            scores.pop(c)
                        except KeyError:
                            pass
            if scores:
                found, _ = zip(*sorted(scores.items(), key=operator.itemgetter(1), reverse=True))
                label = []

                for x in locales:
                    try:
                        label.append(1.0 / (found.index(x) + 1))
                    except ValueError:
                        label.append(0)

                labels.append(label)
            # Edge case: happens when using filter and every word is filtered.
            else:
                labels.append(len(locales) * [0])

        return np.array(labels)

    def _sentence_to_locale_w2v(self, sentence, locales):
        """
        Compares each word in a given sentence to each given locale.

        :param sentence: a string of words representing a sentence
        :param locales: a list of locales
        :param use_dict_filter: whether to use the dictionaries as a filter.
        :return: a dictionary of numbers, representing the counts for each region.
        """

        counts = defaultdict(int)

        for word in sentence:

            try:
                # Calculate the similarity score from each word to the name of all regions.
                sims = [(l, self._model.similarity(l, word)) for l in locales]
                # Sort the regions by their similarity scores.
                regions, _ = zip(*sorted(sims, key=operator.itemgetter(1), reverse=True))

            except KeyError:
                continue
            # Add a count to the region with the greatest similarity.
            counts[regions[0]] += 1

        return counts

    def _sentence_to_locale_dict(self, sentence, locales):
        """
        Determines the locale of a single sentence using the dictionary.

        :param sentence: The sentence for which to determine the locale.
        :param locales: The list of locales to consider.
        :return:
        """

        counts = OrderedDict()

        for word in sentence:

            for l in locales:
                if word in self._dictionaries[l]:
                    try:
                        counts[l] += 1
                    except KeyError:
                        counts[l] = 1

        return counts

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    dictionary = {k: set(v) for k, v in json.load(open('data/dictionaries.json')).items()}

    # Load the word2vec model.
    pathtomodel = ""
    model = Word2Vec.load_word2vec_format(pathtomodel)

    dia = DialectDetector(dictionary=dictionary, model=model)

    # Load data and labels.
    x = json.load(open(""))
    y = json.load(open(""))

    # Set MRR to true or false.
    mrr = True
    y = dia.featurize_labels(y, include_countries=False)
    result, mean = dia.run_dictionary(x, y, mrr)
    print("TASK 3: {0}".format(result))
    print("Mean 3: {0}".format(mean))

    result, mean = dia.run_model(x, y, mrr)
    print("TASK 1: {0}".format(result))
    print("Mean 1: {0}".format(mean))

    result, mean = dia.run_model(x, y, mrr)
    print("TASK 2: {0}".format(result))
    print("Mean 2: {0}".format(mean))