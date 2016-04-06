# -*- coding: utf-8 -*-

import logging
import operator
import csv
import re
import numpy as np
import json
import time

from collections import defaultdict
from string import punctuation
from gensim.models import Word2Vec
from itertools import chain
from sppmimodel import SPPMIModel

# Same prepocessing used in the corpus. For consistency.
COW_preprocess = re.compile(r"(&.*?;)|((?<=\s)([a-tv-z]|\[.*\]|[^a-zA-Z\s]+?))(?=\s|\b$)")
removepunct = re.compile(r"[^\w\s'-]")


class DialectDetector(object):

    def __init__(self, datapath, labelpath):
        """
        Initialization of regions + countries.
        """

        # Noord-brabant ontbreekt in het w2v model.
        self._regions = (u"antwerpen", u"oost-vlaanderen", u"west-vlaanderen",
                         u"groningen", u"friesland", u"drenthe", u"overijssel", u"flevoland",
                         u"gelderland", u"utrecht", u"noord-holland", u"zuid-holland", u"zeeland", u"noord-brabant",
                         u"limburg", u"vlaams-brabant")

        self._countries = (u"belgië", u"nederland")  # misschien ook gewesten (vlaanderen?)

        self.model = None
        self._dictionaries = None
        self.data, labels = self.load_frogged_data(datapath, labelpath)

        self.labels = []

        locales = self._regions + self._countries
        for l in labels:
            self.labels.append([1 if x == locales.index(l) else 0 for x in range(len(locales))])

        self.labels = np.array(self.labels)
        self.labels_region = self.labels[:, range(len(self._regions))]

    @staticmethod
    def load_frogged_data(pathtofile, pathtolabels):
        """
        Returns posts and labels from a frogged file.

        :param pathtofile: path to the frog file.
        :param pathtolabels: path to the labels file.
        :param regions: list of regions.
        :return: A tuple of posts and labels.
        """

        posts = []
        labels = []

        f = csv.reader(open(pathtolabels))
        post = []
        for d in open(pathtofile, encoding='utf-8'):

            if not d.rstrip():
                posts.append(post)
                post = []
                labels.append(next(f)[0].lower())
            else:
                d = d.split()
                if d[1] not in punctuation:
                    post.append((d[1].lower(), d[2]))

        return posts, labels

    @property
    def regions(self):

        return self._regions

    @property
    def countries(self):

        return self._countries

    @property
    def locations(self):

        return self._countries + self._regions

    def load_dictionaries(self, pathtodicts):
        """
        load the dictionaries from disk and assign them to the _dictionaries param

        Format: {LOCATION (e.g. limburg): [list_of_words]}

        :param pathtodicts: the path to the JSON file.
        :return: None
        """

        self._dictionaries = json.load(open(pathtodicts))

    def task_1(self):
        """
        Task 1 assumes that every post belongs to a region, and does not take countries into account.

        :param sentences: list of sentences
        :return: a list of labels, one for each sentence
        """

        return self._calc_sentences_model(self.data, self._regions)

    def task_2(self):
        """
        Task 2 assumes that every post belongs to a region or country.

        :param sentences: list of sentences
        :return: a list of labels, one for each sentence
        """

        return self._calc_sentences_model(self.data, self._regions + self._countries)

    def task_3(self):
        """
        Task 3 is a dictionary baseline.

        :param sentences: list of sentences
        :return: a list of labels, one for each sentence
        """

        return self._calc_sentences_dict(self.data, self._regions)

    def task_4(self):
        """
        Task 4 is like task 2, but instead of using the countries as possible labels, we remove any
        words in the sentence that are given the label country, effectively removing their influence.

        :param sentences: list of sentences
        :return: list of labels, one for each sentence
        """

        return self._calc_sentences_model(self.data, self._regions + self._countries, use_filter=True)

    def task_5(self):
        """
        Like task 4, but using the dictionaries as filters.

        :param sentences: list of sentences
        :return: list of labels, one for each sentence
        """

        return self._calc_sentences_model(self.data, self._regions + self._countries, use_dict_filter=True)

    @staticmethod
    def detect(function, labels, mrr):

        """
        Runs a w2v dialect detection using a specific function.

        :param function: the function to use
        :param labels: a list of labels
        :param mrr: whether to use Mean Reciprocal Rank or Accuracy
        :return: The score (MRR or Accuracy) and the mean score.
        """

        X = function()

        if not mrr:
            X[X < 1.0] = 0.0

        # Multiplying X with the labels has the effect of giving incorrect answers weight 0.

        result = X * labels
        mean = np.sum(result) / np.sum(labels)

        return list(np.sum(result, axis=0) / np.sum(labels, axis=0)), mean

    def run_model(self, mrr):

        scores = dict()

        scores[self.task_1.__name__] = self.detect(self.task_1, self.labels_region, mrr)
        scores[self.task_4.__name__] = self.detect(self.task_4, self.labels, mrr)

        return scores

    def run_dict(self, mrr):

        scores = dict()

        scores[self.task_3.__name__] = self.detect(self.task_3, self.labels_region, mrr)

        return scores

    def _calc_sentences_model(self, sentences, locations, use_filter=False, use_dict_filter=False):
        """
        Calculates, for each sentence, for each word in that sentence, the closest neighbour in the list of locations.
        The location that is most often chosen is then returned as the most likely region for this sentence.

        :param sentences: list of sentences
        :param locations: list of locations to be compared to the words in the sentences.
        :param use_filter: whether to use the countries as a filter.
        :param use_dict_filter: whether to use the dictionaries as a filter.
        :return: a list of labels, one for each sentence.
        """

        labels = []

        start = time.time()

        for idx, s in enumerate(sentences):

            if idx % 1000 == 0:
                print("did {0} sentences in {1} seconds".format(idx, time.time() - start))

            scores = self._sentence_to_location_model(s, locations, use_dict_filter)

            if use_filter:
                for c in self._countries:
                    try:
                        scores.pop(c)
                    except KeyError:
                        pass
            if scores:
                found = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)[0][0]
                label = []

                for x in locations:
                    try:
                        label.append(1.0 / (found.index(x) + 1))
                    except ValueError:
                        label.append(0)

                labels.append(label)
            # Edge case: happens when using filter and every word is filtered.
            else:
                labels.append(len(locations) * [0])

        return np.array(labels)

    def _sentence_to_location_model(self, sentence, locations, use_dict_filter):
        """
        Compares each word in a given sentence to each given location.

        :param sentence: a string of words representing a sentence
        :param locations: a list of locations
        :param use_dict_filter: whether to use the dictionaries as a filter.
        :return: a dictionary of numbers, representing the counts for each region.
        """

        counts = defaultdict(int)

        for word, lemma in sentence:

            if use_dict_filter:
                if lemma in self._dictionaries["nederland"]:
                    counts["nederland"] += 1
                    continue

            try:
                region = self._calc_word(word, locations)

            except KeyError:
                continue
            counts[region] += 1

        return counts

    def _calc_word(self, word, locations):
        """
        compares a word to each location and returns location with highest similarity to the word.

        :param word: the word to be compared
        :param locations: the list of locations
        :return: the most probable location for this word
        """

        return sorted([(l, self.model.similarity(l, word)) for l in locations], key=operator.itemgetter(1), reverse=True)[0][0]

    def _calc_sentences_dict(self, sentences, locations):
        """
        For each word in each sentence, see if it is in a dict corresponding to a location.

        :param sentences: a list of sentences
        :param locations: a list of locations
        :return:
        """

        labels = []

        for s in sentences:

            scores = self._sentence_to_location_dict(s, locations)
        
            if scores:
                found = list(zip(*(sorted(scores.items(), key=operator.itemgetter(1), reverse=True))))[0]
                label = []

                for idx, _ in enumerate(locations):
                    try:
                        label.append(1.0 / (found.index(idx) + 1))
                    except ValueError:
                        label.append(0)

                labels.append(label)

            else:
                labels.append([0 for x in range(len(locations))])

        return np.array(labels)

    def _sentence_to_location_dict(self, sentence, locations):
        """
        For each word in a sentence, note how many words correspond
         to a dialect for a certain location.

        :param sentence: the sentence, tokenized.
        :param locations: a list of locations
        :return: a dictionary of with keys as locations, values as counts.
        """

        counts = defaultdict(int)

        for word, lemma in sentence:

            for l in locations:
                word_in_location = word in self._dictionaries[l]
                counts[l] += word_in_location

        return {k: v for k, v in counts.items() if v}

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    modelpath = "/home/tulkens/embeddings_dutch/COW/BIG/COW-wordvecs-big"
    pathtosparse = "cow_small/cow_small-SPPMI-sparse-1-shift.npz"
    pathtosparsewords = "cow_small/cow_smallmapping.json"

    mrr = True

    isw2v = False
    dia = DialectDetector("facebook-nl-region_frogged.tsv", "facebook-nl-region.csv")

    dia.model = Word2Vec.load_word2vec_format(modelpath)

    emb = set(dia.model.vocab.keys())
    corpus = {word.lower() for word, lemma in chain.from_iterable(dia.data)}

    cov = len(emb & corpus) / len(corpus)

    sparse = json.load(open(pathtosparsewords))

    emb = set(sparse.keys())

    cov2 = len(emb & corpus) / len(corpus)

    '''if isw2v:
        dia.model = Word2Vec.load_word2vec_format(modelpath)
    else:
        dia.model = SPPMIModel(pathtosparsewords, pathtosparse, initkeys=dia.locations)

    dia.load_dictionaries("dictionaries.json")

    taskscores = dia.run_model(mrr=mrr)
    json.dump(taskscores, open("results-{0}-mrr:{1}.json".format(modelpath.split("/")[-1], mrr), 'w'))

    dictscores = dia.run_dict(mrr)
    json.dump(dictscores, open("results-dict-mrr:{0}.json".format(mrr), 'w'))

    #dictscores = dia.run_dict(mrr=True)'''


