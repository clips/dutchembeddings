import fnmatch
import os
import re

COW_preprocess = re.compile(r"(&.*?;)|((?<=\s)([a-tv-z]|[^a-zA-Z\s]+?))(?=(\s))")
punct = re.compile(r"\s[a-tv-z]\s|(&.*?;)|[^a-z\s]+?", re.I)


class SentenceIter(object):

    def __init__(self, root, extension='*.txt', iscow=False):
        """
        A sentence iterator for use with word2vec and the other corpora. It iterates over all files in a folder and
        subfolders with a given extension.

        :param root: The root folder.
        :param extension: The extension for which to look.
        :param iscow: Boolean value which determines the regex used for preprocessing the data.
        :return:
        """

        self.path = root
        self.filenames = []

        if iscow:
            self.regex = punct
        else:
            self.regex = COW_preprocess
        
        for root, dirnames, filenames in os.walk(self.path):
            for filename in fnmatch.filter(filenames, extension):
                self.filenames.append(os.path.join(root, filename))

    def __iter__(self):

        # assume corpus consists of multiple documents
        for filename in self.filenames:
            for line in open(os.path.join(self.path, filename), encoding='utf-8'):
                sent = line.lower().split()

                if len(sent) < 5:
                    continue
                else:
                    yield self.regex.sub(" ", line.lower()).split()
