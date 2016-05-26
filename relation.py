import json
import logging

from itertools import permutations
from sppmimodel import SPPMIModel


class Relation:
    """A class for making relationship/analogy tests easy"""

    def __init__(self, pathtoset):
        """
        A class which is used to test the accuracy of models viz. some set of questions/predicates.

        :param pathtoset: the path to the predicate set.
        :return: None
        """

        self.pathtoset = pathtoset

    def test_model(self, model):
        """
        Tests a given model with the set.

        :param model: the model for which to test accuracy
        :return: a dictionary with scores per section, and a total score.
        """

        # The most_similar assignment is neccessary because the most_similar function might refer to the original
        # Word2Vec function.
        return model.accuracy(self.pathtoset, most_similar=model.__class__.most_similar, restrict_vocab=None)

    @staticmethod
    def create_set(categories, outfile):
        """
        Creates a test-set .txt file for use in word2vec.
        Conforms to word2vec specs, from the google code repository: https://code.google.com/archive/p/word2vec/

        :param categories: The categories and words in the categories: {NAME: [[tuple_1],[tuple_2],...,[tuple_n]]}
        :param outfile: The file to which to write the text.
        :return: None
        """

        with open(outfile, 'w', encoding='utf8') as f:

            for k, v in categories.items():
                f.write(u": {0}\n".format(k))
                for x in permutations([" ".join(x).lower() for x in v], 2):
                    f.write(u"{0}\n".format(" ".join(x)))

if __name__ == "__main__":

    # Loads the category file for the Dutch relation test words.
    cats = json.load(open("data/semtest.json"))

    # Create the relation set tuples, and saves the result to question-words.txt
    Relation.create_set(cats, "data/question-words.txt")
    
    pathtomodel = ""
    model = Word2Vec.load_word2vec_format(pathtomodel)

    rel = Relation("data/question-words.txt")
    scores = rel.test_model(model)