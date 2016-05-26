# dutchembeddings

Repository for the word embeddings described in [Evaluating Unsupervised Dutch Word Embeddings as a Linguistic Resource](http://www.lrec-conf.org/proceedings/lrec2016/pdf/1026_Paper.pdf), presented at [LREC 2016](http://lrec2016.lrec-conf.org/en/).

## Embeddings

To download the embeddings, please click on the links in the following table. In almost all cases, the 320-dimensional embeddings outperform the 160-dimensional embeddings.

All embeddings are released under the [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

The software is released under the [GNU GPL 2.0](http://www.gnu.org/licenses/old-licenses/gpl-2.0.html).

| Corpus        | 160           | 320   |
| ------------- |:-------------:| -----:|
| Roularta      | [link](http://www.clips.uantwerpen.be/dutchembeddings/roularta-160.tar.gz) | [link](http://www.clips.uantwerpen.be/dutchembeddings/roularta-320.tar.gz) |
| [Wikipedia](https://dumps.wikimedia.org/nlwiki/20160501/)     | [link](http://www.clips.uantwerpen.be/dutchembeddings/wikipedia-160.tar.gz)      |   [link](http://www.clips.uantwerpen.be/dutchembeddings/wikipedia-320.tar.gz) |
| [Sonar500](http://tst-centrale.org/nl/tst-materialen/corpora/sonar-corpus-detail)      | [link](http://www.clips.uantwerpen.be/dutchembeddings/sonar-160.tar.gz)      |    [link](http://www.clips.uantwerpen.be/dutchembeddings/sonar-320.tar.gz) |
| Combined      |   [link](http://www.clips.uantwerpen.be/dutchembeddings/combined-160.tar.gz)         |  [link](http://www.clips.uantwerpen.be/dutchembeddings/combined-320.tar.gz)   |
| [COW](http://corporafromtheweb.org/)           | -           |  [small](http://www.clips.uantwerpen.be/dutchembeddings/cow-320.tar.gz), [big](http://www.clips.uantwerpen.be/dutchembeddings/cow-big.tar.gz)   |

The embeddings are currently provided in `.txt` files which contain vectors in `word2vec` format, which is structured as follows:

The first line contains the size of the vectors and the vocabulary size, separated by a space.

Ex: `320 50000`

Each line thereafter contains the vector data for a single word, and is presented as a string delimited by spaces. The first item on each line is the word itself, the `n` following items are numbers, representing the vector of length `n`. Because the items are represented as strings, these should be converted to floating point numbers.

Ex: `hond 0.2 -0.542 0.253 etc.`

If you use `python`, these files can be loaded with [`gensim`](https://github.com/piskvorky/gensim) or [`reach`](https://github.com/stephantul/reach), as follows.

```python
# Gensim
from gensim.models import word2vec

model = Word2Vec.load_word2vec_format("path/to/vector", binary=False)
katvec = model['kat']
model.most_similar('kat')

# Reach
from reach import Reach

r = Reach("path/to/vector", header=True)
katvec = r['kat']
r.most_similar('kat')
```

## Relationship dataset

If you want to test the quality of your embeddings, you can use the `relation.py` script. This script takes a `.txt` file of predicates, and creates dataset which is used for evaluation.

This currently only works with the gensim word2vec models.

Example:
```python
# Load the predicates.
rel = Relation("data/question-words.txt")

# load a word2vec model
model = Word2vec.load_word2vec_format("path/to/model")

# Test the model
rel.test_model(model)
```

## Citing

If you use any of the resources from this paper, please cite our paper, as follows:

```bibtex
@InProceedings{tulkens2016evaluating,
  author = {Stephan Tulkens and Chris Emmery and Walter Daelemans},
  title = {Evaluating Unsupervised Dutch Word Embeddings as a Linguistic Resource},
  booktitle = {Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016)},
  year = {2016},
  month = {may},
  date = {23-28},
  location = {Portorož, Slovenia},
  editor = {Nicoletta Calzolari (Conference Chair) and Khalid Choukri and Thierry Declerck and Marko Grobelnik and Bente Maegaard and Joseph Mariani and Asuncion Moreno and Jan Odijk and Stelios Piperidis},
  publisher = {European Language Resources Association (ELRA)},
  address = {Paris, France},
  isbn = {978-2-9517408-9-1},
  language = {english}
 }
 ```

Please also consider citing the corpora of the embeddings you use. Without the people who made the corpora, the embeddings could never have been created.
