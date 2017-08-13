import warnings
from asl_data import SinglesData


def recognize(models, test_set):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for tword, (x, l) in test_set.get_all_Xlengths().items():
        score = float("-inf")
        guess = ""
        pdict = {}

        for trainword, model in models.items():
            try:
                lp = model.score(x, l)
                pdict[trainword] = lp

            except:
                pdict[tword] = float("-inf")

            if lp > score:
                score = lp
                guess = trainword

        probabilities.append(pdict)
        guesses.append(guess)

    return probabilities, guesses
