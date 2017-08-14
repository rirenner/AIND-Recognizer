import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


from pprint import pprint as pp
def jfp(obj):
     return pp([x for x in dir(obj) if not x.startswith('_')])

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences, all_word_Xlengths, this_word,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    # def train_a_word(word, num_hidden_states, features):

    #   warnings.filterwarnings("ignore", category=DeprecationWarning)
    #    training = asl.build_training(features)
    #    X, lengths = training.get_word_Xlengths(word)
    #    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    #    logL = model.score(X, lengths)
    #    return model, logL

    # demoword = 'BOOK'
    # model, logL = train_a_word(demoword, 3, features_ground)
    # print("Number of states trained in model for {} is {}".format(demoword, model.n_components))
    # print("logL = {}".format(logL))

    def getscore(self,i):
        model = self.base_model(i)
        logLoss = model.score(self.X, self.lengths)
        logn = np.log(len(self.X))
        d = model.n_features
        p = i ** 2 + 2 * d * i - 1

        return -2.0 * logLoss + p * logn, model


    def select(self):
        try:
            l = range(self.min_n_components, self.max_n_components + 1)
            score = float("Inf")
            model = self.base_model(l[0])
            for i in l:
                logl, tmodel = self.getscore(i)
                if logl < score:
                    score = logl
                    model = tmodel
            return model
        except:
            #import pdb; pdb.set_trace()
            return self.base_model(self.n_constant)
        #for component in components:
        #    v,m =


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def getscore(self, i):
        model = self.base_model(i)
        vals = []
        for w, (x, l) in self.hwords.items():
            if w not in self.this_word:
                vals.append(model.score(x, l))
        return model.score(self.X, self.lengths) - np.mean(vals), model

    def select(self):
        """ select the best model for self.this_word based on
        DIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        try:
            score = float("-Inf")
            model = None
            comps = range(self.min_n_components, self.max_n_components+1)
            for c in comps:
                l, tmodel = self.getscore(c)
                if l > score:
                    score = l
                    model = tmodel
            return model

        except:
            #import pdb; pdb.set_trace()
            return self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def getscore(self, i):
        vals = []
        for tx, testx in KFold().split(self.sequences):
            #for tx, testx in KFold(n_splits=2).split(self.sequences):
            self.X, self.lengths = combine_sequences(tx, self.sequences)

            model = self.base_model(i)
            rx, l = combine_sequences(testx, self.sequences)
            vals.append(model.score(rx, l))

        return np.mean(vals), model

    def select(self):
        """ select the best model for self.this_word based on
        CV score for n between self.min_n_components and self.max_n_components
        It is based on log likehood
        :return: GaussianHMM object
        """
        try:
            score = float("-Inf")
            model = None
            comps = range(self.min_n_components, self.max_n_components+1)
            for c in comps:
                l, tmodel = self.getscore(c)
                if l > score:
                    score = l
                    model = tmodel
            return model
        except:
            #import pdb; pdb.set_trace()
            return self.base_model(self.n_constant)
