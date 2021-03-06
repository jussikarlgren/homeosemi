import sparsevectors
import math
import pickle
from languagemodel import LanguageModel

from logger import logger  # Simplest possible logger, replace with any variant of your choice.
error = True      # loglevel
debug = False     # loglevel
monitor = False   # loglevel


class SemanticSpace:
    def __init__(self, dimensionality: int=2000, denseness: int=10, name: str="no name"):
        self.name = name
        self.indexspace = {}    # dict: string - sparse vector
        self.contextspace = {}  # dict: string - denser vector
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.observedfrequency = {}  # dict: string - int
        self.constantdenseness = 10
        self.changed = False

    def observe(self, word, loglevel=False):
        """
        :rtype: object
        """
        if not self.contains(word):
            self.additem(word, None)
            logger("'" + str(word) + "' is new and now introduced: " + str(self.indexspace[word]), loglevel)
        self.observedfrequency[word] += 1

    def additem(self, item, vector=None):
        """
        Add new item to the space. Add randomly generated index vector (unless one is given as an argument or one
        already is recorded in index space); add empty context space, prep LanguageModel to accommodate item. Should
        normally be called from observe() but also at times from addintoitem.
        """
        if item not in self.indexspace:
            if vector is None:
                vector = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
            self.indexspace[item] = vector
        self.contextspace[item] = sparsevectors.newemptyvector(self.dimensionality)
        self.changed = True
        self.observedfrequency[item] = 0


    def additemintoitem(self, item, otheritem, weight=1, operator=None):
        """
        Update the context vector of item by adding in the index vector of otheritem multiplied by the scalar weight.
        If item is unknown, add it to the space. If otheritem is unknown add only an index vector to the space.
        :param item: str
        :param otheritem: str
        :param weight: float
        :param permutation: list
        :return: None
        """
        if not self.contains(item):
            self.additem(item)
        if otheritem not in self.indexspace:
            self.indexspace[otheritem] = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
        self.addintoitem(item, self.indexspace[otheritem], weight, operator)

    def addintoitem(self, item, vector, weight=1, operator=None):
        if not self.contains(item):
            self.additem(item)
        if operator is not None:
            vector = sparsevectors.permute(vector, operator)
        self.contextspace[item] = sparsevectors.sparseadd(self.contextspace[item],
                                                          sparsevectors.normalise(vector),
                                                          weight)
        self.changed = True

    def observecollocation(self, item, otheritem):
        if not self.contains(item):
            self.additem(item)
        if not self.contains(otheritem):
            self.additem(otheritem)
        self.addintoitem(item, sparsevectors.normalise(self.indexspace[otheritem]))
        self.addintoitem(otheritem, sparsevectors.normalise(self.indexspace[item]))

    def removeitem(self, item):
        if self.contains(item):
            del self.indexspace[item]
            del self.contextspace[item]
            del self.observedfrequency[item]
            self.changed = True

    def reducewordspace(self, threshold=1):
        items = list(self.indexspace.keys())
        for item in items:
            if self.observedfrequency[item] <= threshold:
                self.removeitem(item)
                self.changed = True

    def comb(self):
        for item in self.contextspace:
            self.contextspace[item].comb()

    #================================================================
    # input output wordspace
    def outputwordspace(self, filename):
        """
        Save wordspace to disk.
        :param filename: str
        """
        try:
            with open(filename, 'wb') as outfile:
                pickle.dump(self, outfile)
        except IOError:
                logger("Could not write >>" + filename + ".toto <<", error)

    def inputwordspace(self, vectorfile):
        try:
            cannedspace = open(vectorfile, 'rb')
            wordspace = pickle.load(cannedspace)
            return wordspace
        except IOError:
            logger("Could not read from >>" + vectorfile + "<<", error)
            return SemanticSpace()

    # ===========================================================================
    # querying the semantic space
    def contains(self, item):
        if item in self.indexspace and item in self.contextspace:
            return True
        else:
            return False

    def items(self):
        return self.indexspace.keys()

    def similarity(self, vector, anothervector):
        return sparsevectors.sparsecosine(vector, anothervector)

    def contextsimilarity(self, item, anotheritem):
        return sparsevectors.sparsecosine(self.contextspace[item], self.contextspace[anotheritem])

    def indextocontextsimilarity(self, item, anotheritem):
        if self.contains(item):
            return sparsevectors.sparsecosine(self.indexspace[item], self.contextspace[anotheritem])
        else:
            return 0.0

    def contextneighbours(self, item: str, number: int=10, weights: bool=False,
                          filtertag: bool=False, threshold: int=-1) -> list:
        '''
        Return the items from the contextspace most similar to the given item. I.e. items which have similar
        neighbours to the item. Specify number of items desired (0 will give all), if weights are desired, if
        only items with the same tag are desired, and if thresholding to a certain horizon is desired.
        '''
        neighbourhood = {}
        if filtertag:
            targetset = self.tagged[self.tag[item]]
        else:
            targetset = self.contextspace
        for i in targetset:  # was: for i in self.contextspace:
            if i == item:
                continue
            k = sparsevectors.sparsecosine(self.contextspace[item], self.contextspace[i])
            if k > threshold:
                neighbourhood[i] = k
        if not number:
            number = len(neighbourhood)
        if weights:
            r = sorted(neighbourhood.items(), key=lambda k: neighbourhood[k[0]], reverse=True)[:number]
        else:
            r = sorted(neighbourhood, key=lambda k: neighbourhood[k], reverse=True)[:number]
        return r

    def contexttoindexneighbours(self, item, number=10, weights=False, permutationname="nil"):
        '''
        Return the items whose index vectors are most similar to the given item's context vector. I.e. items which
        have occurred in contexts with the item.
        '''
        permutation = self.permutationcollection[permutationname]
        neighbourhood = {}
        for i in self.indexspace:
            neighbourhood[i] = sparsevectors.sparsecosine(self.contextspace[item],
                   sparsevectors.permute(self.indexspace[i], permutation))
        if not number:
            number = len(neighbourhood)
        if weights:
            r = sorted(neighbourhood.items(), key=lambda k: neighbourhood[k[0]], reverse=True)[:number]
        else:
            r = sorted(neighbourhood, key=lambda k: neighbourhood[k], reverse=True)[:number]
        return r

    def indextocontextneighbours(self, item, number=10, weights=False, permutationname="nil"):
        permutation = self.permutationcollection[permutationname]
        neighbourhood = {}
        for i in self.contextspace:
            neighbourhood[i] = sparsevectors.sparsecosine(sparsevectors.permute(self.indexspace[item], permutation),
                                              self.contextspace[i])
        if weights:
            r = sorted(neighbourhood.items(), key=lambda k: neighbourhood[k[0]], reverse=True)[:number]
        else:
            r = sorted(neighbourhood, key=lambda k: neighbourhood[k], reverse=True)[:number]
        return r
