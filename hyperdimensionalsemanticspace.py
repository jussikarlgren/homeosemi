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
        self.tag = {}          # dict: string - string
        self.tagged = {}       # dict: string - list: str
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.permutationcollection = {}
        self.permutationcollection["nil"] = list(range(self.dimensionality))
        self.permutationcollection["before"] = sparsevectors.createpermutation(self.dimensionality)
        self.permutationcollection["after"] = sparsevectors.createpermutation(self.dimensionality)
        self.constantdenseness = 10
        self.languagemodel = LanguageModel()
        self.poswindow = 3
        self.changed = False

    def addoperator(self, item):
        self.permutationcollection[item] = sparsevectors.createpermutation(self.dimensionality)
        self.changed = True

    def isoperator(self, item):
        if item in self.permutationcollection:
            return True
        else:
            return False

    def useoperator(self, vector, operator):
        if not self.isoperator(operator):
            self.addoperator(operator)
        p = self.permutationcollection[operator]
        newvec = sparsevectors.permute(vector, p)
        return newvec

    def addconstant(self, item):
        self.changed = True
        self.additem(item,
                     sparsevectors.newrandomvector(self.dimensionality,
                                                   self.dimensionality // self.constantdenseness))

    def observe(self, word, update=True, tag=None, loglevel=False):
        """

        :rtype: object
        """
        if not self.contains(word):
            self.additem(word, None, tag)
            logger("'" + str(word) + "' is new and now introduced: " + str(self.indexspace[word]), loglevel)
        if update:
            self.languagemodel.observe(word)

    def additem(self, item, vector=None, tag=None):
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
        self.languagemodel.additem(item)
        self.changed = True
        self.tag[item] = tag
        if tag not in self.tagged:
            self.tagged[tag] = []
        self.tagged[tag].append(item)

    def addintoitem(self, item, otheritem, weight=1, permutation=None):
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
        if permutation is None:
            vector = self.indexspace[otheritem]
        else:
            vector = sparsevectors.permute(self.indexspace[otheritem],
                                           self.permutationcollection[permutation])
        self.contextspace[item] = sparsevectors.sparseadd(self.contextspace[item],
                                                          sparsevectors.normalise(vector),
                                                          weight)
        self.changed = True

    def observecollocation(self, item, otheritem, operator="nil"):
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
            self.languagemodel.removeitem(item)
            self.changed = True

    def reducewordspace(self, threshold=1):
        items = list(self.indexspace.keys())
        for item in items:
            if self.languagemodel.globalfrequency[item] <= threshold:
                self.removeitem(item)
                self.changed = True

    def outputwordspace(self, filename):
        """
        Save wordspace to disk.
        :param filename: str
        """
        try:
            with open(filename, 'wb') as outfile:
                itemj = {}
                itemj["dimensionality"] = self.dimensionality
                itemj["densenss"] = self.denseness
                itemj["poswindow"] = self.poswindow
                itemj["constantdensenss"] = self.constantdenseness
                itemj["indexspace"] = self.indexspace
                itemj["contextspace"] = self.contextspace
                itemj["permutationcollection"] = self.permutationcollection
                itemj["languagemodel"] = self.languagemodel
                pickle.dump(itemj, outfile)
        except IOError:
                logger("Could not write >>" + filename + ".toto <<", error)

    def inputwordspace(self, vectorfile):
        try:
            cannedspace = open(vectorfile, 'rb')
            itemj = pickle.load(cannedspace)
            self.dimensionality = itemj["dimensionality"]
            self.denseness = itemj["densenss"]
            self.poswindow = itemj["poswindow"]
            self.constantdenseness = itemj["constantdensenss"]
            self.indexspace = itemj["indexspace"]
            self.contextspace = itemj["contextspace"]
            self.permutationcollection = itemj["permutationcollection"]
            self.languagemodel = itemj["languagemodel"]
        except IOError:
            logger("Could not read from >>" + vectorfile + "<<", error)

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
                          filtertag: bool=False, threshold: int=-1)->list:
        '''
        Return the items from the contextspace most similar to the given item. I.e. items which have similar
        neighbours to the item.
        '''
        neighbourhood = {}
        if filtertag:
            targetset = self.tagged[self.tag[item]]
        else:
            targetset = self.contextspace
        for i in targetset:  # was: for i in self.contextspace:
            if i == item:
                continue
#            if filtertag:
#                if self.tag[i] != self.tag[item]:
#                    continue
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
        n = {}
        for i in self.contextspace:
            n[i] = sparsevectors.sparsecosine(sparsevectors.permute(self.indexspace[item], permutation),
                                              self.contextspace[i])
        if weights:
            r = sorted(n.items(), key=lambda k: n[k[0]], reverse=True)[:number]
        else:
            r = sorted(n, key=lambda k: n[k], reverse=True)[:number]
        return r
