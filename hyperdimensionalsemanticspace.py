import sparsevectors
import math
import pickle
from languagemodel import LanguageModel
import re

from logger import logger  # Simplest possible logger, replace with any variant of your choice.
error = True      # loglevel
debug = False     # loglevel
monitor = False   # loglevel


def inputwordspace(vectorfile):
    try:
        cannedspace = open(vectorfile, 'rb')
        wordspace = pickle.load(cannedspace)
        return wordspace
    except IOError:
        logger("Could not read from >>" + vectorfile + "<<", error)
        return SemanticSpace()


def similarity(vector, anothervector):
    return sparsevectors.sparsecosine(vector, anothervector)


class SemanticSpace:
    def __init__(self, dimensionality: int=2000, denseness: int=10, name: str="no name"):
        self.name = name
        self.indexspace = {}    # dict: string - sparse vector
        self.contextspace = {}  # dict: string - denser vector
        self.tag = {}           # dict: string - string
        self.tagged = {}        # dict: string - list: str
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.permutationcollection = {"nil": list(range(self.dimensionality)),
                                      "before": sparsevectors.createpermutation(self.dimensionality),
                                      "after": sparsevectors.createpermutation(self.dimensionality)}
        self.observedfrequency = {}  # dict: string - int
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
        newvec = vector
        if operator:
            if not self.isoperator(operator):
                self.addoperator(operator)
            newvec = sparsevectors.permute(vector, self.permutationcollection[operator])
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
        self.observedfrequency[word] += 1
        if update:
            self.languagemodel.observe(word)

    def observedfrequency(self, item):
        return self.languagemodel.observedfrequency(item)

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
        self.observedfrequency[item] = 0
        if tag not in self.tagged:
            self.tagged[tag] = []
        self.tagged[tag].append(item)

    def additemintoitem(self, item, otheritem, weight=1, permutation=None):
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
        self.addintoitem(item, self.indexspace[otheritem], weight, permutation)

    def addintoitem(self, item, vector, weight=1, operator=None):
        if not self.contains(item):
            self.additem(item)
        if operator is not None:
            vector = sparsevectors.permute(vector,
                                           self.permutationcollection[operator])
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

    def comb(self, k: float = 0.2):
        for item in self.contextspace:
            self.contextspace[item] = sparsevectors.comb(self.contextspace[item], k, self.dimensionality)

    # ================================================================
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

    #
    # def outputwordspace(self, filename):
    #     """
    #     Save wordspace to disk.
    #     :param filename: str
    #     """
    #     try:
    #         with open(filename, 'wb') as outfile:
    #             itemj = {}
    #             itemj["dimensionality"] = self.dimensionality
    #             itemj["densenss"] = self.denseness
    #             itemj["poswindow"] = self.poswindow
    #             itemj["constantdensenss"] = self.constantdenseness
    #             itemj["indexspace"] = self.indexspace
    #             itemj["contextspace"] = self.contextspace
    #             itemj["permutationcollection"] = self.permutationcollection
    #             itemj["languagemodel"] = self.languagemodel
    #             itemj["observedfrequency"] = self.observedfrequency
    #             pickle.dump(itemj, outfile)
    #     except IOError:
    #             logger("Could not write >>" + filename + ".toto <<", error)
    #
    # def inputwordspace(self, vectorfile):
    #     try:
    #         cannedspace = open(vectorfile, 'rb')
    #         itemj = pickle.load(cannedspace)
    #         self.dimensionality = itemj["dimensionality"]
    #         self.denseness = itemj["densenss"]
    #         self.poswindow = itemj["poswindow"]
    #         self.constantdenseness = itemj["constantdensenss"]
    #         self.indexspace = itemj["indexspace"]
    #         self.contextspace = itemj["contextspace"]
    #         self.permutationcollection = itemj["permutationcollection"]
    #         self.languagemodel = itemj["languagemodel"]
    #         try:
    #             self.observedfrequency = itemj["observedfrequency"]
    #         except KeyError:
    #             self.observedfrequency = {}
    #             for knownitem in self.contextspace:
    #                 self.observedfrequency[knownitem] = 1
    #     except IOError:
    #         logger("Could not read from >>" + vectorfile + "<<", error)

    def importgavagaiwordspace(self, vectorfile:str, threshold=5):
        vectorpattern = re.compile(r"\(\"(.*)\" #S(\d+);([\d\+\-\;]+): #S\d+;(.+): (\d+)\)",
                                   re.IGNORECASE)
        itempattern = re.compile(r"(\d+)\+?(\-?[\d\.e\-]+)$")
        antal = 0
        antalkvar = 0
        try:
            with open(vectorfile, 'rt', errors="replace") as gavagaispace:
                for line in gavagaispace:
                    antal += 1
                    vectors = vectorpattern.match(line)
                    if vectors:
                        string = str(vectors.group(1))
                        dim = int(vectors.group(2))
                        idx = vectors.group(3)
                        ctx = vectors.group(4)
                        freq = int(vectors.group(5))
                        if freq > threshold:
                            antalkvar += 1
#                            logger("{} {} {} {} {}".format(antal, antalkvar, string, freq, idx), debug)
                            idxvector = sparsevectors.newemptyvector(dim)
                            idxlist = idx.split(";")
                            for ii in idxlist:
                                try:
                                    item = itempattern.match(ii)
                                    idxvector[int(item.group(1))] = float(item.group(2))
                                except:
                                    logger("{} {} {} {}".format(antal, string, ii, idx), error)
                            self.additem(string, idxvector)
                            ctxvector = sparsevectors.newemptyvector(dim)
                            ctxlist = ctx.split(";")
                            for ii in ctxlist:
                                try:
                                    item = itempattern.match(ii)
                                    ctxvector[int(item.group(1))] = float(item.group(2))
                                except:
                                    logger("{} {} {} {}".format(antal, string, ii, idx), error)
                            self.contextspace[string] = ctxvector
                            self.observedfrequency[string] = freq
                            self.languagemodel.additem(string, freq)
        except IOError:
            logger("Could not read from >>" + vectorfile + "<<", error)

    # ===========================================================================
    # querying the semantic space
    def contains(self, item):
        if item in self.indexspace and item in self.contextspace:
            return True
        else:
            return False

    def items(self):
        return self.indexspace.keys()

    def contextsimilarity(self, item, anotheritem):
        return sparsevectors.sparsecosine(self.contextspace[item], self.contextspace[anotheritem])

    def indextocontextsimilarity(self, item, anotheritem):
        if self.contains(item):
            return sparsevectors.sparsecosine(self.indexspace[item], self.contextspace[anotheritem])
        else:
            return 0.0

    def contextneighbours(self, item: str, number: int=10, weights: bool=False,
                          filtertag: bool=False, threshold: int=-1) -> list:
        """
        Return the items from the contextspace most similar to the given item. I.e. items which have similar
        neighbours to the item. Specify number of items desired (0 will give all), if weights are desired, if
        only items with the same tag are desired, and if thresholding to a certain horizon is desired.
        """
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
            r = sorted(neighbourhood.items(), key=lambda m: neighbourhood[m[0]], reverse=True)[:number]
        else:
            r = sorted(neighbourhood, key=lambda m: neighbourhood[m], reverse=True)[:number]
        return r

    def contexttoindexneighbours(self, item, number=10, weights=False, permutationname="nil"):
        """
        Return the items whose index vectors are most similar to the given item's context vector. I.e. items which
        have occurred in contexts with the item.
        """
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
