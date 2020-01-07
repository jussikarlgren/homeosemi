import random
import sparsevectors
import pickle
import math
from logger import logger
import nltk

error = True


class StringSequenceSpace:
    def __init__(self, dimensionality=2000, denseness=10, window=5, bin=False):
        self.indexspace = {}
        self.binaryfrequencies = bin
        self.dimensionality = dimensionality
        self.denseness = denseness
        self.window = window
        self.globalfrequency = {}
        self.bign = 0
        self.pospermutations = {}
        self.pospermutations["vector"] = {90: -1,
                                          290: -1,
                                          733: 1,
                                          873: 1,
                                          885: 1,
                                          1268: -1,
                                          1269: 1,
                                          1523: -1,
                                          1569: -1,
                                          1573: 1}
    #            sparsevectors.newrandomvector(dimensionality,denseness)

    def importstats(self, wordstatsfile, loglevel=False):
        logger(wordstatsfile, loglevel)
        with open(wordstatsfile, "r") as savedstats:
            i = 0
            try:
                for line in savedstats:
                    i += 1
                    seqstats = line.rstrip().split("\t")
                    try:
                        if not seqstats[0] in self.globalfrequency:
                            self.globalfrequency[seqstats[0]] = int(seqstats[1])
                        self.bign += int(seqstats[1])
                    except IndexError:
                        pass
            except UnicodeDecodeError:
                logger("Error at line " + str(i), error)

    def observe(self, string, loglevel=False):
        if self.window > 0:
            words = [string[ii:ii + self.window] for ii in range(len(string) - self.window + 1)]
        else:
            words = nltk.word_tokenize(string)
        for w in words:
            self.bign += 1
            if w in self.globalfrequency:
                self.globalfrequency[w] += 1
            else:
                self.globalfrequency[w] = 1

    def textvector(self, string, frequencyweighting=True, loglevel=False):
        uvector = sparsevectors.newemptyvector(self.dimensionality)
        if self.window > 0:
            windows = [string[ii:ii + self.window] for ii in range(len(string) - self.window + 1)]
            for sequence in windows:
                thisvector = self.makevector(sequence)
                if frequencyweighting:
                    factor = self.frequencyweight(sequence)
                else:
                    factor = 1
                logger(sequence + " " + str(factor), loglevel)
                if loglevel:
                    logger(str(sparsevectors.sparsecosine(uvector, sparsevectors.normalise(thisvector))), loglevel)
                uvector = sparsevectors.sparseadd(uvector, sparsevectors.normalise(thisvector), factor)
        else:
            words = nltk.word_tokenize(string)
            if self.binaryfrequencies:
                wordlist = set(words)  # not a list, a set but hey
            else:
                wordlist = words
            for w in wordlist:
                if frequencyweighting:
                    factor = self.frequencyweight(w)
                else:
                    factor = 1
                if w not in self.indexspace:
                    self.additem(w)
                else:
                    self.observe(w)
                uvector = sparsevectors.sparseadd(uvector, sparsevectors.normalise(self.indexspace[w]), factor)
        return uvector

    def additem(self, item):
        self.indexspace[item] = sparsevectors.newrandomvector(self.dimensionality, self.denseness)
        self.globalfrequency[item] = 1
        self.bign += 1

    def saveelementspace(self, filename):
        with open(filename, "wb") as outfile:
            pickle.dump(self.indexspace, outfile)

    def importelementspace(self, indexvectorfile):
        self.indexspace = pickle.load(open(indexvectorfile, "rb"))

    def makevector(self, string):
        stringvector = {}  # np.array([0] * self.dimensionality)
        for character in string[::-1]:    # reverse the string! (to keep strings that share prefixes similar)
            if character not in self.indexspace:
                vec = {}
                nonzeros = random.sample(list(range(self.dimensionality)), self.denseness)
                random.shuffle(nonzeros)
                split = self.denseness // 2
                for i in nonzeros[:split]:
                    vec[i] = 1
                for i in nonzeros[split:]:
                    vec[i] = -1
                self.indexspace[character] = vec
                self.globalfrequency[character] = 1
                self.bign += 1
            stringvector = sparsevectors.sparseadd(sparsevectors.sparseshift(stringvector, self.dimensionality),
                                                   self.indexspace[character])
            # np.append(stringvector[1:], stringvector[0]) + self.indexspace[character]
        return stringvector  # lil_matrix(stringvector.reshape(self.dimensionality, -1))

    def frequencyweight(self, word):
        l = 500
        try:
            w = math.exp(-l * self.globalfrequency[word] / self.bign)
            # w = math.exp(-300 * math.pi * int(self.globalfrequency[word]) / self.bign)
        except KeyError:
            w = 0.5
        return w

    def getvector(self, word, loglevel=False):
        if word not in self.indexspace:
            logger("Made new vector for " + word, loglevel)
            self.additem(word)
        else:
            self.observe(word, loglevel)
        return self.indexspace[word]

    def postriplevector(self, text, poswindow=3):
        poses = nltk.pos_tag(text)
        windows = [poses[ii:ii + poswindow] for ii in range(len(poses) - poswindow + 1 + 2)]
        onevector = self.pospermutations["vector"]
        vector = sparsevectors.newemptyvector(self.dimensionality)
        for sequence in windows:
            for item in sequence:
                if item[1] not in self.pospermutations:
                    self.pospermutations[item[1]] = sparsevectors.createpermutation(self.dimensionality)
                onevector = sparsevectors.permute(onevector, self.pospermutations[item[1]])
            vector = sparsevectors.sparseadd(vector, onevector)
        return vector

    def savepospermutations(self, filename):
        with open(filename, "wb") as outfile:
            pickle.dump(self.pospermutations, outfile)

    def importpospermutations(self, filename):
        self.pospermutations = pickle.load(open(filename, "rb"))

    def savefrequencies(self, filename):
        with open(filename, "wb") as outfile:
            pickle.dump(self.globalfrequency, outfile)

    def importfrequencies(self, filename):
        self.globalfrequency = pickle.load(open(filename, "rb"))
        for i in self.globalfrequency:
            self.bign += self.globalfrequency[i]
