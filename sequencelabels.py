from logger import logger
import sparsevectors
import pickle


#parser = CoreNLPClient(annotators="tokenize ssplit pos lemma depparse".split())

class SequenceLabels:
    def __init__(self, dimensionality=2000, window=3,
                 sequencelabel=None, permutations={}):
        self.window = window
        self.changed = False
        self.dimensionality = dimensionality
        if sequencelabel is None:
            self.sequencelabel = sparsevectors.newrandomvector(dimensionality, dimensionality // 10)
            self.changed = True
        else:
            self.sequencelabel = sequencelabel
        self.permutations = permutations
        self.error = True
        self.debug = False
        self.monitor = False

    def windows(self, sequence):
        windowlist = []
        if self.window > 0:
           windowlist = [sequence[ii:ii + self.window] for ii in range(len(sequence) - self.window + 1)]
        return windowlist

    def onesequencevector(self, subsequence, accumulator=None, loglevel=False):
        if accumulator == None:
            accumulator = self.sequencelabel
        if subsequence == []:
            return accumulator
        else:
            head = subsequence[0]  # type: str
            tail = subsequence[1:]
            if not head in self.permutations:
                self.permutations[head] = sparsevectors.createpermutation(self.dimensionality)
                self.changed = True
            return self.onesequencevector(tail, sparsevectors.permute(accumulator, self.permutations[head]))

    def sequencevector(self, sequence, initialvector=None, loglevel=False):
        if initialvector == None:
            initialvector = sparsevectors.newemptyvector(self.dimensionality)
        windowlist = self.windows(sequence)
        logger(str(windowlist), loglevel)
        for w in windowlist:
            initialvector = sparsevectors.sparseadd(initialvector,
                                                    sparsevectors.normalise(self.onesequencevector(w, None, loglevel)))
        return initialvector


    #================================================================
    # save and restore sequence model
    def save(self,  filename="/home/jussi/data/storm/vectorspace/sequencemodel.hyp"):
        try:
            with open(filename, 'wb') as outfile:
                pickle.dump(self.window, outfile)
                pickle.dump(self.dimensionality, outfile)
                pickle.dump(self.sequencelabel, outfile)
                pickle.dump(self.permutations, outfile)
        except TypeError:
            logger("Could not write to file", True)

    def restore(self, modelfilename):
        try:
            modelfile = open(modelfilename, "rb")
            self.window = pickle.load(modelfile)
            self.dimensionality = pickle.load(modelfile)
            self.sequencelabel = pickle.load(modelfile)
            self.permutations = pickle.load(modelfile)
        except:
            logger("Fix the missing file error, touch file for next cycle, e.g.", True)
            logger("Could not load sequence model", self.error)