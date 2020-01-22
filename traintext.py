import hyperdimensionalsemanticspace
from nltk import word_tokenize
from nltk import sent_tokenize
from languagemodel import LanguageModel


#  "/some/path/to/resources/"
#  "term-tab-frequency-list.file"
def uselanguagemodel(resourcedirectory:str="/home/jussi/data", resourcefile:str="enfreqfilteredmin10.list"):
    languagemodel = LanguageModel()
    languagemodel.importstats(resourcedirectory + "/" + resourcefile)
    return languagemodel


def weight(item: str):
    return languagemodel.frequencyweight(item, False)


def trainusingtext(text: str, window: int=2):
    cspace = hyperdimensionalsemanticspace.SemanticSpace()
    cspace.addoperator("before")
    cspace.addoperator("after")
    dspace = hyperdimensionalsemanticspace.SemanticSpace()
    sentences = sent_tokenize(text.lower())
    for sentence in sentences:
        ii = 0
        words = word_tokenize(sentence)
        for word in words:
            ii += 1
            dspace.observe(word)
            dspace.addintoitem(word, sentence)
            lhs = words[ii - window:ii]
            rhs = words[ii + 1:ii + window + 1]
            for lw in lhs:
                w = weight(lw)
                cspace.addintoitem(word, lw, w, "before")
            for rw in rhs:
                w = weight(rw)
                cspace.addintoitem(word, rw, w, "after")


#cspace.outputwordspace("context.wordspace")