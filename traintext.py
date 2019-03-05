import hyperdimensionalsemanticspace
from nltk import word_tokenize
from nltk import sent_tokenize
import languagemodel

resourcedirectory = "/some/path/to/resources/"    # configure this appropriately
languagemodel = languagemodel.LanguageModel()
languagemodel.importstats(resourcedirectory + "term-tab-frequency-list.file")  # insert file name here

cspace = hyperdimensionalsemanticspace.SemanticSpace()
cspace.addoperator("before")
cspace.addoperator("after")

dspace = hyperdimensionalsemanticspace.SemanticSpace()


def weight(item:str):
    return languagemodel.frequencyweight(item, False)


def trainusingtext(text:str, window:int=2):
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
#dspace.outputwordspace("sentence.wordspace")