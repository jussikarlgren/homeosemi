from logger import logger
import hyperdimensionalsemanticspace
from nltk import word_tokenize
from nltk import sent_tokenize
import languagemodel
# ===========================================================================
# filename = "/home/jussi/data/alice_adventures_one_par_per_line.txt"
filename = "/home/jussi/data/news/news.txt"
datadirectory = "/home/jussi/data/vectorspace/"
# ===========================================================================
debug = False
monitor = True
error = True
dimensionality = 2000
denseness = 10
ngramwindow = 3
# ===========================================================================
languagemodel = languagemodel.LanguageModel()
languagemodel.importstats(datadirectory + "bgwordfrequency.list")  # insert file name here
# ===========================================================================
# files = simpletextfilereader.getfilelist(datadirectory, re.compile(r".*09*.i*"))

cspace = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)

cspace.addoperator("before")
cspace.addoperator("after")

dspace = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)


def weight(item: str):
    return languagemodel.frequencyweight(item, False)


def trainusingtext(text: str, window: int=2):
    ticker = 0
    sentences = sent_tokenize(text.lower())  # type: str
    for sentence in sentences:
        ticker += 1
        logger("{}\t{}".format(ticker, sentence), monitor)
        ii = 0
        words = word_tokenize(sentence)
        if len(words) > 2:
            for word in words:
                languagemodel.observe(word)
                ii += 1
                dspace.observe(word)
                dspace.additemintoitem(word, sentence)
                lhs = words[ii - window:ii]
                rhs = words[ii + 1:ii + window + 1]
                for lw in lhs:
                    w = weight(lw)
                    cspace.additemintoitem(word, lw, w, "before")
                for rw in rhs:
                    w = weight(rw)
                    cspace.additemintoitem(word, rw, w, "after")
            dspace.removeitem(sentence)  # not necessary after being used the once
            if ticker % 100 == 0:
                if languagemodel.changed:
                    languagemodel.save()
                if dspace.changed:
                    dspace.outputwordspace(datadirectory + "documentspace.hyp")
                if cspace.changed:
                    cspace.outputwordspace(datadirectory + "ctxspace.hyp")
                cspace.comb()
                dspace.comb()


with open(filename, "r+") as inputtextfile:
    for line in inputtextfile:
        if len(line) > 2:
            logger(line, debug)
            trainusingtext(line)
