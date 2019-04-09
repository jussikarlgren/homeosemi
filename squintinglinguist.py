from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
import nltk
nltk.download('averaged_perceptron_tagger')

import re
from lexicalfeatures import lexicon
from logger import logger
import semanticroles

urlpatternexpression = re.compile(r"https?://[/A-Za-z0-9\.\-\?_]+", re.IGNORECASE)
handlepattern = re.compile(r"@[A-Za-z0-9_\-±.]+", re.IGNORECASE)
verbtags = ["VB", "VBZ", "VBP", "VBN", "VBD", "VBG"]
adjectivetags = ["JJ", "JJR", "JJS"]


def restartCoreNlpClient():
    semanticroles.restartCoreNlpClient()


def generalise(text, handlesandurls=True, nouns=True, verbs=True, adjectives=True, adverbs=False):
    accumulator = []
    if handlesandurls:
        text = urlpatternexpression.sub("U", text)
        text = handlepattern.sub("H", text)
    sents = sent_tokenize(text)
    for sentence in sents:
        words = word_tokenize(sentence)
        poses = pos_tag(words)
        for item in poses:
            if nouns and item[1] == "NN":
                accumulator.append("N")
            elif nouns and item[1] == "NNS":
                accumulator.append("Ns")
            elif adjectives and item[1] in adjectivetags:
                accumulator.append(item[1])
            elif verbs and item[1] in verbtags:
                tag = item[1]
                if tag == "VBZ":
                    tag = "VBP"  #  neutralise for 3d present -- VBP is present
                accumulator.append(tag)
            elif adverbs and item[1] == "RB":
                accumulator.append("R")
            else:
                accumulator.append(item[0])
    return " ".join(accumulator)


# do  MD (modal) (separate out 'not' from RB)

#def featurise_sentence(sentence, loglevel=False):
#    features = []
##    words = tokenise(sentence)
#    for word in words:
#        for feature in lexicon:
#            if word.lower() in lexicon[feature]:
#                features.append("JiK" + feature)
#    logger(sentence + "->" + str(features), loglevel)
#    return features

def tokenise(text):
    return word_tokenize(text)

def postags(string):
    return [t[1] for t in pos_tag(word_tokenize(string))]

def window(text, window=2, direction=True):
    return False

def featurise(text, loglevel=False):
    returnfeatures = {}
    features = []
    words = []
    sents = sent_tokenize(text)
    for sentence in sents:
        words = tokenise(sentence)
        for word in words:
            for feature in lexicon:
                if word.lower() in lexicon[feature]:
                    features.append("JiK" + feature)
        returnfeatures = semanticroles.semanticdependencyparse(text)
        returnfeatures["features"] += features
        poses = postags(text)
        returnfeatures["pos"] = poses
    returnfeatures["words"] = words
    logger(text + "->" + str(features), loglevel)
    return returnfeatures


def mildpositems(string, full=False):
    leaveintags = ["IN", "DT", "MD", "PRP", "PRP$", "POS", "CC", "EX", "PDT", "RP", "TO", "WP", "WP$", "WDT", "WRB"]
    words = tokenise(string)
    poses = pos_tag(words)
    if not full:
        returnposes = [("START", "START")]
        for p in poses:
            if p[1] in leaveintags:
                returnposes.append((p[1],p[0]))
            else:
                returnposes.append(p)
        returnposes.append(("END","END"))
    else:
        returnposes = [("START", "BEG")] + poses + [("END", "END")]
    return returnposes

