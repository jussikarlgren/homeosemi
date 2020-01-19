import re
#import semanticroles
from lexicalfeatures import lexicon
from logger import logger
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
#nltk.download('averaged_perceptron_tagger')


urlpatternexpression = re.compile(r"https?://[/A-Za-z0-9\.\-\?_]+", re.IGNORECASE)
handlepattern = re.compile(r"@[A-Za-z0-9_\-±.]+", re.IGNORECASE)
verbtags = ["VB", "VBZ", "VBP", "VBN", "VBD", "VBG"]
adjectivetags = ["JJ", "JJR", "JJS"]


#def restartCoreNlpClient():
#    semanticroles.restartCoreNlpClient()


def sentence_list(text):
    sents = sent_tokenize(text)
    r = []
    for s in sents:
        r.append(word_tokenize(s))
    return r


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
        returnfeatures["features"] = []
        returnfeatures["roles"] = []
#        returnfeatures = semanticroles.semanticdependencyparse(text)
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


class Sentence:
    '''Holds information about one sentence. Maybe should be one clause instead tho.'''
    def __init__(self, sentence:list):
        self.topic = "topic"
        self.comment = "comment"
        self.given = "given"
        self.new = "new"
        self.theme = "theme"
        self.rheme = "rheme"
        self.tempus = "PRESENT"  # PRESENT PAST FUTURE
        self.mood = "TRUE"  #  true, potential, optative
        self.aspect = "STATE"  # state perfect imperfect habitual
        self.speechact = "INDICATIVE"  # indicative imperative question
        self.sentence = sentence
        self.deps = []

    def process(self):
        utterances = semanticroles.semanticdependencyparse(self.sentence)
        for u in utterances:
            self.deps.append(u)
