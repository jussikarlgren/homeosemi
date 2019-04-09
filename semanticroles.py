from logger import logger
import os
os.environ["CORENLP_HOME"] = "/usr/share/stanford-corenlp-full/"
from corenlp import CoreNLPClient, TimeoutException

parser_client = CoreNLPClient(
    annotators=["tokenize ssplit pos lemma depparse"],  # natlog, ner, parse, coref
    timeout=30000,
    be_quiet=True)  # natlog # memory="16G"

#  past,  3psgpresent, past part, present, base, gerund/present participle
verbposes = ["VBD", "VBZ", "VBN", "VBP", "VB", "VBG"]

tag = "JiK"
error = True  # for logger

def semanticdependencyparse(sentence, loglevel=False):
    utterances = []
    try:
        depgraph = parser_client.annotate(sentence)
        for ss in depgraph.sentence:
            utterances.append(processdependencies(ss, loglevel))
    except TimeoutException:
        logger("Time out for corenlp processing '" + sentence + "'", True)
        returgods = {}
        returgods["roles"] = []
        returgods["features"] = []
        utterances.append(returgods)
    return utterances[0]

def restartCoreNlpClient():
    global parser_client
    parser_client.stop()
    parser_client.start()

def processdependencies(ss, loglevel=False):
    roles = {}
    string = []
    deps = []
    negation = False
    adverbial = []
    mainverb = None
    verbchain = []
    tense = None
    subject = None
    mode = None
    aspect = None
    type = None  # question, indicative, imperative, subjunctive ...
    logger("root: " + str(ss.basicDependencies.root), loglevel)
    i = 1
    try:
        for w in ss.token:
            string.append(w.lemma)
            logger(str(i) + "\t" + w.lemma + " " + w.pos, loglevel)
            i += 1
        for e in ss.basicDependencies.edge:
            dd = str(e.source) + " " +  ss.token[e.source - 1].lemma + "-" + e.dep + "->" +\
                 " " + str(e.target) + " " + ss.token[e.target - 1].lemma
            deps.append(dd)
            logger(dd, loglevel)
        sentenceitems = {}
        sentencepos = {}
        scratch = {}
        npweight = {}
        scratch["aux"] = []
        root = ss.basicDependencies.root[0]  # only one root for now fix this!
        i = 1
        for w in ss.token:
            sentenceitems[i] = w.lemma
            sentencepos[i] = w.pos
            scratch[i] = False
            i += 1
        tense = "PRESENT"
        if sentencepos[root] == "VBD":
            tense = "PAST"
        if sentencepos[root] == "VBN":
            tense = "PAST"

        for edge in ss.basicDependencies.edge:
            logger(str(edge.source) + " " + sentenceitems[edge.source] +
                   " " + "-" + " " + edge.dep + " " + "->" + " " +
                   str(edge.target) + " " + sentenceitems[edge.target], loglevel)
            if edge.dep == 'neg': # and sentencepos[edge.source] in verbposes:
                negation = True
            elif edge.dep == 'advmod':
                if edge.source == root:
                    adverbial.append(edge.target)
            elif edge.dep == 'nsubj':
                subject = edge.target
            elif edge.dep == 'amod' or edge.dep == "compound":
                if edge.target in npweight:
                    npweight[edge.target] += 1
                else:
                    npweight[edge.target] = 1
            elif edge.dep == 'auxpass':
                if sentenceitems[edge.target] == "be":
                    scratch['aux'].append("be")
                    mode = "PASSIVE"
            elif edge.dep == 'aux':
                if sentenceitems[edge.target] == "have":
                    scratch['aux'].append("have")
                if sentenceitems[edge.target] == "do":
                    scratch['aux'].append("do")
                if sentenceitems[edge.target] == "be":
                    scratch['aux'].append("be")
                    if sentencepos[edge.source] in verbposes:
                        tense = "PROGRESSIVE"

                if sentenceitems[edge.target] == "can":
                    scratch['aux'].append("can")
                if sentenceitems[edge.target] == "could":
                    scratch['aux'].append("could")
                if sentenceitems[edge.target] == "would":
                    scratch['aux'].append("would")
                if sentenceitems[edge.target] == "should":
                    scratch['aux'].append("should")
                if sentencepos[edge.target] == "VBD":
                    tense = "PAST"
                if sentenceitems[edge.target] == "will":
                    scratch['aux'].append("will")
                if sentenceitems[edge.target] == "shall":
                    scratch['aux'].append("shall")
        try:
            if sentencepos[root] == "VB":
                if 'aux' in scratch:
                    if "will" in scratch['aux'] or "shall" in scratch['aux']:
                        tense = "FUTURE"
        except KeyError:
            logger("tense situation in " + string, True)
        features = []
        if root > len(ss.token) / 2:
            features.append(tag + "VERYLATEMAINV")
        elif root > len(ss.token) / 3:
            features.append(tag + "LATEMAINV")
        else:
            features.append(tag + "EARLYMAINV")
        if 'aux' in scratch:
            for aa in scratch['aux']:
                features.append(tag + aa)
        if mode:
            features.append(tag + mode)
        if tense:
            features.append(tag + tense)
        if negation:
            features.append(tag + "NEGATION")
        for np in npweight:
            if npweight[np] > 2:
                features.append(tag + "HEAVYNP")
        if root in sentenceitems:
            roles["verb"] = [sentenceitems[root]]
        if subject:
            if subject in sentenceitems:
                roles["subject"] = [sentenceitems[subject]]
            if sentenceitems[subject] == "I":
                features.append(tag + "p1sgsubj")
            if sentenceitems[subject] == "we":
                features.append(tag + "p1plsubj")
            if sentenceitems[subject] == "you":
                features.append(tag + "p2subj")
            #        logger(str(features) + "\t" + str(string) + "\t" + str(deps), True)
        if len(adverbial) > 0:
            roles["adverbial"] = []
            for a in adverbial:
                roles["adverbial"].append(sentenceitems[a])
    except IndexError:
        logger("Index Error processing {}".format(ss), error)
    returgods = {}
    returgods["roles"] = roles
    returgods["features"] = features
    return returgods
