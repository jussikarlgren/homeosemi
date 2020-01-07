import hyperdimensionalsemanticspace
import languagemodel
datadirectory = "/home/jussi/data/vectorspace/"
dimensionality = 2000
denseness = 10
languagemodel = languagemodel.LanguageModel()
languagemodel.importstats(datadirectory + "bgwordfrequency.list")  # insert file name here
cspace = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)
dspace = hyperdimensionalsemanticspace.SemanticSpace(dimensionality, denseness)
cspace.inputwordspace(datadirectory + "ctxspace.hyp")
dspace.inputwordspace(datadirectory + "documentspace.hyp")
cspace.comb()

i = 0
for item in cspace.items():
    if languagemodel.observedfrequency(item) > 10:
        i += 1
        ns = cspace.contextneighbours(item,5,True)
        print(item, languagemodel.observedfrequency(item), ns, sep="\t")
print(i)

#cspace.indextocontextneighbours()

