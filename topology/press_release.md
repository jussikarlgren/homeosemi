# Meaning has a topology, not a geometry

For twenty-five years, distributional semantics has quietly agreed on one thing: that meaning lives in *a* space. One big vector space, a few hundred dimensions, every word a point in it, every pair of words a cosine apart. Random Indexing, LSA, word2vec, the embedding layer at the bottom of every large language model today — different machinery, same underlying commitment. One geometry to hold all of language.

I helped build that world. I now think it is overmodelled.

A single global space forces a number onto every pair of words, whether or not the comparison means anything. It insists that *treacle* and *theorem* have a similarity — a small one, but a real one — when the honest answer is that they live in different corners of experience and the question barely parses. It spends enormous capacity representing distinctions no one needs, and it learns slowly because every observation has to find its place against everything else at once.

We have been building an alternative, and it is beginning to work. Instead of one space, a **basket of balls of yarn**: a coarse global index whose only job is *routing*, over a large collection of small local neighbourhoods, each its own low-dimensional semantic space of ten or twenty dimensions. Words settle into balls by the company they keep. Similarity is answered *inside* a ball. Across balls, the model returns **null** — not a small cosine, but a refusal. "These two things are not in the same neighbourhood; the question does not apply."

That refusal is the whole point. It is not a failure mode to be smoothed over; it is information.

Our first prototype, trained on a handful of novels, already shows the shape of the idea. Content words organise themselves into coherent neighbourhoods — the physical transitive verbs cluster apart from the verbs of speech; body parts sit together; the common nouns of persons — *man, woman, child, gentleman* — form one ball while the abstractions form another. Function words — pronouns, auxiliaries, determiners — segregate into their own structural class and, crucially, stop acting as false bridges between content clusters, a problem that has haunted these models for decades. And when we train across several books at once, the proper names from different worlds — Holmes, Dracula, Frankenstein's creature — correctly *do not* compare. They share no context, so the model declines to place them on the same scale. A monolithic space would have handed you a confident, meaningless number. Ours hands you a null, which is the truth.

None of this required a large model or a large corpus. That is the wager we are now setting out to test in earnest: that a topology of small local spaces learns faster and from less data than a single global one, because each neighbourhood only has to model itself.

We are staking out this ground deliberately. The field's default — one space, one geometry, one number for every pair — is a modelling choice, not a law of nature. Meaning is local, plural, and sometimes incommensurable. It deserves a representation that admits as much.

The map of meaning is not a plane. It is a basket of balls of yarn.

*— work in progress; benchmarks and code to follow.*
