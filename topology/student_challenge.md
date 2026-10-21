# An open problem, and an invitation

Most of what you will be taught about representing meaning is settled and works. That is exactly why it is the wrong place to spend a career. I want to point you at something that is *not* settled, that most of the field has stopped questioning, and that is wide open for someone willing to rethink a default.

Here is the default: to a computer, a word is a point in a single large space, and the meaning of any two words is how close those points are. It is a beautiful idea and it powers most of modern language technology. It also quietly assumes that *everything is comparable to everything* — that there is always a meaningful number for how similar any two words are. Spend a moment with that and it starts to feel wrong. What is the similarity between *treacle* and *theorem*? The system will give you an answer. The honest answer is that the question hardly makes sense.

We have been building a model that is allowed to say so. Instead of one space, it keeps many small ones — local neighbourhoods of meaning, each modelling only its own patch of the world — and when you ask it to compare two words that live in different neighbourhoods, it answers **null**. Not zero. Not a tiny cosine. *"Out of scope."* Teaching a model when to decline to answer turns out to be as important as teaching it to answer well.

The prototype works well enough to be interesting and badly enough to be exciting — which is the best state a research problem can be in. Here is some of what is still open, any one of which is a real project:

- **When should a neighbourhood divide?** Right now we split a ball when it grows too big. But size is a crude proxy. A ball should divide when it becomes *incoherent* — when it is quietly holding two ideas that want to come apart. Detecting that, and doing it at the right moment, is an unsolved and genuinely deep question about the birth and death of concepts.

- **Words belong to more than one world.** *Bank* is a river and a vault. A single point cannot be both; a basket of neighbourhoods could route a word into several balls at once. Making that work — and deciding, in context, which neighbourhood is active — is polysemy handled structurally instead of papered over.

- **Meaning is bigger than words.** Everything above is about single words. Sentences, constructions, whole utterances — do they get neighbourhoods too? What is a "ball" for a grammatical pattern? Nobody has a good answer.

- **The economy claim.** We believe many small spaces learn faster and from less data than one giant one. *Believe* is not *shown*. Designing the experiment that would actually confirm or kill that claim — against real benchmarks, honestly — is worth more than a dozen new features.

- **Knowing what you don't know.** A system that returns null in the right places is a system with a kind of epistemic humility. When *should* a language model refuse? That question reaches far past this project, into everything we are nervous about in AI right now.

You do not need permission or a large machine to work on any of this. The whole thing runs on a laptop over a few books. What it needs is someone who is not yet convinced that the current way is the only way — who can look at a default the whole field accepts and ask, plainly, *why like this and not otherwise?*

That question is the entire job. If it itches at you, come and pull on one of these threads. The basket has plenty of loose ends, and that is the good news.
