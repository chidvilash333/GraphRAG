# Sample corpus

Everything in `docs/` is **fictional** (drugs, enzymes, companies, places). That is
deliberate: the LLM cannot answer questions about this corpus from its training data,
so any correct multi-hop answer must come from retrieval. With real facts (e.g.
fluconazole/warfarin), a vector-only baseline could "succeed" from memorized knowledge
and hide the retrieval failure we want to demonstrate.

Facts are split so multi-hop questions require joining documents, e.g.
"Which drug could raise Zoraxin levels?" needs zoraxin.md (cleared by Velkase-3) and
tremidol.md (inhibits Velkase-3); tremidol.md never mentions Zoraxin.
`brimlase.md` is a distractor: similar vocabulary, wrong answer.