"""Extraction prompt. Kept short on purpose: every token counts against Groq's TPM limit."""

_SYSTEM_TEMPLATE = """You extract a knowledge graph from text.
Return ONLY a JSON object of the form:
{"triples": [{"subject": "...", "relation": "...", "object": "..."}]}

Rules:
1. Extract only relationships explicitly stated in the text. Never use outside knowledge.
2. subject and object are named entities (people, organizations, drugs, genes, proteins,
   places, products, specific concepts), written exactly as they appear in the text.
   Never use pronouns like "it" or "they"; use the full name the pronoun refers to.
3. relation is a short lowercase verb phrase of 1-4 words, e.g. "inhibits", "founded",
   "is located in". Keep the direction correct: subject does relation to object.
4. Skip opinions, hypotheticals, and negated statements.
5. Return at most __MAX__ triples. If there are none, return {"triples": []}.

Example
Text: "Aspirin inhibits COX-1. It was first synthesized by Bayer."
Output: {"triples": [{"subject": "Aspirin", "relation": "inhibits", "object": "COX-1"},
{"subject": "Bayer", "relation": "synthesized", "object": "Aspirin"}]}"""


def build_system_prompt(max_triples: int) -> str:
    # str.replace, not str.format: the template contains literal JSON braces.
    return _SYSTEM_TEMPLATE.replace("__MAX__", str(max_triples))


def build_user_prompt(chunk: str) -> str:
    # Delimiters mark where untrusted document text begins and ends.
    return f"Text:\n<<<\n{chunk}\n>>>"
