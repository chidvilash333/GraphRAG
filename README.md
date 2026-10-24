# GraphRAG

Knowledge-graph + vector retrieval for multi-hop questions. Extends "Ask My Docs".
Full README (architecture, thesis, limitations) arrives in Phase 6.

## Status
- Phase 1: LLM-based (subject, relation, object) triple extraction with strict validation.
- Phase 2: ingest pipeline: token-aware chunking, MiniLM embeddings in ChromaDB,
  cached extraction, NetworkX knowledge graph linked to chunks by chunk ID.
- Phase 3: hybrid retrieval: question entities -> bounded graph traversal + vector search, fused with RRF.  


The sample corpus in `data/docs/` is fictional on purpose; see `data/README.md`.