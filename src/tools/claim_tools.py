"""
Tool definitions for Agent Reasoning (A1).
Provides:
1. `corpus_search`: Semantic and keyword retrieval over the CAR and Charter corpus
2. `lookup_clause_by_id`: Deterministic lookup of specific regulatory clauses (e.g. CAR-3.3.2)
3. `get_claim_field`: Access to parsed claim attributes and entities
"""

import time
from typing import Dict, Any, List, Optional
from src.core.contracts import PassengerClaimRequest
from src.ingestion.vector_store import RegulatoryVectorStore


class ClaimToolBox:
    def __init__(self, claim_request: PassengerClaimRequest, vector_store: RegulatoryVectorStore):
        self.claim = claim_request
        self.vstore = vector_store
        self.tool_call_history: List[Dict[str, Any]] = []

    def corpus_search(self, query: str, top_k: int = 4, category: Optional[str] = None,
                      strategy: str = "dense") -> List[Dict[str, Any]]:
        """
        Tool 1: Search across the regulatory and charter corpus (A1).
        The agent chooses whether, when and with what query to search.
        strategy: "dense" (FAISS semantic), "keyword" (lexical) or "hybrid" (both, merged).
        """
        start = time.time()
        if strategy == "keyword":
            results = self.vstore.keyword_search(query=query, top_k=top_k)
        else:
            results = self.vstore.search(query=query, top_k=top_k, category=category)
            if strategy == "hybrid":
                seen = {r["chunk_id"] for r in results}
                results += [r for r in self.vstore.keyword_search(query=query, top_k=top_k) if r["chunk_id"] not in seen]
        duration = time.time() - start

        self.tool_call_history.append({
            "tool": "corpus_search",
            "parameters": {"query": query, "top_k": top_k, "category": category, "strategy": strategy},
            "results_count": len(results),
            "passages": [{"clause_id": r["clause_id"], "score": r.get("score")} for r in results],
            "duration_seconds": round(duration, 4),
        })
        return results

    def lookup_clause_by_id(self, clause_id: str) -> Optional[Dict[str, Any]]:
        """
        Tool 2: Direct lookup of a specific regulatory clause by its canonical ID (A1).
        E.g. 'CAR-3.3.2', 'CAR-3.3.4', 'CAR-3.4.1', 'CAR-3.7', 'INDIGO-3.2'
        """
        start = time.time()
        result = self.vstore.lookup_clause(clause_id=clause_id)
        duration = time.time() - start

        self.tool_call_history.append({
            "tool": "lookup_clause_by_id",
            "parameters": {"clause_id": clause_id},
            "found": result is not None,
            "passages": [{"clause_id": result["clause_id"], "score": 1.0}] if result else [],
            "duration_seconds": round(duration, 4),
        })
        return result

    def get_claim_field(self, field_name: str) -> Any:
        """
        Tool 3: Inspect specific fields and metadata of the passenger claim request (A1).
        """
        start = time.time()
        field_name_clean = field_name.strip().lower()
        claim_dict = self.claim.model_dump()

        val = claim_dict.get(field_name_clean, None)
        duration = time.time() - start

        self.tool_call_history.append({
            "tool": "get_claim_field",
            "parameters": {"field_name": field_name},
            "found": val is not None,
            "value": str(val) if val is not None else None,
            "duration_seconds": round(duration, 4),
        })
        return val

    def get_total_calls(self) -> int:
        return len(self.tool_call_history)
