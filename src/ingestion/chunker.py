"""
Hierarchical Regulatory Chunker (L1, D2).
Preserves regulatory integrity by chunking at clause and paragraph boundaries,
keeping operative amounts paired with their conditional qualifiers and cross-references.
"""

import re
import hashlib
from typing import List, Dict, Any


class RegulatoryChunk:
    def __init__(
        self,
        chunk_id: str,
        clause_id: str,
        section_title: str,
        document_title: str,
        text: str,
        metadata: Dict[str, Any],
    ):
        self.chunk_id = chunk_id
        self.clause_id = clause_id
        self.section_title = section_title
        self.document_title = document_title
        self.text = text
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "clause_id": self.clause_id,
            "section_title": self.section_title,
            "document_title": self.document_title,
            "text": self.text,
            "metadata": self.metadata,
        }


class RegulatoryCorpusChunker:
    """
    Parses CAR Section 3 Series M Part IV and Airline Charters into semantically complete,
    clause-indexed chunks to prevent false positive retrievals (addressing Section 3 difficulty).
    """

    @staticmethod
    def chunk_dgca_car(text: str) -> List[RegulatoryChunk]:
        chunks: List[RegulatoryChunk] = []
        doc_title = "DGCA CAR Section 3 Series M Part IV (Rev 4)"

        # Regular expression to match major clauses like "2. DEFINITIONS", "3.2 DENIED BOARDING", "3.3 CANCELLATION OF FLIGHTS"
        # and subclauses like "3.3.2", "3.3.4", "3.4.1", etc.
        sections = re.split(r"\n(?=(?:[1-3]\.[0-9]+(?:\.[0-9]+)?|[1-3]\.\s+[A-Z]))", text)

        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            # Identify clause identifier
            match = re.match(r"^([1-3]\.[0-9]+(?:\.[0-9]+)?|[1-3]\.[0-9]+\s+[A-Z\s]+)", sec_clean)
            if match:
                raw_id = match.group(1).strip()
                clause_id = f"CAR-{raw_id.split()[0]}"
            else:
                clause_id = "CAR-GEN"

            first_line = sec_clean.split("\n")[0].strip()
            section_title = first_line

            # Generate content hash for idempotent re-indexing
            chunk_hash = hashlib.sha256(f"{clause_id}:{sec_clean}".encode("utf-8")).hexdigest()[:16]

            # Tag metadata to aid reasoning
            category = "general"
            if "DENIED BOARDING" in sec_clean.upper() or "3.2" in clause_id:
                category = "denied_boarding"
            elif "CANCELLATION" in sec_clean.upper() or "3.3" in clause_id:
                category = "cancellation"
            elif "DELAY" in sec_clean.upper() or "3.4" in clause_id:
                category = "delay"
            elif "DOWNGRADING" in sec_clean.upper() or "3.5" in clause_id:
                category = "downgrading"
            elif "FACILITIES" in sec_clean.upper() or "3.7" in clause_id or "3.8" in clause_id:
                category = "facilities"

            metadata = {
                "source": "DGCA_CAR_M_IV",
                "category": category,
                "has_compensation_scale": "INR" in sec_clean,
                "mentions_alternate_flight": "alternate flight" in sec_clean.lower(),
                "mentions_exemptions": "extraordinary circumstances" in sec_clean.lower() or "not be liable" in sec_clean.lower(),
                "mentions_facilities": "refreshment" in sec_clean.lower() or "meal" in sec_clean.lower() or "hotel" in sec_clean.lower(),
            }

            chunks.append(
                RegulatoryChunk(
                    chunk_id=f"chunk-{chunk_hash}",
                    clause_id=clause_id,
                    section_title=section_title,
                    document_title=doc_title,
                    text=sec_clean,
                    metadata=metadata,
                )
            )

        return chunks

    @staticmethod
    def chunk_airline_charter(text: str) -> List[RegulatoryChunk]:
        chunks: List[RegulatoryChunk] = []
        doc_title = "IndiGo Passenger Charter & Conditions of Carriage"

        sections = re.split(r"\n(?=[0-9]+\.[0-9]*\s+[A-Z])", text)
        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s+([A-Z\s]+)", sec_clean)
            if match:
                raw_id = match.group(1).strip()
                clause_id = f"INDIGO-{raw_id}"
                section_title = match.group(2).strip()
            else:
                clause_id = "INDIGO-GEN"
                section_title = sec_clean.split("\n")[0].strip()

            chunk_hash = hashlib.sha256(f"{clause_id}:{sec_clean}".encode("utf-8")).hexdigest()[:16]

            metadata = {
                "source": "INDIGO_CHARTER",
                "category": "airline_policy",
                "mentions_alternate_flight": "alternate flight" in sec_clean.lower() or "plan b" in sec_clean.lower(),
                "mentions_contact_details": "contact" in sec_clean.lower() or "mobile" in sec_clean.lower(),
                "mentions_time_limit": "30 calendar days" in sec_clean.lower() or "time limit" in sec_clean.lower(),
            }

            chunks.append(
                RegulatoryChunk(
                    chunk_id=f"chunk-{chunk_hash}",
                    clause_id=clause_id,
                    section_title=section_title,
                    document_title=doc_title,
                    text=sec_clean,
                    metadata=metadata,
                )
            )

        return chunks
