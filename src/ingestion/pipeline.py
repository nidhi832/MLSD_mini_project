"""
Ingestion Pipeline & CLI Tool (D2, L1).
Invocable from command-line: python -m src.ingestion.pipeline
Reads corpus files, chunks them, generates embeddings, records manifest and freshness metrics.
"""

import os
import sys
import argparse
from typing import Dict, Any

from src.ingestion.chunker import RegulatoryCorpusChunker
from src.ingestion.vector_store import RegulatoryVectorStore


def run_ingestion_pipeline(
    corpus_dir: str = "data/corpus",
    vector_store_dir: str = "data/vector_store",
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    print("=" * 60)
    print("AIR PASSENGER CLAIM TRIAGE - CORPUS INGESTION PIPELINE (D2)")
    print("=" * 60)

    car_file = os.path.join(corpus_dir, "dgca_car_sec3_series_m_part_iv.txt")
    charter_file = os.path.join(corpus_dir, "airline_passenger_charter.txt")

    if not os.path.exists(car_file) or not os.path.exists(charter_file):
        raise FileNotFoundError(f"Required corpus documents missing in {corpus_dir}")

    with open(car_file, "r", encoding="utf-8") as f:
        car_text = f.read()

    with open(charter_file, "r", encoding="utf-8") as f:
        charter_text = f.read()

    print(f"[*] Reading DGCA CAR: {len(car_text)} chars")
    car_chunks = RegulatoryCorpusChunker.chunk_dgca_car(car_text)
    print(f"[+] Chunked CAR into {len(car_chunks)} regulatory units")

    print(f"[*] Reading Airline Charter: {len(charter_text)} chars")
    charter_chunks = RegulatoryCorpusChunker.chunk_airline_charter(charter_text)
    print(f"[+] Chunked Charter into {len(charter_chunks)} regulatory units")

    all_chunks = car_chunks + charter_chunks
    print(f"[+] Total chunks to index: {len(all_chunks)}")

    source_versions = {
        "dgca_car": "Section 3 Series M Part IV Rev 4 (15 Feb 2023)",
        "airline_charter": "IndiGo Passenger Charter 6E-COC-2023-V2",
    }

    vstore = RegulatoryVectorStore(storage_dir=vector_store_dir)
    if force_rebuild:
        print("[!] Force rebuild requested: clearing previous store.")
        vstore.chunks = []
        vstore.embeddings = None

    manifest = vstore.index_chunks(all_chunks, source_versions)

    print("-" * 60)
    print("INGESTION COMPLETE:")
    print(f"  Build Timestamp : {manifest['build_timestamp']}")
    print(f"  Total Chunks    : {manifest['total_chunks']}")
    print(f"  Newly Embedded  : {manifest['newly_embedded']}")
    print(f"  Reused Chunks   : {manifest['reused_chunks']}")
    print(f"  Duration        : {manifest['duration_seconds']}s")
    print(f"  Throughput      : {manifest['throughput_chunks_per_second']} chunks/sec")
    print("=" * 60)

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest regulatory corpus and build vector index.")
    parser.add_argument("--corpus-dir", default="data/corpus", help="Path to corpus files directory")
    parser.add_argument("--vector-store-dir", default="data/vector_store", help="Path to vector store output")
    parser.add_argument("--force", action="store_true", help="Force rebuild of entire index")
    args = parser.parse_args()

    try:
        run_ingestion_pipeline(args.corpus_dir, args.vector_store_dir, args.force)
        sys.exit(0)
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)
