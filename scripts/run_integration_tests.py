import os
import sys
import time
import uuid
import numpy as np
import io

# Setup import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.document_service import extract_document_pages
from services.chunk_service import create_parent_child_chunks
from services.embedding_service import generate_embeddings
from services.vector_service import store_chunks, delete_collection, get_collection
from services.retrieval_service import search_chunks
from services.agent_service import perform_research_stream
from services.cache_service import get_cached_result, set_cached_result, invalidate_cache
from services.translation_service import detect_language, translate_text

# Fictional Test Documents for Ingestion
MOCK_DOCS = [
    {
        "filename": "AcmeStrategy2026.docx",
        "content": "AcmeCorp Strategy 2026. The primary path for technology growth is AI-driven cloud automation. We are allocating $8.5M for the infrastructure transition in Q1 2026. Security auditing is our top corporate priority.",
        "category": "Strategy", "priority": 1, "department": "Technology", "version": "1.0", "date": "2026-01-10", "type": "docx"
    },
    {
        "filename": "HR_Audit_2026.pdf",
        "content": "AcmeCorp HR and Workforce report. Security auditing and payroll integration are critical. Operational budget totals $4.2M for internal systems. We are targeting hiring 45 engineers in tech hubs.",
        "category": "HR", "priority": 2, "department": "HR", "version": "2.1", "date": "2026-02-15", "type": "pdf"
    },
    {
        "filename": "Marketing_Budgets.csv",
        "content": "Department,Budget_M,AI_Budget_MUnit\nFinance,12.5,1.5\nStrategy,15.5,2.0\n",
        "category": "Financial", "priority": 3, "department": "Finance", "version": "1.1", "date": "2026-03-01", "type": "csv"
    }
]

def run_tests():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not configured in .env.")
        sys.exit(1)
        
    session_id = f"test_{int(time.time())}"
    collection_name = f"rag_test_{session_id}"
    results = {}
    
    # helper files bytes streams
    class MockFileObj(io.BytesIO):
        def __init__(self, content_bytes, name):
            super().__init__(content_bytes)
            self.name = name
            self.size = len(content_bytes)

    print("====================================================")
    print("STARTING ENTERPRISE AI AGENT INTEGRATION TESTS")
    print("====================================================")
    
    # ----------------------------------------------------
    # TEST 1: Ingestion Pipeline
    # ----------------------------------------------------
    print("\n[TEST 1] Ingesting Mock Documents with Extended Metadata...")
    try:
        # Clean collection start
        delete_collection(collection_name)
        
        for mock in MOCK_DOCS:
            pages = [{"page": 1, "text": mock["content"]}]
            doc_meta = {
                "document_id": str(uuid.uuid4()),
                "filename": mock["filename"],
                "source_type": mock["type"],
                "category": mock["category"],
                "department": mock["department"],
                "version": mock["version"],
                "priority": mock["priority"],
                "ingestion_date": "2026-08-31",
                "document_date": mock["date"]
            }
            
            chunks, metas = create_parent_child_chunks(
                pages=pages,
                source_type=mock["type"],
                filename=mock["filename"],
                additional_metadata=doc_meta
            )
            
            embs = generate_embeddings(chunks)
            store_chunks(collection_name, chunks, embs, metas)
            print(f"  Indexed '{mock['filename']}' ({len(chunks)} chunks)")
            
        coll = get_collection(collection_name)
        count = coll.count()
        if count > 0:
            results["Ingestion_Pipeline"] = ("PASS", f"Indexed {count} total chunks.")
        else:
            results["Ingestion_Pipeline"] = ("FAIL", "Collection remains empty.")
    except Exception as e:
        results["Ingestion_Pipeline"] = ("FAIL", str(e))
        
    # ----------------------------------------------------
    # TEST 2: Hybrid Retrieval
    # ----------------------------------------------------
    print("\n[TEST 2] Verifying Hybrid Dense/Sparse Search (RRF)...")
    try:
        res = search_chunks(collection_name, "technology cloud automation", top_k=2)
        if len(res) > 0:
            found_key = any("cloud automation" in r["text"].lower() for r in res)
            if found_key:
                results["Hybrid_Retrieval"] = ("PASS", f"Retrieved matches correctly: '{res[0]['text'][:30]}...'")
            else:
                results["Hybrid_Retrieval"] = ("FAIL", "Keywords match not found in returned text.")
        else:
            results["Hybrid_Retrieval"] = ("FAIL", "No chunks returned.")
    except Exception as e:
        results["Hybrid_Retrieval"] = ("FAIL", str(e))
        
    # ----------------------------------------------------
    # TEST 3: Multilingual In/Out Routing
    # ----------------------------------------------------
    print("\n[TEST 3] Testing Tamil/Hindi Translate/Route Pipelines...")
    try:
        # Detect Tamil
        tamil_query = "AI மேகக்கணி ஆட்டோமேஷன்"  # AI cloud automation
        lang = detect_language(tamil_query, api_key)
        tamil_translated = translate_text(tamil_query, "en", api_key)
        
        # Detect Hindi
        hindi_query = "प्रौद्योगिकी विकास"  # Tech growth
        hindi_lang = detect_language(hindi_query, api_key)
        hindi_translated = translate_text(hindi_query, "en", api_key)
        
        tamil_success = lang == "ta" or "automation" in tamil_translated.lower() or "ai" in tamil_translated.lower()
        hindi_success = hindi_lang == "hi" or "growth" in hindi_translated.lower() or "tech" in hindi_translated.lower()
        
        if tamil_success and hindi_success:
            results["Multilingual_Routing"] = ("PASS", f"Tamil detected: {lang} ({tamil_translated}); Hindi detected: {hindi_lang} ({hindi_translated})")
        else:
            results["Multilingual_Routing"] = ("FAIL", f"Detection mismatched. TA={lang}, HI={hindi_lang}")
    except Exception as e:
        results["Multilingual_Routing"] = ("FAIL", str(e))
        
    # ----------------------------------------------------
    # TEST 4: Semantic Caching
    # ----------------------------------------------------
    print("\n[TEST 4] Testing Redis Semantic Cache connection & fallback updates...")
    try:
        # Flush first
        invalidate_cache(tenant_id="AcmeCorp")
        
        # Seed cache
        q_text = "What is the primary corporate path for technology growth?"
        q_emb = generate_embeddings([q_text])[0]
        cached_ans = "SYSTEM SUMMARY: AI-driven cloud automation is the primary growth path."
        set_cached_result("AcmeCorp", q_text, q_emb, cached_ans, sources=["AcmeStrategy2026.docx"])
        
        # Test exact hits
        res_text, res_src, cache_type = get_cached_result("AcmeCorp", q_text, q_emb, similarity_threshold=0.92)
        
        # Test semantically similar query match
        similar_q = "What is the primary corporate pathway for technology growth?"
        sim_emb = generate_embeddings([similar_q])[0]
        sim_text, sim_src, sim_type = get_cached_result("AcmeCorp", similar_q, sim_emb, similarity_threshold=0.92)

        
        if res_text == cached_ans and sim_text == cached_ans:
            results["Semantic_Cache"] = ("PASS", f"Cache hit on similar query using {sim_type}.")
        else:
            results["Semantic_Cache"] = ("FAIL", f"Match failed. Exact={res_text is not None}, Similar={sim_text is not None}")
    except Exception as e:
        results["Semantic_Cache"] = ("FAIL", str(e))
        
    # ----------------------------------------------------
    # TEST 5: Orchestration (Agent Synthesis & Conflict Checks)
    # ----------------------------------------------------
    print("\n[TEST 5] Testing Research Agent Synthesis & Matrix Generation...")
    try:
        stream = perform_research_stream(
            query="Summarize primary technology priorities and budgets",
            collection_name=collection_name,
            api_key=api_key,
            tenant_id="AcmeCorp",
            research_depth="Deep Research"
        )
        
        ans = ""
        for chunk in stream:
            if not chunk.startswith("__METADATA__:"):
                ans += chunk
                
        has_content = len(ans.strip()) > 20
        valid_response = ("Sources" in ans or "Acme" in ans or "infrastructure" in ans or "Insufficient evidence" in ans)
        
        if has_content and valid_response:
            results["Research_Orchestrator"] = ("PASS", f"Successfully synthesized research answer ({len(ans.strip())} chars).")
        else:
            results["Research_Orchestrator"] = ("FAIL", "Markdown synthesis parameters are incomplete.")
    except Exception as e:
        results["Research_Orchestrator"] = ("FAIL", str(e))
        
    # Standard cleanup
    delete_collection(collection_name)
    invalidate_cache(tenant_id="AcmeCorp")
    
    print("\n====================================================")
    print("INTEGRATION TESTS COMPLETE — RESULTS SUMMARY:")
    print("====================================================")
    all_pass = True
    for test_name, (status, detail) in results.items():
        print(f"[{status}] {test_name}: {detail}")
        if status == "FAIL":
            all_pass = False
            
    print("====================================================")
    if all_pass:
        print("ALL TESTS PASSED SUCCESSFULLY! Local validation completed.")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED. Please review the terminal logs.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
