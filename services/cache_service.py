import os
import time
import numpy as np
from dotenv import load_dotenv

# Load env in case
load_dotenv()

# Global states
_redis_client = None
_redis_tried = False
_in_memory_cache = {}  # {tenant_id: {query_text: {"response": str, "embedding": np.array, "timestamp": float, "sources": list, "metadata": dict}}}

def _get_ram_cache():
    global _in_memory_cache
    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists():
            if "_in_memory_cache" not in st.session_state:
                st.session_state._in_memory_cache = {}
            return st.session_state._in_memory_cache
    except Exception:
        pass
    return _in_memory_cache

def get_redis_client():
    """
    Connects to Redis using value stored in REDIS_URL env variable (default localhost:6379).
    Pings once to check availability.
    """
    global _redis_client, _redis_tried
    if _redis_tried:
        return _redis_client
        
    _redis_tried = True
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        import redis
        client = redis.Redis.from_url(redis_url, socket_timeout=2.0)
        client.ping()
        _redis_client = client
        print("Successfully connected to Redis server.")
    except Exception as e:
        print(f"Redis is unavailable: {e}. Falling back to in-memory RAM cache.")
        _redis_client = None
        
    return _redis_client

def _get_cache_scope(session_id, tenant_id, department):
    s_id = str(session_id or "default")
    t_id = str(tenant_id or "default")
    d_id = str(department or "default")
    return f"{s_id}:{t_id}:{d_id}"

def get_cached_result(tenant_id, query_text, query_embedding, similarity_threshold=0.92, session_id=None, department=None):
    """
    Searches the semantic cache (Redis or RAM dictionary) for a semantically similar query.
    Similarity calculation uses Cosine Distance.
    Returns (cached_response, cached_sources, cache_type) if matched, else (None, None, None).
    """
    client = get_redis_client()
    query_vector = np.array(query_embedding, dtype=np.float32)
    qn_norm = np.linalg.norm(query_vector)
    if qn_norm == 0:
        return None, None, None

    # Normalise query vector for cosine similarity calculations
    query_vector = query_vector / qn_norm
    scope = _get_cache_scope(session_id, tenant_id, department)
    
    if client is not None:
        try:
            import json
            # Retrieve all cache keys belonging to this tenant/organization context
            pattern = f"tenant:{scope}:cache:*"
            keys = client.keys(pattern)
            
            best_similarity = -1.0
            best_val = None
            
            for key in keys:
                raw_data = client.get(key)
                if not raw_data:
                    continue
                try:
                    data = json.loads(raw_data.decode("utf-8"))
                    cached_emb = np.array(data["embedding"], dtype=np.float32)
                    cen_norm = np.linalg.norm(cached_emb)
                    if cen_norm == 0:
                        continue
                    cached_emb = cached_emb / cen_norm
                    
                    similarity = float(np.dot(query_vector, cached_emb))
                    if similarity >= similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_val = data
                except Exception:
                    pass
            
            if best_val:
                print(f"Redis cache hit! Semantic similarity: {best_similarity:.4f}")
                return best_val["response"], best_val.get("sources", []), "Redis (Semantic)"
                
        except Exception as e:
            print(f"Error querying Redis cache: {e}. Defaulting search to RAM fallback.")
            
    # Local RAM Fallback Cache Search
    ram_cache = _get_ram_cache()
    tenant_ram = ram_cache.get(scope, {})
    print(f"[CACHE DEBUG] get_cached_result for scope '{scope}'. RAM cache size: {len(tenant_ram)}. Keys: {list(tenant_ram.keys())}")
    best_similarity = -1.0
    best_val = None
    
    for q_text, entry in tenant_ram.items():
        cached_emb = np.array(entry["embedding"], dtype=np.float32)
        cen_norm = np.linalg.norm(cached_emb)
        if cen_norm == 0:
            continue
        cached_emb = cached_emb / cen_norm
        
        similarity = float(np.dot(query_vector, cached_emb))
        if similarity >= similarity_threshold and similarity > best_similarity:
            best_similarity = similarity
            best_val = entry
            
    if best_val:
        print(f"RAM cache hit! Semantic similarity: {best_similarity:.4f}")
        return best_val["response"], best_val.get("sources", []), "RAM (Semantic)"
        
    return None, None, None

def set_cached_result(tenant_id, query_text, query_embedding, response_text, sources=None, ttl_seconds=3600, session_id=None, department=None):
    """
    Saves the research response, query text, sources metadata, and embedding vector.
    Saves to Redis (if present) and synchronizes to local RAM cache.
    """
    if sources is None:
        sources = []
        
    embedding_list = list(query_embedding) if hasattr(query_embedding, "tolist") else [float(x) for x in query_embedding]
    scope = _get_cache_scope(session_id, tenant_id, department)
    
    # Save to Redis
    client = get_redis_client()
    if client is not None:
        try:
            import json
            import hashlib
            query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:16]
            key = f"tenant:{scope}:cache:{query_hash}"
            cache_payload = {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "department": department,
                "query": query_text,
                "embedding": embedding_list,
                "response": response_text,
                "sources": sources,
                "timestamp": time.time()
            }
            client.setex(key, ttl_seconds, json.dumps(cache_payload))
            print(f"Saved result to Redis cache: {key}")
        except Exception as e:
            print(f"Error saving to Redis: {e}")
            
    # Save to RAM Fallback
    ram_cache = _get_ram_cache()
    if scope not in ram_cache:
        ram_cache[scope] = {}
        
    ram_cache[scope][query_text] = {
        "response": response_text,
        "embedding": np.array(embedding_list, dtype=np.float32),
        "sources": sources,
        "timestamp": time.time()
    }
    print(f"Saved result to RAM cache for scope '{scope}'")

def invalidate_cache(tenant_id=None, session_id=None, department=None):
    """
    Clear cache entries when document database changes or session resets.
    If tenant_id/session_id/department are specified, only that scope cache gets wiped, else the entire cache is invalidated.
    """
    ram_cache = _get_ram_cache()
    
    # 1. Invalidation of RAM
    if session_id or tenant_id or department:
        scope = _get_cache_scope(session_id, tenant_id, department)
        if scope in ram_cache:
            ram_cache[scope] = {}
            print(f"Cleared local RAM cache for scope '{scope}'")
    else:
        ram_cache.clear()
        print("Cleared global local RAM cache")
        
    # 2. Invalidation of Redis
    client = get_redis_client()
    if client is not None:
        try:
            if session_id or tenant_id or department:
                scope = _get_cache_scope(session_id, tenant_id, department)
                pattern = f"tenant:{scope}:cache:*"
                keys = client.keys(pattern)
                if keys:
                    client.delete(*keys)
                print(f"Cleared Redis cache keys for scope '{scope}'")
            else:
                pattern = "tenant:*:cache:*"
                keys = client.keys(pattern)
                if keys:
                    client.delete(*keys)
                print("Cleared all Redis cache keys")
        except Exception as e:
            print(f"Error clearing Redis cache keys: {e}")
