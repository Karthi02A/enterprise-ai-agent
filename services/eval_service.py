import json
import os
import time

import uuid

LOG_FILE = "evaluation_log.json"

def log_query_execution(
    query, 
    complexity, 
    sub_query_count, 
    retrieval_latency, 
    llm_latency, 
    total_latency, 
    cache_hit_or_miss, 
    sources_retrieved,
    session_id=None,
    tenant_id=None,
    department=None,
    error_message=None
):
    """
    Appends query execution metadata to evaluation_log.json. Returns the unique log_id.
    """
    log_id = str(uuid.uuid4())
    log_entry = {
        "log_id": log_id,
        "session_id": session_id,
        "tenant_id": tenant_id,
        "department": department,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "complexity": complexity,
        "sub_query_count": sub_query_count,
        "retrieval_latency": round(retrieval_latency, 3),
        "llm_latency": round(llm_latency, 3),
        "total_latency": round(total_latency, 3),
        "cache_hit_or_miss": cache_hit_or_miss,
        "sources_retrieved": sources_retrieved,
        "error": error_message,
        "feedback": None  # Placeholder for explicit thumbs voting
    }
    
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
            
    logs.append(log_entry)
    
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error logging query telemetry: {e}")
        
    return log_id

def read_evaluation_data(session_id=None):
    """
    Reads query logs from evaluation_log.json. Optional filtering by session_id.
    """
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
            if session_id is not None:
                return [log for log in logs if log.get("session_id") == session_id]
            return logs
    except Exception:
        return []

def update_feedback(log_id, feedback_value):
    """
    Updates the feedback field of a specific query entry identified by log_id.
    """
    if not log_id:
        return False
    logs = read_evaluation_data(session_id=None)  # Read all logs
    updated = False
    for log in logs:
        if log.get("log_id") == log_id:
            log["feedback"] = feedback_value
            updated = True
            break
            
    if updated:
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error updating feedback: {e}")
    return False
