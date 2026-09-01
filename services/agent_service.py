import requests
import json
import time
from services.retrieval_service import search_chunks
from services.llm_service import is_conversational_query

def classify_query_type(query, api_key):
    """
    Classifies the input query into: 'greeting', 'simple_factual', 'complex_business', 'recommendation', or 'comparison_conflict'.
    """
    if is_conversational_query(query):
        return "greeting"
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = (
        "You are an expert query classifier for an enterprise search agent.\n"
        "Classify the input query into one of these types:\n"
        "- 'greeting': Friendly conversational entries, hi, hello, how are you.\n"
        "- 'simple_factual': Direct factual lookups or questions about growth metrics, GPAs, names, or values.\n"
        "- 'complex_business': Multi-step business research, strategic planning, or operational overviews.\n"
        "- 'recommendation': Queries asking for suggestions, actionable steps, choices, or plans (e.g. 'what should we prioritize').\n"
        "- 'comparison_conflict': Queries comparing multiple elements or asking about discrepancies/differences in reports.\n\n"
        "Return a JSON object with one key 'query_type' containing the classification code from above. Do not write explanations."
    )
    
    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ],
        "temperature": 0.0,
        "max_tokens": 30,
        "response_format": {"type": "json_object"}
    }
    
    try:
        for attempt in range(3):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=8
            )
            if response.status_code == 200:
                res_json = response.json()
                data = json.loads(res_json["choices"][0]["message"]["content"])
                q_type = data.get("query_type", "simple_factual").lower().strip()
                if q_type in ["greeting", "simple_factual", "complex_business", "recommendation", "comparison_conflict"]:
                    return q_type
                break
            elif response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            else:
                break
    except Exception:
        pass
        
    return "simple_factual"

def decompose_query(query, api_key):
    """
    Decomposes a complex business query into 2 to 3 distinct sub-questions for thorough coverage.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = (
        "You are an AI research planner. Decompose the input question into 2 to 3 distinct, simpler sub-questions "
        "required to compile a complete answer. Keep queries short, keyword-focused, and factual.\n"
        "Return a JSON object with a single key 'sub_questions' containing an array of strings.\n"
        "Example: {\"sub_questions\": [\"Workforce report AI policies\", \"Technology report AI priorities\"]}\n"
        "Do not write explanations."
    )
    
    payload = {
        "model": "qwen/qwen3.6-27b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}"}
        ],
        "temperature": 0.1,
        "max_tokens": 150,
        "response_format": {"type": "json_object"}
    }
    
    try:
        for attempt in range(3):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                res_json = response.json()
                data = json.loads(res_json["choices"][0]["message"]["content"])
                questions = data.get("sub_questions", [])
                return questions[:3]
            elif response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            else:
                break
    except Exception as e:
        print(f"Decomposition error: {e}")
        
    return [query]

def perform_research_stream(
    query, 
    collection_name, 
    api_key, 
    tenant_id="AcmeCorp", 
    research_depth="Deep Research", 
    use_hybrid=True, 
    use_reranker=False, 
    temperature=0.2,
    model="qwen/qwen3.6-27b",
    progress_callback=None
):
    """
    Executive orchestrator for multi-step research.
    """
    # For Quick Research, skip the classify API call entirely (greetings already intercepted in app.py)
    if research_depth == "Quick Research":
        q_type = "simple_factual"
    else:
        q_type = classify_query_type(query, api_key)
    
    # 1.5 Handle Conversational Greeting Queries Directly
    if q_type == "greeting":
        if progress_callback:
            progress_callback("Responding to greeting...")
        system_prompt = (
            "You are a helpful, professional AI assistant for the Enterprise RAG System.\n"
            "The user is saying hello, greeting you, or saying thank you. Respond politely, "
            "warmly, and concisely, guiding them to ask research questions about their business "
            "documents or files. Do not use emojis."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.5,
            "max_tokens": 150,
            "stream": False
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        meta_json = json.dumps({
            "complexity": "greeting",
            "sub_queries": [query],
            "sources": []
        })
        yield f"__METADATA__:{meta_json}\n"
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=12
            )
            if response.status_code == 200:
                res_content = response.json()["choices"][0]["message"]["content"].strip()
                if "</think>" in res_content:
                    res_content = res_content.split("</think>")[-1].strip()
                yield res_content
            else:
                yield "Hello! How can I assist you with your corporate knowledge documents today?"
        except Exception:
            yield "Hello! How can I assist you with your corporate knowledge documents today?"
        return
    
    # Define decomposition check based on classification type
    sub_queries = [query]
    if research_depth == "Deep Research" and q_type in ["complex_business", "recommendation", "comparison_conflict"]:
        if progress_callback:
            progress_callback(f"Planning research for {q_type} and decomposing query...")
        sub_queries = decompose_query(query, api_key)
        if not sub_queries:
            sub_queries = [query]
            
    # 2. Document Search with larger top_k=8 to resolve retrieval coverage
    if progress_callback:
        progress_callback(f"Searching knowledge base with {len(sub_queries)} queries...")
    
    print(f"[DEBUG perform_research_stream] collection_name={collection_name}, sub_queries={sub_queries}")
        
    all_retrieved = []
    seen_texts = set()
    
    for sq in sub_queries:
        results = search_chunks(
            collection_name=collection_name,
            query=sq,
            top_k=8,
            use_hybrid=use_hybrid,
            use_reranker=use_reranker
        )
        print(f"[DEBUG perform_research_stream] query='{sq}' returned {len(results)} results")
        for item in results:
            text = item.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                all_retrieved.append(item)
                
    # 3. Evidence Validation & Context Budgeting
    if progress_callback:
        progress_callback("Validating evidence and preparing context budget...")
    
    print(f"[DEBUG perform_research_stream] all_retrieved count = {len(all_retrieved)}")
        
    if not all_retrieved:
        yield "Insufficient evidence found in the available knowledge base."
        return
        
    # Restrict total retrieved chunks to top 8 max to keep prompt tokens within Groq limits
    all_retrieved = all_retrieved[:8]
    
    # Format candidates for Context prompt with chunk character truncation
    formatted_sources = []
    sources_telemetry = []
    for idx, item in enumerate(all_retrieved):
        meta = item.get("metadata", {}) or {}
        fn = meta.get("filename", "Doc")
        page = meta.get("page", 1)
        cat = meta.get("category", "General")
        dept = meta.get("department", "All")
        ver = meta.get("version", "1.0")
        priority = meta.get("priority", 3)
        dt = meta.get("date", "Unknown")
        
        src = f"{fn} (Page {page})" if meta.get("source") == "pdf" else fn
        
        # Truncate long chunk texts to max 1200 characters per chunk snippet
        chunk_text = item.get("text", "")
        if len(chunk_text) > 1200:
            chunk_text = chunk_text[:1200] + "... [truncated]"
            
        formatted_sources.append(
            f"--- Context [{idx+1}] ---\n"
            f"Source: {src}\n"
            f"Category: {cat}\n"
            f"Department: {dept}\n"
            f"Date/Version: {dt} (v{ver})\n"
            f"Priority: {priority}\n"
            f"Content: {chunk_text}"
        )
        if src not in sources_telemetry:
            sources_telemetry.append(src)
            
    context_str = "\n\n".join(formatted_sources)
    # Enforce global maximum context budget ceiling of 10,000 characters (~2,500 tokens)
    if len(context_str) > 10000:
        context_str = context_str[:10000] + "\n... [Remaining context omitted for payload limits]"
    
    system_prompt = (
        "You are an Enterprise AI Research Assistant. You have been given relevant document excerpts as Context below.\n\n"
        "INSTRUCTIONS:\n"
        "1. Thoroughly analyze the Context and provide a rich, accurate, and direct answer to the user's question.\n"
        "2. For general summary, overview, or report questions, synthesize and highlight all key facts, metrics, figures, dates, tables, and takeaways present in the Context.\n"
        "3. Write naturally like ChatGPT — clear, well-structured paragraphs with bullet points for lists and bolding for key findings.\n"
        "4. Refer to documents by their filename (e.g. 'According to AcmeStrategy2026.docx'). Do not say 'Context [1]'.\n"
        "5. At the end, add a line with '---' followed by '**Sources:** ' and list the filenames used.\n"
        "6. Do NOT use <think> tags. Do NOT show reasoning steps. Output ONLY the final answer.\n"
    )
    
    # Append /no_think to disable qwen thinking mode
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer: /no_think"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": 4096,
        "stream": False
    }
    
    meta_json = json.dumps({
        "complexity": q_type,
        "sub_queries": sub_queries,
        "sources": sources_telemetry
    })
    yield f"__METADATA__:{meta_json}\n"
    
    try:
        max_retries = 4
        for attempt in range(max_retries):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45
            )
            if response.status_code == 200:
                res_json = response.json()
                full_response = res_json["choices"][0]["message"]["content"].strip()
                
                print(f"[DEBUG GROQ RAW] length={len(full_response)}, first 200 chars: {repr(full_response[:200])}")
                
                # Strip any think blocks that may still appear
                import re
                cleaned = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
                
                if not cleaned and '<think>' in full_response:
                    if '</think>' in full_response:
                        cleaned = full_response.split('</think>')[-1].strip()
                    if not cleaned:
                        think_match = re.search(r'<think>(.*?)</think>', full_response, flags=re.DOTALL)
                        if think_match:
                            cleaned = think_match.group(1).strip()
                
                if '<think>' in cleaned:
                    cleaned = cleaned.split('<think>')[0].strip()
                if '</think>' in cleaned:
                    cleaned = cleaned.split('</think>')[-1].strip()
                
                print(f"[DEBUG GROQ CLEANED] length={len(cleaned)}, first 200 chars: {repr(cleaned[:200])}")
                
                if cleaned:
                    yield cleaned
                else:
                    yield "Insufficient evidence found in the available knowledge base."
                return
            elif response.status_code == 413:
                # HTTP 413: Payload / Request Entity Too Large. Trim context dynamically and retry!
                print(f"[DEBUG GROQ 413] Request entity too large (payload length {len(user_prompt)}). Truncating context and retrying...")
                if progress_callback:
                    progress_callback("Payload large. Optimizing context budget for LLM...")
                # Trim context by 50%
                context_str = context_str[:4000] + "\n... [Context trimmed for payload size]"
                user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer: /no_think"
                payload["messages"][1]["content"] = user_prompt
                time.sleep(1)
                continue
            elif response.status_code == 429:
                wait_sec = 10 * (attempt + 1)
                if progress_callback:
                    progress_callback(f"Rate limit hit. Waiting {wait_sec}s before retry ({attempt+1}/{max_retries})...")
                time.sleep(wait_sec)
                continue
            else:
                print(f"[DEBUG GROQ ERROR] status={response.status_code}, body={response.text[:300]}")
                yield f"Error generating research synthesis: Groq error {response.status_code}"
                return
        yield "Rate limit reached after multiple retries. Please wait 1-2 minutes and try again."
    except Exception as e:
        yield f"Error generating research synthesis: {str(e)}"


