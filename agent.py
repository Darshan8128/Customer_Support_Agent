"""
agent.py — Agentic RAG Customer Support Agent for Date AI Chatbot

Uses Gemini native function calling via LangChain's ChatGoogleGenerativeAI
to provide multi-tool reasoning, retrieval self-correction, escalation, and memory.

Same client config as utils.py:
  - LLM: ChatGoogleGenerativeAI("gemini-1.5-flash-8b-latest")
  - Embeddings: GoogleGenerativeAIEmbeddings("models/gemini-embedding-2")
  - Auth: GOOGLE_API_KEY from .env
"""

import os
import json
import time
import uuid
import sqlite3
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool
from deep_translator import GoogleTranslator

from utils import load_vectorstore
import database

load_dotenv(override=True)
try:
    import streamlit as st
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
except Exception:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# Shared LLM factory — same model/key as utils.py
# ============================================================

def _get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Returns a ChatGoogleGenerativeAI instance matching the existing utils.py config."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )


# ============================================================
# TOOL 1: search_knowledge_base — Agentic RAG Loop
# ============================================================
#
# Flow:
#   retrieve_documents(query)
#       ↓
#   grade_retrieval(query, docs)   ← separate LLM call
#       ↓ sufficient? → generate answer
#       ↓ insufficient?
#   rewrite_query(original_query, reason)  ← separate LLM call
#       ↓
#   retrieve_documents(rewritten_query)    ← 2nd attempt
#       ↓
#   grade_retrieval again
#       ↓ sufficient? → generate answer
#       ↓ still insufficient? → signal escalation
# ============================================================


def _retrieve_documents(query: str, k: int = 5) -> list[dict]:
    """
    Retrieves top-k documents from the FAISS vectorstore with similarity scores.
    
    Uses the existing load_vectorstore() from utils.py, so it respects
    session state caching, uploaded files, and the knowledge_updated flag.
    
    Returns:
        list of dicts: [{"content": str, "score": float, "index": int}, ...]
        Lower FAISS L2 score = better match.
    """
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return []

    # similarity_search_with_score returns [(Document, score), ...]
    results = vectorstore.similarity_search_with_score(query, k=k)

    retrieved = []
    for i, (doc, score) in enumerate(results):
        retrieved.append({
            "content": doc.page_content,
            "score": round(float(score), 4),
            "index": i,
        })

    return retrieved


def _grade_retrieval(query: str, retrieved_docs: list[dict]) -> dict:
    """
    Separate LLM call: judges whether retrieved documents are sufficient
    to answer the query accurately.
    
    Returns:
        {"verdict": "sufficient" | "insufficient", "reason": str}
    """
    llm = _get_llm(temperature=0.0)

    docs_text = "\n\n---\n\n".join(
        f"[Doc {d['index']}] (similarity score: {d['score']})\n{d['content']}"
        for d in retrieved_docs
    )

    grading_prompt = f"""You are a retrieval quality grader for a customer support knowledge base.
Judge whether the retrieved documents contain sufficient information to answer the user's query.

USER QUERY: {query}

RETRIEVED DOCUMENTS:
{docs_text}

Evaluate these criteria:
1. Do the documents contain information DIRECTLY relevant to the query?
2. Is there enough context to give a helpful, accurate answer?
3. Would answering from these documents risk giving incorrect or misleading information?

Respond with EXACTLY this JSON (no markdown fences, no extra text):
{{"verdict": "sufficient", "reason": "brief explanation"}}
or
{{"verdict": "insufficient", "reason": "brief explanation"}}"""

    response = llm.invoke([HumanMessage(content=grading_prompt)])

    try:
        result = json.loads(response.content.strip())
        if result.get("verdict") not in ("sufficient", "insufficient"):
            result["verdict"] = "sufficient"  # safe default
        return result
    except json.JSONDecodeError:
        # Fallback: parse verdict from free text
        content = response.content.lower()
        if "insufficient" in content:
            return {"verdict": "insufficient", "reason": "Parsed from unstructured LLM response"}
        return {"verdict": "sufficient", "reason": "Default: could not parse structured response"}


def _rewrite_query(original_query: str, reason: str) -> str:
    """
    Separate LLM call: rewrites/expands the query to improve retrieval,
    informed by why the first retrieval was judged insufficient.
    
    Returns:
        The rewritten query string.
    """
    llm = _get_llm(temperature=0.4)

    rewrite_prompt = f"""You are a search query optimizer for a customer support knowledge base.
The original query did not retrieve sufficient documents.

ORIGINAL QUERY: {original_query}
REASON RETRIEVAL WAS INSUFFICIENT: {reason}

Rewrite the query to improve retrieval. Strategies:
- Use synonyms or alternative phrasings
- Break complex questions into core concepts
- Add contextual terms likely to appear in product documentation
- Expand abbreviations or jargon

Respond with ONLY the rewritten query — no explanation, no quotes."""

    response = llm.invoke([HumanMessage(content=rewrite_prompt)])
    return response.content.strip().strip('"').strip("'")


def search_knowledge_base(query: str) -> dict:
    """
    Agentic RAG loop: retrieve → grade → (rewrite → re-retrieve) → answer or escalate.
    
    This is NOT a simple retrieve-and-return. It self-corrects retrieval quality
    through a grade → rewrite → retry cycle (max 2 retrieval attempts).
    
    Args:
        query: The user's question (already translated to English).
    
    Returns:
        dict with keys:
        - answer: str | None  — generated answer, or None if escalating
        - retrieval_attempts: list[dict]  — log of each attempt's details
        - final_verdict: "sufficient" | "insufficient"
        - escalate: bool  — True means fall back to create_support_ticket
        - escalation_reason: str | None  — why escalation is needed
    """
    MAX_ATTEMPTS = 2
    retrieval_log = []
    current_query = query
    sufficient_docs = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # --- Step A: Retrieve documents ---
        docs = _retrieve_documents(current_query, k=5)

        if not docs:
            retrieval_log.append({
                "attempt": attempt,
                "query_used": current_query,
                "docs_found": 0,
                "verdict": "insufficient",
                "reason": "No documents retrieved — vectorstore may be empty or unavailable",
            })
            if attempt < MAX_ATTEMPTS:
                rewritten = _rewrite_query(query, "No documents were retrieved at all")
                retrieval_log[-1]["rewritten_query"] = rewritten
                current_query = rewritten
                continue
            else:
                break

        # --- Step B: Grade retrieval quality ---
        grade = _grade_retrieval(current_query, docs)

        retrieval_log.append({
            "attempt": attempt,
            "query_used": current_query,
            "docs_found": len(docs),
            "top_score": docs[0]["score"],
            "verdict": grade["verdict"],
            "reason": grade["reason"],
        })

        if grade["verdict"] == "sufficient":
            sufficient_docs = docs
            break

        # --- Step C: Insufficient — rewrite and retry (if attempts remain) ---
        if attempt < MAX_ATTEMPTS:
            rewritten = _rewrite_query(query, grade["reason"])
            retrieval_log[-1]["rewritten_query"] = rewritten
            current_query = rewritten

    # --- Step D: All attempts exhausted, still insufficient → escalation ---
    if sufficient_docs is None:
        return {
            "answer": None,
            "retrieval_attempts": retrieval_log,
            "final_verdict": "insufficient",
            "escalate": True,
            "escalation_reason": (
                f"Could not find sufficient information after {MAX_ATTEMPTS} "
                f"retrieval attempts. Last reason: {retrieval_log[-1].get('reason', 'unknown')}"
            ),
        }

    # --- Step E: Sufficient — generate answer from retrieved context ---
    llm = _get_llm(temperature=0.3)

    context = "\n\n---\n\n".join(doc["content"] for doc in sufficient_docs)

    answer_prompt = f"""You are a friendly, professional customer support agent for Date AI.
Answer the user's question based ONLY on the provided knowledge base context.

KNOWLEDGE BASE CONTEXT:
{context}

USER QUESTION: {query}

Rules:
- Answer accurately using only the provided context
- If the context partially answers the question, share what you can and note what's missing
- Be concise but thorough
- Do NOT fabricate information not present in the context
- If relevant, suggest the user contact support for further help"""

    response = llm.invoke([HumanMessage(content=answer_prompt)])

    return {
        "answer": response.content,
        "retrieval_attempts": retrieval_log,
        "final_verdict": "sufficient",
        "escalate": False,
        "escalation_reason": None,
    }


# ============================================================
# TOOL 2: check_order_status — SQLite-backed Order Lookup
# ============================================================


def check_order_status(order_id: str) -> dict:
    """
    Looks up an order by ID from the SQLite orders database.
    
    Args:
        order_id: The order identifier (e.g., "ORD-1001"). Case-insensitive,
                  auto-uppercased, and auto-prefixed with "ORD-" if missing.
    
    Returns:
        dict with order details if found, or an error message if not found.
    """
    # Normalize: uppercase, ensure ORD- prefix
    order_id = order_id.strip().upper()
    if not order_id.startswith("ORD-"):
        # Handle cases like "1001" or "#1001"
        numeric_part = order_id.lstrip("#").replace("ORD", "").strip("-").strip()
        order_id = f"ORD-{numeric_part}"

    order = database.get_order(order_id)
    if order:
        return {
            "found": True,
            "order": order,
        }
    else:
        return {
            "found": False,
            "order_id": order_id,
            "message": (
                f"No order found with ID '{order_id}'. "
                f"Please verify the order number. Valid format: ORD-XXXX."
            ),
            "suggestion": "If you believe this is an error, I can create a support ticket to investigate.",
        }


# ============================================================
# TOOL 3 & 4: Support Tickets — SQLite-backed
# ============================================================

TICKETS_DB_PATH = os.path.join(os.path.dirname(__file__), "support_tickets.db")


def _init_tickets_db():
    """Creates the support_tickets table if it doesn't exist."""
    conn = sqlite3.connect(TICKETS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS support_tickets (
            ticket_id   TEXT PRIMARY KEY,
            issue       TEXT NOT NULL,
            priority    TEXT NOT NULL DEFAULT 'medium',
            status      TEXT NOT NULL DEFAULT 'open',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _generate_ticket_id() -> str:
    """Generates a human-friendly ticket ID like TKT-A3F8."""
    short_uuid = uuid.uuid4().hex[:4].upper()
    return f"TKT-{short_uuid}"


def create_support_ticket(issue: str, priority: str = "medium") -> dict:
    """
    Creates a new support ticket in the local SQLite database.
    
    Used for:
    - Explicit user escalation requests ("I want to talk to a human")
    - Agentic RAG fallback when knowledge base can't answer after 2 attempts
    - Frustration/complaint detection by the agent orchestrator
    
    Args:
        issue: Description of the customer's issue.
        priority: "low", "medium", "high", or "urgent". Defaults to "medium".
    
    Returns:
        dict with ticket_id, status confirmation, and details.
    """
    _init_tickets_db()

    # Validate priority
    valid_priorities = ("low", "medium", "high", "urgent")
    priority = priority.strip().lower()
    if priority not in valid_priorities:
        priority = "medium"

    ticket_id = _generate_ticket_id()
    now = datetime.now().isoformat()

    conn = sqlite3.connect(TICKETS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO support_tickets (ticket_id, issue, priority, status, created_at, updated_at)
           VALUES (?, ?, ?, 'open', ?, ?)""",
        (ticket_id, issue, priority, now, now),
    )
    conn.commit()
    conn.close()

    return {
        "success": True,
        "ticket_id": ticket_id,
        "issue": issue,
        "priority": priority,
        "status": "open",
        "created_at": now,
        "message": (
            f"Support ticket {ticket_id} has been created with {priority} priority. "
            f"Our support team will review your issue shortly. "
            f"You can check the status anytime by asking about ticket {ticket_id}."
        ),
    }


def check_ticket_status(ticket_id: str) -> dict:
    """
    Looks up an existing support ticket by its ID.
    
    Args:
        ticket_id: The ticket identifier (e.g., "TKT-A3F8"). Case-insensitive,
                   auto-uppercased, and auto-prefixed with "TKT-" if missing.
    
    Returns:
        dict with ticket details if found, or an error message if not found.
    """
    _init_tickets_db()

    # Normalize: uppercase, ensure TKT- prefix
    ticket_id = ticket_id.strip().upper()
    if not ticket_id.startswith("TKT-"):
        ticket_id = f"TKT-{ticket_id.lstrip('#').replace('TKT', '').strip('-').strip()}"

    conn = sqlite3.connect(TICKETS_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM support_tickets WHERE ticket_id = ?",
        (ticket_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "found": True,
            "ticket": {
                "ticket_id": row["ticket_id"],
                "issue": row["issue"],
                "priority": row["priority"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            },
        }
    else:
        return {
            "found": False,
            "ticket_id": ticket_id,
            "message": (
                f"No support ticket found with ID '{ticket_id}'. "
                f"Please verify the ticket number. Format: TKT-XXXX."
            ),
        }


# ============================================================
# STEP 3: Agent Orchestration — Gemini Native Function Calling
# ============================================================

# --- Tool Wrappers for LangChain bind_tools() ---
# These wrap the raw functions above with @tool decorator so
# Gemini sees proper function declarations with descriptions.

@langchain_tool
def search_knowledge_base_tool(query: str) -> str:
    """Search the Date AI knowledge base to answer questions about products,
    services, features, pricing, troubleshooting, and company information.
    This performs an intelligent search with automatic quality checking and
    query refinement. Returns an answer or an escalation signal if the
    knowledge base cannot answer."""
    result = search_knowledge_base(query)
    return json.dumps(result, default=str)


@langchain_tool
def check_order_status_tool(order_id: str) -> str:
    """Look up the current status of a customer order including shipping,
    delivery, and tracking information. Use when the customer mentions an
    order number or asks about order status. Order ID format: ORD-XXXX
    (e.g., ORD-1001). Also accepts bare numbers like 1001."""
    result = check_order_status(order_id)
    return json.dumps(result, default=str)


@langchain_tool
def create_support_ticket_tool(issue: str, priority: str) -> str:
    """Create a support ticket to escalate an issue to the human support team.
    Use when: (1) search_knowledge_base returned escalate=true, (2) the
    customer explicitly asks for a human or to escalate, (3) the customer is
    frustrated or dissatisfied, (4) the issue is too complex for automated
    support. Priority must be: low, medium, high, or urgent."""
    result = create_support_ticket(issue, priority)
    return json.dumps(result, default=str)


@langchain_tool
def check_ticket_status_tool(ticket_id: str) -> str:
    """Check the status of a previously created support ticket.
    Use when the customer asks about an existing ticket.
    Ticket ID format: TKT-XXXX (e.g., TKT-A3F8)."""
    result = check_ticket_status(ticket_id)
    return json.dumps(result, default=str)


# All tools available to the agent
AGENT_TOOLS = [
    search_knowledge_base_tool,
    check_order_status_tool,
    create_support_ticket_tool,
    check_ticket_status_tool,
]

# --- System Prompt ---

AGENT_SYSTEM_PROMPT = """You are a friendly, professional customer support agent for Date AI, a technology company. Help customers by using your tools to look up accurate information before responding.

## Tool Usage Guidelines

1. **search_knowledge_base_tool** — Use for ANY question about Date AI's products, services, features, pricing, troubleshooting, or policies. IMPORTANT: If the tool's result contains "escalate": true, you MUST call create_support_ticket_tool instead of guessing an answer from weak context.

2. **check_order_status_tool** — Use when the customer mentions an order number or asks about shipping/delivery/tracking.

3. **create_support_ticket_tool** — Create a ticket when:
   - search_knowledge_base_tool returned escalate=true (insufficient knowledge base info)
   - The customer explicitly asks to speak with a human or escalate
   - The customer expresses frustration, anger, or repeated dissatisfaction
   - You are NOT confident you can resolve the issue accurately
   Set priority based on urgency and customer tone:
   - "urgent" — system outage, account locked, data loss
   - "high" — frustrated customer, payment issues, broken features
   - "medium" — general questions unresolved by KB, feature requests
   - "low" — minor feedback, cosmetic issues

4. **check_ticket_status_tool** — Use when the customer asks about an existing support ticket.

## Escalation Rules (CRITICAL)
- If search_knowledge_base returns escalate=true → ALWAYS create a support ticket. Do NOT attempt to answer from insufficient context.
- If the customer's tone suggests frustration or complaint (e.g., "this is terrible", "I've asked 3 times", "nothing works") → prefer creating a high-priority ticket over guessing.
- When in doubt about accuracy, escalate rather than risk giving wrong information.

## Response Style
- Warm, empathetic, and professional
- Use tools FIRST, then respond based on tool results
- Resolve references to prior conversation ("that order", "my earlier question") using context
- Always include ticket/order IDs in your response so the customer can track them
- Keep responses concise but thorough
- Do not reveal internal tool mechanics or JSON to the customer
"""


# --- Helper: Summarize tool results for agent reasoning display ---

def _summarize_tool_result(tool_name: str, result_data: dict) -> str:
    """Creates a human-readable one-line summary of a tool's result."""
    if tool_name == "search_knowledge_base_tool":
        verdict = result_data.get("final_verdict", "unknown")
        escalate = result_data.get("escalate", False)
        attempts = len(result_data.get("retrieval_attempts", []))
        if escalate:
            return f"KB search: {attempts} attempt(s), verdict={verdict} → ESCALATION"
        return f"KB search: {attempts} attempt(s), verdict={verdict} → answered from context"

    elif tool_name == "check_order_status_tool":
        if result_data.get("found"):
            order = result_data["order"]
            return f"Order {order['order_id']}: status={order['status']}"
        return f"Order not found: {result_data.get('order_id', '?')}"

    elif tool_name == "create_support_ticket_tool":
        if result_data.get("success"):
            return f"Ticket created: {result_data['ticket_id']} (priority={result_data['priority']})"
        return "Ticket creation failed"

    elif tool_name == "check_ticket_status_tool":
        if result_data.get("found"):
            t = result_data["ticket"]
            return f"Ticket {t['ticket_id']}: status={t['status']}"
        return f"Ticket not found: {result_data.get('ticket_id', '?')}"

    return "Tool executed"


# --- Main Orchestration Function ---

def get_agent_response(query: str, lang: str = "en", conversation_history=None):
    """
    Main entry point: lets Gemini decide which tool(s) to call based on the query.
    Genuine LLM-driven tool selection — no keyword routing.

    Args:
        query: User's message (in their chosen language).
        lang: Language code (e.g., "en", "es", "hi"). Defaults to "en".
        conversation_history: List of past turns. Accepts either:
            - List of (query, answer) tuples (current app.py format)
            - List of {"role": str, "content": str} dicts

    Returns:
        tuple: (answer: str, agent_reasoning: dict)
        - answer: Final response translated to the user's language
        - agent_reasoning: {
              "tools_used": [{"name", "args", "result_summary"}, ...],
              "retrieval_attempts": [...],  # if KB search was used
              "total_iterations": int,
              "total_latency_seconds": float,
          }
    """
    start_time = time.time()

    # --- 1. Translate query to English for the LLM ---
    try:
        query_en = (
            GoogleTranslator(source=lang, target="en").translate(query)
            if lang != "en" else query
        )
    except Exception:
        query_en = query  # Fallback: use original if translation fails

    # --- 2. Build message history (last 5 turns for context) ---
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

    if conversation_history:
        # Take last 5 turns
        recent = conversation_history[-5:]
        for turn in recent:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                # (query_en, answer_en) tuple format from current app
                messages.append(HumanMessage(content=str(turn[0])))
                messages.append(AIMessage(content=str(turn[1])))
            elif isinstance(turn, dict):
                # {"role": ..., "content": ...} dict format
                if turn.get("role") == "user":
                    messages.append(HumanMessage(content=turn["content"]))
                elif turn.get("role") == "assistant":
                    messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=query_en))

    # --- 3. Create LLM with tools bound ---
    llm = _get_llm(temperature=0.3)
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)

    # --- 4. Tool-calling loop ---
    agent_reasoning = {
        "tools_used": [],
        "retrieval_attempts": [],
        "total_iterations": 0,
        "total_latency_seconds": 0.0,
    }

    tool_map = {t.name: t for t in AGENT_TOOLS}
    MAX_ITERATIONS = 6  # Safety cap: prevents infinite tool-call loops

    for iteration in range(MAX_ITERATIONS):
        agent_reasoning["total_iterations"] = iteration + 1

        # Invoke LLM
        try:
            response = llm_with_tools.invoke(messages)
            messages.append(response)  # Add AI response to message history
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                return "I'm sorry, but my Google API free-tier quota has been exhausted. Please wait a bit and try again!", agent_reasoning
            else:
                return f"I encountered an error connecting to my brain: {e}", agent_reasoning

        # If no tool calls, we have the final answer
        if not response.tool_calls:
            break

        # Execute each tool call the LLM requested
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Execute the tool
            if tool_name in tool_map:
                try:
                    tool_result_str = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    tool_result_str = json.dumps({
                        "error": f"Tool execution failed: {str(e)}"
                    })
            else:
                tool_result_str = json.dumps({
                    "error": f"Unknown tool: {tool_name}"
                })

            # Parse result for reasoning metadata
            try:
                result_data = json.loads(tool_result_str)
            except (json.JSONDecodeError, TypeError):
                result_data = {"raw": str(tool_result_str)}

            # Record tool usage for agent reasoning display
            tool_info = {
                "name": tool_name,
                "args": tool_args,
                "result_summary": _summarize_tool_result(tool_name, result_data),
            }
            agent_reasoning["tools_used"].append(tool_info)

            # Track retrieval attempts if KB search was used
            if (tool_name == "search_knowledge_base_tool"
                    and isinstance(result_data, dict)
                    and "retrieval_attempts" in result_data):
                agent_reasoning["retrieval_attempts"] = result_data["retrieval_attempts"]

            # Send tool result back to LLM as a ToolMessage
            messages.append(
                ToolMessage(
                    content=str(tool_result_str),
                    tool_call_id=tool_call["id"],
                )
            )

    # --- 5. Extract final answer ---
    raw_content = response.content
    if isinstance(raw_content, list):
        # Extract the 'text' fields from content blocks
        text_blocks = [block["text"] for block in raw_content if isinstance(block, dict) and "text" in block]
        answer_en = "\n".join(text_blocks) if text_blocks else str(raw_content)
    else:
        answer_en = str(raw_content) if raw_content else ""

    if not answer_en:
        answer_en = "I apologize, but I wasn't able to process your request. Please try again or ask me to create a support ticket."

    # --- 6. Translate response back to user's language ---
    try:
        answer = (
            GoogleTranslator(source="en", target=lang).translate(answer_en)
            if lang != "en" else answer_en
        )
    except Exception:
        answer = answer_en  # Fallback: return English if translation fails

    # --- 7. Record latency ---
    agent_reasoning["total_latency_seconds"] = round(time.time() - start_time, 2)

    # --- 8. Log the interaction ---
    try:
        _log_agent_interaction(
            original_query=query,
            query_language=lang,
            answer=answer,
            agent_reasoning=agent_reasoning,
        )
    except Exception:
        pass  # Logging should never break the user experience

    return answer, agent_reasoning


# ============================================================
# STEP 5: Interaction Logging — SQLite
# ============================================================

LOGS_DB_PATH = os.path.join(os.path.dirname(__file__), "agent_logs.db")


def _init_logs_db():
    """Creates the agent_logs table if it doesn't exist."""
    conn = sqlite3.connect(LOGS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_logs (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp             TEXT    NOT NULL,
            original_query        TEXT    NOT NULL,
            query_language        TEXT    DEFAULT 'en',
            tools_called          TEXT,
            retrieval_attempts    TEXT,
            grade_verdicts        TEXT,
            rewritten_queries     TEXT,
            final_answer          TEXT,
            agent_reasoning_json  TEXT,
            total_latency_seconds REAL,
            error                 TEXT
        )
    """)
    conn.commit()
    conn.close()


def _log_agent_interaction(
    original_query: str,
    query_language: str,
    answer: str,
    agent_reasoning: dict,
    error: str = None,
):
    """
    Logs a complete agent interaction to the SQLite database.
    
    Captures every stage for post-hoc review:
    - What the user asked
    - Which tool(s) were called and in what order
    - Retrieval attempts: how many, grade verdicts, rewritten queries
    - The final answer given
    - Total end-to-end latency
    - Any errors encountered
    """
    _init_logs_db()

    now = datetime.now().isoformat()
    tools_used = agent_reasoning.get("tools_used", [])
    retrieval_attempts = agent_reasoning.get("retrieval_attempts", [])

    # Extract tool names in call order
    tools_called = [t["name"] for t in tools_used]

    # Extract grade verdicts from retrieval attempts
    grade_verdicts = [
        {"attempt": a.get("attempt"), "verdict": a.get("verdict"), "reason": a.get("reason")}
        for a in retrieval_attempts
    ]

    # Extract any rewritten queries
    rewritten_queries = [
        a["rewritten_query"]
        for a in retrieval_attempts
        if "rewritten_query" in a
    ]

    conn = sqlite3.connect(LOGS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO agent_logs (
            timestamp, original_query, query_language, tools_called,
            retrieval_attempts, grade_verdicts, rewritten_queries,
            final_answer, agent_reasoning_json, total_latency_seconds, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now,
            original_query,
            query_language,
            json.dumps(tools_called),
            json.dumps(retrieval_attempts, default=str),
            json.dumps(grade_verdicts),
            json.dumps(rewritten_queries),
            answer[:2000] if answer else None,  # Cap at 2000 chars to avoid bloat
            json.dumps(agent_reasoning, default=str),
            agent_reasoning.get("total_latency_seconds"),
            error,
        ),
    )
    conn.commit()
    conn.close()
