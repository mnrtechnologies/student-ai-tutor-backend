from typing import Dict, TypedDict, List
import os
import logging
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# Configure logging to write only to debug.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log', encoding='utf-8'),
        logging.FileHandler('errors.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

MAX_TOKENS = 2048

def safe_log_string(text):
    return text.encode('utf-8', errors='replace').decode('utf-8') if isinstance(text, str) else str(text)

class State(TypedDict):
    query: str
    subject: str
    latex_needed: bool
    embeddings: List[Dict]
    response_type: str
    output: str
    user_conversation: List[str]
    ai_conversation: List[str]
    context_ids: List[str]
    parent_paragraph_id: List[str]
    query_type: str
    chunk_size_tokens: int
    content_type: str
    file_type: str
    feedback: str
    chapter: str

summarize_template = PromptTemplate(
    input_variables=['history', 'embeddings', 'query', 'subject'],
    template="""Conversation history with subject context: {history}\n\nSubject: {subject}\n\nContent: {embeddings}\n\nSummarize the topic: {query}\n\n
    Make sure the summary is precise, short, and understandable for a CBSE student. Adjust the summary to the user's query."""
)

brief_template = PromptTemplate(
    input_variables=['history', 'embeddings', 'query', 'subject'],
    template="""Conversation history with subject context: {history}\n\nSubject: {subject}\n\nContent: {embeddings}\n\nBriefly explain the topic: {query}\n\n
    Make sure the explanation is precise and understandable for a CBSE student."""
)

from typing import Dict, TypedDict, List
import os
from fastapi import FastAPI
import logging
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from pydantic import BaseModel,Field
class ChatRequest(BaseModel):
    query: str = Field(..., description="User's query about Jira issues")
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# Configure logging to write only to debug.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log', encoding='utf-8'),
        logging.FileHandler('errors.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

MAX_TOKENS = 2048

def safe_log_string(text):
    return text.encode('utf-8', errors='replace').decode('utf-8') if isinstance(text, str) else str(text)

class State(TypedDict):
    query: str
    subject: str
    latex_needed: bool
    embeddings: List[Dict]
    response_type: str
    output: str
    user_conversation: List[str]
    ai_conversation: List[str]
    context_ids: List[str]
    parent_paragraph_id: List[str]
    query_type: str
    chunk_size_tokens: int
    content_type: str
    file_type: str
    feedback: str
    chapter: str

summarize_template = PromptTemplate(
    input_variables=['history', 'embeddings', 'query', 'subject'],
    template="""Conversation history with subject context: {history}\n\nSubject: {subject}\n\nContent: {embeddings}\n\nSummarize the topic: {query}\n\n
    Make sure the summary is precise, short, and understandable for a CBSE student. Adjust the summary to the user's query."""
)

brief_template = PromptTemplate(
    input_variables=['history', 'embeddings', 'query', 'subject'],
    template="""Conversation history with subject context: {history}\n\nSubject: {subject}\n\nContent: {embeddings}\n\nBriefly explain the topic: {query}\n\n
    Make sure the explanation is precise and understandable for a CBSE student."""
)

practice_template = PromptTemplate(
    input_variables=['history', 'embeddings', 'query', 'subject'],
    template="""Conversation history with subject context: {history}\n\nSubject: {subject}\n\nContent: {embeddings}\n\ntopic: {query}\n\nProvide up to 5 practice questions for the topic\n\n
    Make sure the practice questions are relevant to the topic and meet the standard of the CBSE board."""
)

step_by_step_template = PromptTemplate(
    input_variables=["history", "embeddings", 'query', 'subject'],
    template="""Conversation history with subject context: {history}\n\nSubject: {subject}\n\nContent: {embeddings}\n\nProvide a detailed step-by-step solution for: {query}\n\n
    If the query is numeric, break down the problem into smaller steps. If it's a chemistry equation, explain the steps clearly."""
)

explain_template = PromptTemplate(
    input_variables=["history", 'embeddings', 'query', 'subject'],
    template="""Conversation history with subject context: {history}\n\nSubject: {subject}\n\nContent: {embeddings}\n\nExplain the topic: {query}\n\n
    Provide a detailed explanation understandable for a CBSE student."""
)

def initialize_state(initial_data: Dict = {}) -> State:
    return {
        "query": initial_data.get("query", ""),
        "subject": initial_data.get("subject", ""),
        "latex_needed": initial_data.get("latex_needed", False),
        "embeddings": initial_data.get("embeddings", []),
        "response_type": initial_data.get("response_type", ""),
        "output": initial_data.get("output", ""),
        "user_conversation": initial_data.get("user_conversation", []),
        "ai_conversation": initial_data.get("ai_conversation", []),
        "context_ids": initial_data.get("context_ids", []),
        "parent_paragraph_id": initial_data.get("parent_paragraph_id", []),
        "query_type": initial_data.get("query_type", ""),
        "chunk_size_tokens": initial_data.get("chunk_size_tokens", 0),
        "content_type": initial_data.get("content_type", ""),
        "file_type": initial_data.get("file_type", ""),
        "feedback": initial_data.get("feedback", ""),
        "chapter": initial_data.get("chapter", "")
    }

def detect_topic(state: State) -> State:
    load_dotenv()
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    state['user_conversation'].append(state['query'])

    messages = [
        {"role": "system", "content": "You are a subject classifier for the CBSE curriculum. Given the query, classify it into one of the following subjects: physics, chemistry, maths, social-science, or biology. If the query is unrelated to these subjects, respond with 'invalid'. Respond only with the subject name or 'invalid'."},
        {"role": "user", "content": state['query'].lower()}
    ]
    
    try:
        response = llm.invoke(messages)
        raw_subject = response.content.strip().lower()
        logger.info(f"Raw LLM Subject Response: {raw_subject}")

        latex_subjects = {"physics", "chemistry", "maths"}
        valid_subjects = {"physics", "chemistry", "maths", "social-science", "biology"}

        if raw_subject in valid_subjects:
            state["subject"] = raw_subject
            state["latex_needed"] = raw_subject in latex_subjects
            state["output"] = "Pending Response"
        else:
            fallback_messages = [
                {"role": "system", "content": "You are a subject classifier. The previous classification was ambiguous. Re-evaluate the query and determine the subject based on these rules: if 'accommodation' is mentioned, classify as 'physics'; otherwise, respond with 'invalid'."},
                {"role": "user", "content": state['query'].lower()}
            ]
            fallback_response = llm.invoke(fallback_messages)
            fallback_subject = fallback_response.content.strip().lower()
            if fallback_subject in valid_subjects:
                state["subject"] = fallback_subject
                state["latex_needed"] = fallback_subject in latex_subjects
                state["output"] = "Pending Response"
            else:
                state['subject'] = ''
                state['latex_needed'] = False
                state['output'] = "I'm sorry, I can only answer questions about physics, chemistry, math, social science, or biology."

        if state['subject']:
            state['ai_conversation'].append(f"Subject: {state['subject']}")
        logger.info(f"Detected Subject: {state['subject']}")
    except Exception as e:
        logger.error(f"Error processing query '{state['query']}': {e}")
        state['subject'] = ''
        state['latex_needed'] = False
        state['output'] = "Sorry, I can't assist with that. Please try again later"

    return state

def classify_query_type(state: State) -> State:
    state['user_conversation'].append(state['query'])

    if len(state['user_conversation']) == 1:
        state['query_type'] = 'new_query'
    else:
        query_lower = state['query'].lower()
        last_subject = next((msg.split(":")[1] for msg in state['ai_conversation']
                             if msg.startswith("Subject:")), None)

        relevant_history = state['user_conversation'] + state['ai_conversation']

        if last_subject and state['subject'] != last_subject:
            state['query_type'] = 'new_query'
        else:
            load_dotenv()
            llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
            history = '\n'.join(relevant_history)
            prompt = f"""History: {history}
Query: {query_lower}

Based on the query and conversation history, classify this as one of the following:
1. 'new_query'
2. 'feedback'
3. 'continuous_query'

Respond with only one option."""
            try:
                response = llm.invoke([{"role": "system", "content": "Classify the query type."}, {"role": "user", "content": prompt}])
                state['query_type'] = response.content.strip().lower()
                logger.info(f"LLM Classification Response: {response.content.strip()}")
            except Exception as e:
                logger.error(f"Error in LLM classification for query '{state['query']}': {e}")
                state['query_type'] = 'continuous_query'
        
        if last_subject and last_subject != state['subject']:
            relevant_history = [msg for msg in relevant_history if msg.startswith(f"Subject: {state['subject']}") or not msg.startswith('Subject: ')]
        
        state['user_conversation'] = [msg for msg in relevant_history if msg in state['user_conversation']]
        state['ai_conversation'] = [msg for msg in relevant_history if msg in state['ai_conversation']]

    if state["embeddings"]:
        state["chapter"] = state["embeddings"][0].get("chapter", "")
        state["chunk_size_tokens"] = state["embeddings"][0].get("chunk_size_tokens", 0)
        state["content_type"] = state["embeddings"][0].get("content_type", "")
        state["context_ids"] = state["embeddings"][0].get("context_ids", [])
        state["parent_paragraph_id"] = state["embeddings"][0].get("parent_paragraph_id", [])
        state["file_type"] = state["embeddings"][0].get("file_type", "")

    logger.info(f"Classified Query Type: {state['query_type']}")
    return state

def retrieve_embeddings(state: State) -> State:
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    subject_indexes = {
        "physics": "physics-index",
        "chemistry": "chemistry-index",
        "maths": "maths-index",
        "social-science": "social-science-index",
        "biology": "biology-index"
    }
    
    if not state['subject']:
        logger.error(f"No valid subject set for query '{state['query']}'")
        state['embeddings'] = []
        state['output'] = "No valid subject detected."
        return state
    
    index_name = subject_indexes.get(state['subject'])
    if not index_name:
        logger.error(f"No Pinecone index found for subject '{state['subject']}'")
        state['embeddings'] = []
        state['output'] = "No index found for subject."
        return state
    
    index = pc.Index(index_name)
    
    try:
        # Log index stats for debugging
        stats = index.describe_index_stats()
        logger.info(f"Index '{index_name}' Stats: {stats}")

        # Normalize query
        query_text = state['query'].lower().replace("can you explain", "").strip()
        logger.info(f"Normalized Query for Embedding: {query_text}")

        response = client.embeddings.create(
            input=query_text,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        full_retrieval_modes = {'explain', 'step_by_step'}
        relevant_retrieval_modes = {'summarize', 'brief'}

        filter_dict = {"subject": state['subject'], "file_type": "markdown"}
        if state['query_type'] != 'new_query':
            current_parent_ids = state.get("parent_paragraph_id", [])
            current_context_ids = state.get("context_ids", [])
            if current_parent_ids:
                if isinstance(current_parent_ids, list) and len(current_parent_ids) == 1:
                    filter_dict["parent_paragraph_id"] = current_parent_ids[0]
                else:
                    filter_dict["parent_paragraph_id"] = {"$in": current_parent_ids}
            if current_context_ids:
                filter_dict["context_ids"] = {"$in": current_context_ids}

        logger.info(f"Retrieval Parameters - Query: {query_text}, Subject: {state['subject']}, Response Type: {state['response_type']}")
        logger.info(f"Filter Dictionary: {filter_dict}")

        top_k = 10  # Increased for broader retrieval
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter=filter_dict, namespace='default')

        matches = results.get("matches", [])
        state["embeddings"] = [{"score": hit.score, **hit.metadata} for hit in matches]
        
        logger.info(f"Number of chunks retrieved: {len(matches)}")
        if matches:
            for idx, hit in enumerate(matches, 1):
                metadata = hit.metadata
                logger.info(f"\nChunk {idx} Details:")
                logger.info(f"Score: {hit.score}")
                logger.info(f"Content ID: {metadata.get('content_id', 'Not found')}")
                logger.info(f"Original Text: {safe_log_string(metadata.get('original_text', 'Not found'))}")
                logger.info(f"Chapter: {metadata.get('chapter', 'Not found')}")
                logger.info(f"Content Type: {metadata.get('content_type', 'Not found')}")
                logger.info(f"Context IDs: {metadata.get('context_ids', [])}")
                logger.info(f"Parent Paragraph ID: {metadata.get('parent_paragraph_id', [])}")
                logger.info(f"File Type: {metadata.get('file_type', 'Not found')}")
                logger.info(f"Chunk Size Tokens: {metadata.get('chunk_size_tokens', 0)}")
                logger.info("-" * 80)
        else:
            logger.info("No matches retrieved for the query.")
            # Fallback: Try a broader query
            broad_query = " ".join(query_text.split()[:2])  # Use first two words for broader match
            response = client.embeddings.create(input=broad_query, model="text-embedding-3-small")
            broad_embedding = response.data[0].embedding
            results = index.query(vector=broad_embedding, top_k=5, include_metadata=True, filter={"subject": state['subject'], "file_type": "markdown"}, namespace='default')
            matches = results.get("matches", [])
            state["embeddings"] = [{"score": hit.score, **hit.metadata} for hit in matches]
            logger.info(f"Fallback query '{broad_query}' retrieved {len(matches)} chunks.")

        if state["embeddings"]:
            top_chunk = max(state["embeddings"], key=lambda x: x['score'])
            state["context_ids"] = top_chunk.get("context_ids", [])
            state['chunk_size_tokens'] = top_chunk.get("chunk_size_tokens", 0)
            state['chapter'] = top_chunk.get("chapter", "")
            state['content_type'] = top_chunk.get("content_type", "")
            state['parent_paragraph_id'] = top_chunk.get("parent_paragraph_id", [])
            state['file_type'] = top_chunk.get("file_type", "")

    except Exception as e:
        logger.error(f"Error retrieving embeddings for query '{query_text}': {e}")
        state['embeddings'] = []
    
    return state

def decide_next_node(state: State) -> Dict:
    load_dotenv()
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    history = '\n'.join(state['user_conversation'] + state['ai_conversation'])
    prompt = f"""History: {history}\nQuery: {state['query'].lower()}\nBased on the query and history, decide the next action. Options are: 'summarize', 'explain', 'step_by_step', 'practice', 'brief', 'summarize_existing'. 
    For queries asking to 'calculate' or 'find' a numerical value, choose 'step_by_step'.
    Respond with only one option."""

    try:
        response = llm.invoke([{
            "role": "system",
            "content": "Decide the next node"
        }, {
            "role": "user",
            "content": prompt
        }])

        next_node = response.content.strip().lower()
        valid_nodes = {'summarize', 'explain', 'step_by_step', 'practice', 'brief', 'summarize_existing'}

        state["response_type"] = next_node if next_node in valid_nodes else 'step_by_step'
        state["output"] = ""
        logger.info(f"Decided Next Node: {next_node}")
        return {"next_node": state["response_type"]}
    except Exception as e:
        logger.error(f"Error deciding next node for '{state['query']}': {e}")
        state["response_type"] = 'step_by_step'
        state["output"] = ""
        return {"next_node": 'step_by_step'}

def node_summarize(state: State) -> State:
    load_dotenv()
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    history = "\n".join(state["user_conversation"] + state["ai_conversation"])
    if not state["embeddings"]:
        if state["output"]:  # Fallback response already set
            state["ai_conversation"].append(state["output"])
            return state
        state["output"] = "No relevant content found to summarize."
        state["ai_conversation"].append(state["output"])
        return state
    
    top_chunk = max(state["embeddings"], key=lambda x: x['score'])
    chunk_text = top_chunk.get("original_text", "")

    prompt = summarize_template.format(history=history, embeddings=chunk_text, query=state['query'], subject=state['subject'])
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        state["output"] = response.content.strip()
        logger.info(f"Node: summarize, User Query: {state['query']}, Retrieved Text: {chunk_text}, LLM Response: {state['output']}")
    except Exception as e:
        logger.error(f"Error in summarize node for query '{state['query']}': {e}")
        state["output"] = "Sorry, couldn't summarize the content."

    state["ai_conversation"].append(state["output"])
    return state

def node_step_by_step(state: State) -> State:
    load_dotenv()
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    history = "\n".join(state["user_conversation"] + state["ai_conversation"])
    if not state["embeddings"]:
        if state["output"]:  # Fallback response already set
            state["ai_conversation"].append(state["output"])
            return state
        state["output"] = "No relevant content found for step-by-step solution."
        state["ai_conversation"].append(state["output"])
        return state
    
    top_chunk = max(state["embeddings"], key=lambda x: x['score'])
    chunk_text = top_chunk.get("original_text", "")

    prompt = step_by_step_template.format(history=history, embeddings=chunk_text, query=state['query'], subject=state['subject'])
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        output = response.content.strip()
        if state["latex_needed"]:
            output = f"$${output}$$"
        state["output"] = output
        logger.info(f"Node: step_by_step, User Query: {state['query']}, Retrieved Text: {chunk_text}, LLM Response: {output}")
    except Exception as e:
        logger.error(f"Error in step_by_step node for query '{state['query']}': {e}")
        state["output"] = "Sorry, couldn't provide a step-by-step solution."

    state["ai_conversation"].append(state["output"])
    return state

def node_practice(state: State) -> State:
    load_dotenv()
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    history = "\n".join(state["user_conversation"] + state["ai_conversation"])
    if not state["embeddings"]:
        state["output"] = "No relevant content found for practice questions."
        state["ai_conversation"].append(state["output"])
        return state
    
    top_chunk = max(state["embeddings"], key=lambda x: x['score'])
    chunk_text = top_chunk.get("original_text", "")

    prompt = practice_template.format(history=history, embeddings=chunk_text, query=state["query"], subject=state['subject'])
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        state["output"] = response.content.strip()
        logger.info(f"Node: practice, User Query: {state['query']}, Retrieved Text: {chunk_text}, LLM Response: {state['output']}")
    except Exception as e:
        logger.error(f"Error in practice node for query '{state['query']}': {e}")
        state["output"] = "Sorry, couldn't generate practice questions."

    state["ai_conversation"].append(state["output"])
    return state

def node_explain(state: State) -> State:
    load_dotenv()
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    history = "\n".join(state["user_conversation"] + state["ai_conversation"])
    if not state["embeddings"]:
        if state["output"]:  # Fallback response already set
            state["ai_conversation"].append(state["output"])
            return state
        state["output"] = "No relevant content found to explain."
        state["ai_conversation"].append(state["output"])
        return state
    
    top_chunk = max(state["embeddings"], key=lambda x: x['score'])
    chunk_text = top_chunk.get("original_text", "")

    prompt = explain_template.format(history=history, embeddings=chunk_text, query=state["query"], subject=state["subject"])
    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        state["output"] = response.content.strip()
        logger.info(f"Node: explain, User Query: {safe_log_string(state['query'])}, Retrieved Text: {safe_log_string(chunk_text)}, LLM Response: {safe_log_string(state['output'])}")
    except Exception as e:
        logger.error(f"Error in explain node for query '{state['query']}': {e}")
        state["output"] = "Sorry, couldn't explain the content."

    state["ai_conversation"].append(state["output"])
    return state

def handle_feedback(state: State) -> State:
    if not state["feedback"]:
        return state
    state["user_conversation"].append(state["feedback"])
    state["query"] = state["feedback"]

    current_subject = state.get("subject", "")
    state = classify_query_type(state)
    
    if not state["subject"] and current_subject:
        state["subject"] = current_subject
        state["latex_needed"] = current_subject in {"physics", "chemistry", "maths"}

    if state["query_type"] == "feedback":
        load_dotenv()
        llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
        history = "\n".join(state["user_conversation"] + state["ai_conversation"])
        prompt = f"""History: {history}
Query: {state['feedback']}

Decide the most appropriate action: 'explain', 'step_by_step', 'practice', 'summarize'.
Respond with only one option."""
        try:
            response = llm.invoke([{"role": "system", "content": "Decide the action for feedback."}, {"role": "user", "content": prompt}])
            action = response.content.strip().lower()
            state["response_type"] = action if action in ['explain', 'step_by_step', 'practice', 'summarize'] else 'explain'
        except Exception as e:
            logger.error(f"Error in feedback handling for '{state['feedback']}': {e}")
            state["response_type"] = 'explain'
    return state

def create_workflow() -> CompiledStateGraph:
    workflow = StateGraph(State)

    workflow.add_node("detect_topic", detect_topic)
    workflow.add_node("classify_query_type", classify_query_type)
    workflow.add_node("retrieve_embeddings", retrieve_embeddings)
    workflow.add_node("decide_next_node", decide_next_node)
    workflow.add_node("summarize", node_summarize)
    workflow.add_node("explain", node_explain)
    workflow.add_node("step_by_step", node_step_by_step)
    workflow.add_node("practice", node_practice)
    workflow.add_node("handle_feedback", handle_feedback)

    workflow.set_entry_point("detect_topic")
    workflow.add_edge("detect_topic", "classify_query_type")
    workflow.add_edge("classify_query_type", "retrieve_embeddings")
    workflow.add_edge("retrieve_embeddings", "decide_next_node")

    def route_next_node(state: State) -> str:
        return state.get("next_node", "step_by_step")

    workflow.add_conditional_edges(
        "decide_next_node",
        route_next_node,
        {
            "summarize": "summarize",
            "explain": "explain",
            "step_by_step": "step_by_step",
            "practice": "practice",
            "brief": "explain",
            "summarize_existing": "summarize"
        }
    )

    workflow.add_edge("summarize", "handle_feedback")
    workflow.add_edge("explain", "handle_feedback")
    workflow.add_edge("step_by_step", "handle_feedback")
    workflow.add_edge("practice", "handle_feedback")
    workflow.add_edge("handle_feedback", END)

    return workflow.compile()



app=FastAPI(title="Student AI tutor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "https://pm-ai-tool.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the chatbot API"}

@app.post('/chatbot')
def chat_endpoint(request: ChatRequest):
    """
    Process a student query and return appropriate response
    """
    try:
        # Initialize state properly using the initialize_state function
        state = initialize_state({'query': request.query})
        
        # Create and run the workflow
        langgraph_app = create_workflow()
        result = langgraph_app.invoke(state)
        print(f"DEBUG - Final result: {result}")
        
        return {
            "query": request.query,
            "response": result.get('output', 'No response generated'),
            "status": "success"
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {
            "query": request.query,
            "error": str(e),
            "status": "error"
        }
if __name__ == "__main__":
    import uvicorn
    import socket
    
    def find_free_port():
        """Find a free port to use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    # Check if running on Render (or other cloud platform)
    is_production = os.environ.get("RENDER") or os.environ.get("PORT")
    
    if is_production:
        port = int(os.environ.get("PORT", 8000))
        host = "0.0.0.0"
        print(f"Production mode: Starting server on {host}:{port}")
        uvicorn.run("main:app", host=host, port=port)
    else:
        # For local development, find a free port
        port = find_free_port()
        host = "127.0.0.1"
        print(f"Local development: Starting server on {host}:{port}")
        print(f"Access your API at: http://localhost:{port}")
        print(f"API docs available at: http://localhost:{port}/docs")
        uvicorn.run("main:app", host=host, port=port, reload=True)
