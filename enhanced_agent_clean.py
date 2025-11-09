"""
Enhanced LangGraph Agent with:
- Calculator, Date, Weather, Web Search tools
- RAG with Neo4j (BM25 + Vector search)
- Memory persistence with checkpointing
- Logging for debugging
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated
from typing import Literal
import operator
import requests

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model initialization
api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.1,
    max_retries=2,
    google_api_key=api_key,
)
logger.info("Model initialized")

# Tools
@tool
def calculator(expression: str) -> str:
    """Performs mathematical calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        logger.info(f"Calculator: {expression} = {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        return f"Error: {str(e)}"


@tool
def get_date() -> str:
    """Returns today's date in YYYY-MM-DD format."""
    date = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Date requested: {date}")
    return date


@tool
def get_weather(city: str) -> str:
    """Fetches weather for a city using OpenWeather API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY not found"
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        weather = data['weather'][0]['description']
        temp = data['main']['temp']
        feels = data['main']['feels_like']
        humidity = data['main']['humidity']
        
        result = f"Weather in {city}: {weather}, {temp}°C (feels {feels}°C), {humidity}% humidity"
        logger.info(f"Weather fetched for {city}")
        return result
    except Exception as e:
        logger.error(f"Weather error for {city}: {e}")
        return f"Error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Performs web search using SerpAPI."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not found"
    
    try:
        url = "https://serpapi.com/search"
        params = {"q": query, "api_key": api_key, "engine": "google"}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if 'organic_results' in data:
            for result in data['organic_results'][:3]:
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No description')
                results.append(f"{title}: {snippet}")
        
        if results:
            logger.info(f"Web search completed: {query}")
            return f"Search results for '{query}':\n" + "\n".join(results)
        return f"No results found for '{query}'"
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Error: {str(e)}"


@tool
def search_knowledge(query: str) -> str:
    """Searches Neo4j knowledge graph using BM25 + Vector hybrid search."""
    try:
        from langchain_community.vectorstores import Neo4jVector
        from langchain_huggingface import HuggingFaceEmbeddings
        
        logger.info(f"RAG search initiated: {query}")
        logger.info("Using hybrid search: Vector similarity + BM25 keyword matching")
        
        neo4j_url = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not neo4j_password:
            logger.error("Neo4j password not configured")
            return "Error: NEO4J_PASSWORD not set"
        
        logger.info("Loading HuggingFace embeddings model")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Embeddings model loaded successfully")
        
        logger.info(f"Connecting to Neo4j at {neo4j_url}")
        vector_store = Neo4jVector.from_existing_index(
            embedding=embeddings,
            url=neo4j_url,
            username=neo4j_user,
            password=neo4j_password,
            index_name="document_chunks"
        )
        logger.info("Connected to Neo4j vector store")
        
        logger.info("Converting query to vector embedding")
        logger.info("Executing hybrid search (Vector + BM25)")
        docs = vector_store.similarity_search(
            query, 
            k=3,
            search_type="hybrid"
        )
        
        if not docs:
            logger.warning(f"No results found for: {query}")
            logger.info("Vector search: 0 semantically similar chunks found")
            logger.info("BM25 search: 0 keyword matches found")
            return "No relevant information found in knowledge base."
        
        logger.info(f"Vector search: Found {len(docs)} semantically similar chunks")
        logger.info(f"BM25 search: Matched keywords across {len(docs)} document chunks")
        
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            logger.info(f"Result {i+1}: Source={source}")
            logger.info(f"Result {i+1}: Chunk preview: {doc.page_content[:150]}...")
            results.append(f"[Source {i+1}]: {doc.page_content}")
        
        logger.info(f"RAG search completed: Returning {len(docs)} results to LLM")
        
        return "\n\n".join(results)
        
    except ImportError:
        logger.error("Missing Neo4j libraries - install neo4j langchain-community")
        return "Error: Install neo4j langchain-community"
    except Exception as e:
        logger.error(f"RAG search failed with error: {e}")
        return f"Error searching knowledge: {str(e)}"


# Tool registration
tools = [calculator, get_date, get_weather, web_search, search_knowledge]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

logger.info(f"Tools registered: {[t.name for t in tools]}")

# State definition
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def llm_call(state: dict):
    """LLM decides which tools to use."""
    logger.info(f"LLM call #{state.get('llm_calls', 0) + 1}")
    
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant with access to calculator, date, weather, web search, and knowledge base search tools.")]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


def tool_node(state: dict):
    """Executes tool calls."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        logger.info(f"Executing tool: {tool_call['name']}")
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide whether to continue or stop."""
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END


# Graph with memory
memory = MemorySaver()

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile(checkpointer=memory)

logger.info("Agent compiled with memory persistence")

# Interactive mode
def run_agent():
    """Interactive agent with persistent memory."""
    print("\nLangGraph Agent - Interactive Mode")
    print("="*70)
    print("Available tools: Calculator, Date, Weather, Web Search, Knowledge Base")
    print("Memory: Enabled | Logs: agent.log")
    print("Type 'quit' to exit")
    print("="*70 + "\n")
    
    thread_id = "session-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    logger.info("Interactive session started")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q', '']:
            if user_input.lower() in ['quit', 'exit', 'q']:
                logger.info("Session ended by user")
                print("\nGoodbye!")
            break
        
        logger.info(f"User query: {user_input}")
        print("\nProcessing...\n")
        
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            response = result["messages"][-1].content
            print("="*70)
            print(f"Agent: {response}")
            print("="*70 + "\n")
            logger.info(f"Response delivered")
            
        except Exception as e:
            logger.error(f"Error during invocation: {e}")
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    logger.info("Agent startup")
    print("\nStarting Enhanced LangGraph Agent...")
    print("Version: LangGraph 1.0.2 | Gemini 2.0 Flash | HuggingFace Embeddings")
    print("Features: Calculator, Weather, Search, RAG (BM25+Vector), Memory\n")
    
    run_agent()