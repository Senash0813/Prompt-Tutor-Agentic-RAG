from langchain.tools import Tool
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_db.as_retriever(search_kwargs={"k": 4})

def retriever_tool_func(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found."
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = Tool(
    name="Retriever",
    func=retriever_tool_func,
    description="Fetch relevant document chunks from the FAISS knowledge base."
)


prompt_llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="openai/gpt-oss-20b"
)

def prompt_generator_func(query: str) -> str:
    """Generate custom prompts based on user requirements."""
    generation_prompt = f"""
    You are an expert prompt engineer. Create a well-structured prompt based on the following request:
    
    Request: {query}
    
    Provide a clear, effective prompt that can be used for the specified purpose. Include:
    1. Clear instructions
    2. Context setting
    3. Expected output format
    
    Generated Prompt:
    """
    
    response = prompt_llm.invoke(generation_prompt)
    return response.content

prompt_generator_tool = Tool(
    name="PromptGenerator",
    func=prompt_generator_func,
    description="Generate custom prompts, templates, and examples for various AI tasks like chatbots, content creation, etc."
)


def query_classifier_func(query: str) -> str:
    """Classify queries to determine the appropriate tool."""
    retrieval_keywords = ["what is", "define", "explain", "describe", "tell me about"]
    generation_keywords = ["create", "generate", "make", "build", "design", "write"]
    
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in generation_keywords):
        return "generation"
    elif any(keyword in query_lower for keyword in retrieval_keywords):
        return "retrieval"
    else:
        return "unknown"

query_classifier_tool = Tool(
    name="QueryClassifier",
    func=query_classifier_func,
    description="Classify user queries to determine if they need information retrieval or content generation."
)