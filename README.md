# Prompt-Tutor Agentic RAG

**Prompt-Tutor** is an **Agentic Retrieval-Augmented Generation (RAG)** system that leverages a vector store and LLM-based agent behavior to answer questions about prompt engineering — and even generate sample prompts based on users needs.

---

##  Features

- **Agentic RAG pipeline**: Dynamically retrieves relevant documents, uses tools for reasoning, and composes answers with the flexibility of an agent.
- **Custom Tooling**: Retrieves from a FAISS vector store using a HuggingFace embedding model, creates prompts using a custom tool

---

##  How to Run Locally

### 1. Clone this repository
```bash
git clone https://github.com/Senash0813/Prompt-Tutor-Agentic-RAG.git
cd Prompt-Tutor-Agentic-RAG

pip install -r requirements.txt

```
### 2. run the notebook
run the .ipynb Notebook from begining. (I have commented the code for using the web source of the original resource)

**Note:**
- This project uses ChatGroq (Groq Cloud free-tier). Most models there do not support structured ReAct tool-calling (Thought / Action / Observation).
- For full agentic transparency (recommended), use a model that supports ReAct-style structured calls via ZERO_SHOT_REACT_DESCRIPTION.




