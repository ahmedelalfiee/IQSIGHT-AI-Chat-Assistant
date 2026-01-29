Overview

Multi-Document RAG Assistant is a web-based application that leverages Retrieval-Augmented Generation (RAG) to answer user queries based on multiple project documents. It uses vector embeddings to efficiently retrieve relevant context from multiple sources and feeds them into a language model to generate precise answers.

Key features:

Supports multiple project document collections simultaneously.

Uses Chroma vector stores to store embeddings.

Integrates with Azure OpenAI for embeddings and LLM responses.

Built with Streamlit for an interactive web interface.

Designed for deployment in Azure Web App with Docker.

🏗 Project Structure
.
├── app.py                  # Main Streamlit app
├── chroma_store/           # Directory containing Chroma vectorstores
├── docstore_<project>.pkl  # Precomputed document stores
├── Dockerfile              # Dockerfile for Azure deployment
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (Azure keys)
└── README.md               # Project documentation

⚡ Features

Dynamic Retrieval: Automatically detects which project documents are relevant to a user query.

Multi-Vector Retrieval: Combines multiple vector stores for context aggregation.

Context Parsing: Separates text content and base64-encoded images.

LLM Integration: Uses AzureChatOpenAI for question-answering.

Flexible Deployment: Supports local Docker deployment and Azure Web Apps.

🛠 Installation & Setup

Clone the repository

git clone https://github.com/<your-username>/multi-document-rag-assistant.git
cd multi-document-rag-assistant


Create and activate a Python virtual environment

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows


Install dependencies

pip install -r requirements.txt


Add your Azure OpenAI credentials in .env:

AZURE_OPENAI_KEY=<your_api_key>
AZURE_OPENAI_ENDPOINT=<your_endpoint>
AZURE_OPENAI_DEPLOYMENT=<your_llm_deployment>


Ensure your Chroma vector stores and docstore files exist:

chroma_store/<project_name>/
docstore_<project_name>.pkl

🚀 Running Locally
streamlit run app.py


Access the app at http://localhost:8501.

Enter a query to retrieve information across all project documents.

🐳 Docker Deployment

Build the Docker image

docker build -t multi-document-rag-assistant .


Run the container locally

docker run -p 8501:8501 multi-document-rag-assistant


Visit http://localhost:8501 in your browser.

Deploy to Azure Web App

Push your Docker image to Azure Container Registry (ACR).

Configure the Web App to pull the image from ACR.

Set environment variables in Azure (AZURE_OPENAI_KEY, etc.).

Use port 8501 for Streamlit.

📁 Project Configuration

DOCUMENTS: List of project documents to retrieve from.

CHROMA_PARENT_DIR: Path to the vector store folder.

AzureChatOpenAI: LLM used for both routing queries and generating answers.

MultiVectorRetriever: Retrieves relevant documents across multiple collections.

⚡ Usage

Enter a natural language query in the Streamlit UI.

The app identifies the relevant project documents.

Retrieves context via Chroma embeddings.

Generates a concise answer using Azure OpenAI LLM.

📦 Dependencies

streamlit

langchain

langchain-azure

langchain-community

pickle

python-dotenv

base64

🔐 Security & Notes

Keep your Azure API key and endpoint secret. Do not commit .env to GitHub.

Chroma vector stores contain precomputed embeddings; ensure they are included in your Docker image.