# 🤖 IQSight AI Chat Assistant

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-Enabled-0078D4?logo=microsoftazure&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)

## Overview

**IQSight AI Chat Assistant** is a web-based application that leverages Retrieval-Augmented Generation (RAG) to answer user queries based on multiple project documents. It uses vector embeddings to efficiently retrieve relevant context from multiple sources and feeds them into a language model to generate precise answers.



### Key Features
* **Multi-Source Support:** Supports multiple project document collections simultaneously.
* **Vector Storage:** Uses Chroma vector stores to persist and query embeddings.
* **Azure Integration:** Seamlessly integrates with Azure OpenAI for embeddings and LLM responses.
* **Interactive UI:** Built with Streamlit for a clean, chat-based web interface.
* **Production Ready:** Designed for deployment in Azure Web App using Docker.

---



## 🏗 Project Structure

```text
.
├── app.py                  # Main Streamlit application entry point
├── chroma_store/           # Directory containing Chroma vectorstores
├── docstore_<project>.pkl  # Precomputed document stores (Pickle format)
├── Dockerfile              # Configuration for Azure/Docker deployment
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables 
└── README.md               # Project documentation

```

## 🛠 Installation & Setup

### Clone the repository
```bash
git clone https://github.com/ahmedelalfiee/IQSIGHT-AI-Chat-Assistant.git
cd IQSIGHT-AI-Chat-Assistant
```

### Create and activate a Python virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Add your Azure OpenAI credentials in `.env`
```bash
AZURE_OPENAI_KEY=<your_api_key>
AZURE_OPENAI_ENDPOINT=<your_endpoint>
AZURE_OPENAI_DEPLOYMENT=<your_llm_deployment>
```

## 🚀 Running Locally
```bash
streamlit run app.py
```

## 🐳 Docker Deployment
```bash
docker build -t IQSIGHT-AI-Chat-Assistant .
```
## Run the container locally
```bash
docker run -p 8501:8501 IQSIGHT-AI-Chat-Assistant
```

### Deploy to Azure Web App

* Push your Docker image to **Azure Container Registry (ACR)**.
* Configure the Web App to pull the image from ACR.
* Set environment variables in Azure (`AZURE_OPENAI_KEY`, etc.).
* Use port **8501** for Streamlit.


