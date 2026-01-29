import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pickle

AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
  


from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
#from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from base64 import b64decode



# ---------------------------------------
# CONFIG
# ---------------------------------------
#DOCUMENTS = [
#    "Project_Status_Report_IQSight_Kamino2",
##    "Project_Status_Report_IQSight_Rosen",
 #   "Project_Status_Report_IQSight_Interius",
 #   "Project_Status_Report_IQSight_tripple_power",
 #   "Project_Status_Report_IQSight_webb",
#]

DOCUMENTS = [
    "Project_Status_Report_IQSight_Kamino2",
]

CHROMA_PARENT_DIR = "chroma_store"


# ---------------------------------------
# Load All Precomputed Vectorstores
# ---------------------------------------

@st.cache_resource
def load_all_retrievers():

    embedding_fn = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


    retrievers = {}

    for docname in DOCUMENTS:
        persist_dir = os.path.join(CHROMA_PARENT_DIR, docname)
        docstore_path = f"docstore_{docname}.pkl"

        if not os.path.exists(persist_dir):
            st.error(f"❌ Missing vectorstore directory: {persist_dir}")
            continue

        if not os.path.exists(docstore_path):
            st.error(f"❌ Missing docstore file: {docstore_path}")
            continue

        with open(docstore_path, "rb") as f:
            store = pickle.load(f)

        vectorstore = Chroma(
            collection_name=docname,
            persist_directory=persist_dir,
            embedding_function=embedding_fn
        )

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key="doc_id"
        )

        retrievers[docname] = retriever

    return retrievers


retrievers = load_all_retrievers()


# ---------------------------------------
# RAG utilities
# ---------------------------------------

def parse_docs(docs):
    b64, text = [], []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    ctx = kwargs["context"]
    q = kwargs["question"]

    context_text = "\n\n".join([t.text for t in ctx["texts"]])

    prompt = f"""
Answer based ONLY on this context:

{context_text}

Question: {q}
"""
    return [HumanMessage(content=prompt)]


#llm = ChatOpenAI(model="gpt-4o-mini")

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT,
    api_version="2024-02-15-preview",
    api_key=AZURE_API_KEY,
    temperature=0,
)

#router_llm = ChatOpenAI(model="gpt-4o-mini")


router_llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT,
    api_version="2024-02-15-preview",
    api_key=AZURE_API_KEY,
    temperature=0,
)


def route_projects(query, project_names):
    prompt = f"""
    You are an expert data extraction engine.

    Task: Extract project names mentioned in the user's query.

    User query:
    {query}

    Valid project names (canonical list):
    {", ".join(project_names)}

    Rules:
    - Return ONLY valid project names from the canonical list.
    - Correct minor spelling mistakes or variations to the closest valid name.
    - Ignore names that are not in the valid list.
    - Output a strict JSON array of strings, no commentary or extra text.
    - Handle punctuation, extra spaces, and casing differences.

    Example:
    User query: "Who manages Kamino 2, Interius, and rosen?"
    Output: ["Kamino2", "Interius", "Rosen"]
    """

    response = router_llm.invoke(prompt).content

    try:
        import json
        extracted = json.loads(response)
        # Ensure only valid names are returned
        return [p for p in extracted if p in project_names]
    except:
        # fallback: naive containment check
        return [p for p in project_names if p.lower() in response.lower()]
    
def filter_retrievers(query, retrievers: dict):
    # extract raw names (short project names)
    project_names = [

    (name.split("IQSight_", 1)[1] if "IQSight_" in name else name.rsplit("_", 1)[-1]).replace("_", " ")
    for name in retrievers.keys()

    ]

    print("project names")
    print(project_names)

    matched = route_projects(query, project_names)
    print("matched")
    print(matched)
    output = {}
    for key, retr in retrievers.items():
        short_name = (key.split("IQSight_", 1)[1] if "IQSight_" in key else key.rsplit("_", 1)[-1]).replace("_", " ")
        if short_name in matched:
            print("yes")
            print(short_name)
            output[key] = retr

    return output

def retrieve_from_multiple(query, retrievers_dict):
    all_docs = []
    for name, retr in retrievers_dict.items():
        docs = retr.get_relevant_documents(query)
        all_docs.extend(docs)
    return all_docs




# ---------------------------------------
# Streamlit UI
# ---------------------------------------

#st.title("📘 Multi-Document RAG Assistant")

st.set_page_config(
    page_title="IQSight PMO AI Agent",
    layout="centered"
)

st.markdown(
    """
    <style>
        body {
            background-color: #ffffff;
        }
        .stApp {
            background-color: #ffffff;
        }
        .small-text {
            font-size: 10px;
            color: #666;
        }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image("IQSIGHT Cool Gray_RGB.tif", width=300)

with st.expander("Disclaimer & Transparency Notice"):
    st.markdown(
        """
        <div class="small-text">
        <b>Usage Restrictions</b><br><br>
        Users shall not input, prompt, or retrieve any intellectual property rights
        (including copyrights), trade secrets, confidential information, know-how,
        source code, internal sensitive or critical information (particularly regarding
        IT-security), or personal data.<br><br>

        The right to use any output generated by this AI agent is strictly limited
        to commercial purposes in connection with IQSight B.V. and its affiliated companies.
        Any other use, disclosure, or distribution is prohibited.<br><br>

        Users must verify all outputs for accuracy and compliance.
        Violations of these stipulations may result in disciplinary action,
        including termination.<br><br>

        IQSight B.V. and its affiliates assume no liability for outputs or misuse.<br><br>

        <b>PMO AI Agent – Transparency Notice</b><br><br>
        The PMO AI Agent is an AI-powered tool using Retrieval-Augmented Generation (RAG)
        to assist with project management tasks. Outputs are AI-generated and may contain
        bias, errors, or hallucinations. Users must verify all information manually
        before further use. This tool does not provide legal, financial, or professional advice.
        </div>
        """,
        unsafe_allow_html=True
    )


query = st.text_input("Ask a question:")


if query:

    filtered = filter_retrievers(query, retrievers)
    if not filtered:  
        # No projects matched
        # => use all retrievers
        filtered = retrievers
    print(filtered.keys())
    docs = retrieve_from_multiple(query, filtered)

    parsed_context = parse_docs(docs)

    chain = (
        {
            "context": lambda _: parsed_context,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | llm
    )

    with st.spinner("Thinking..."):
        answer = chain.invoke(query).content

    st.subheader("Answer:")
    st.write(answer)

