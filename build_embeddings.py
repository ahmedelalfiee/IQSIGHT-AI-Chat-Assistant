import os
from dotenv import load_dotenv

load_dotenv()

# Keys
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


import uuid
import pickle
import httpx

from unstructured.partition.xlsx import partition_xlsx
from unstructured.chunking.title import chunk_by_title

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------------------------------------------
# MAIN FUNCTION: process_document()
# ---------------------------------------------------

#DOCUMENTS = [
#    "Project Status Report IQSight Kamino2.xlsm",
#    "Project Status Report IQSight Rosen.xlsm",
#    "Project Status Report IQSight Interius.xlsm",
#    "Project Status Report IQSight tripple power.xlsm",
#    "Project Status Report IQSight webb.xlsm",
#]

DOCUMENTS = [
    "Project Status Report IQSight Kamino2.xlsx",
]

CHROMA_PARENT_DIR = "chroma_store"

# ------------------------
embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)


for file in DOCUMENTS:
    print(f"\n📄 Processing: {file}")


    elements = partition_xlsx(filename=file)

    chunks = chunk_by_title(
        elements,
        max_characters=10000,
        combine_text_under_n_chars=2000,
    )

    texts = [c for c in chunks if "CompositeElement" in str(type(c))]
    tables = [c for c in chunks if "Table" in str(type(c))]

    # ------------------------
    # 2. Summarize tables/texts
    # ------------------------

    prompt = ChatPromptTemplate.from_template("""
        Summarize the table or text:

        {element}
    """)

    model = ChatGroq(
        temperature=0.5,
        model="llama-3.1-8b-instant",
    )

    summarize_chain = (
        {"element": lambda x: x}
        | prompt
        | model
        | StrOutputParser()
    )

    table_html = [t.metadata.text_as_html for t in tables]

    table_summaries = summarize_chain.batch(table_html)
    text_summaries = summarize_chain.batch(texts)

    # ------------------------
    # 3. Build vectorstore
 

    collection_name = os.path.splitext(file)[0]
    collection_name = collection_name.replace(" ", "_")
    persist_dir = os.path.join(CHROMA_PARENT_DIR, collection_name)

    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding_function
    )

    store = InMemoryStore()
    doc_ids = [str(uuid.uuid4()) for _ in texts]

    summary_docs = [
        Document(
            page_content=text_summaries[i],
            metadata={"doc_id": doc_ids[i]}
        )
        for i in range(len(texts))
    ]

    # Add to vector DB
    vectorstore.add_documents(summary_docs)

    # Save docstore
    store.mset(list(zip(doc_ids, texts)))

    with open(f"docstore_{collection_name}.pkl", "wb") as f:
        pickle.dump(store, f)

    vectorstore.persist()

    print(f"✅ Finished {file}")
    print(f"   → Vectorstore: {persist_dir}")
    print(f"   → Docstore: docstore_{collection_name}.pkl")

