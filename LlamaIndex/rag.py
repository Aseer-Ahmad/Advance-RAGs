
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

from llama_index.core import VectorStoreIndex

from llama_index.llms.groq import Groq
from llama_index.core.extractors import TitleExtractor,QuestionsAnsweredExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

from llama_index.core import StorageContext, load_index_from_storage

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

import os
import getpass
import nest_asyncio
import pprint

nest_asyncio.apply()


def llama_doc_eg():
    document = Document(
        text="This is a super-customized document",
        metadata={
            "file_name": "super_secret_document.txt",
            "category": "finance",
            "author": "LlamaIndex",
        },
        # excluded_embed_metadata_keys=["file_name"],
        excluded_llm_metadata_keys=["category"],
        metadata_seperator="\n",
        metadata_template="{key}:{value}",
        text_template="Metadata:\n{metadata_str}\n-----\nContent:\n{content}",
    )
 
    print(
        "The LLM sees this: \n",
        document.get_content(metadata_mode=MetadataMode.LLM),
    )

    print(document.get_content(metadata_mode=MetadataMode.EMBED))


def readData():
    docs = SimpleDirectoryReader(input_dir=".").load_data()
    pprint.pprint(docs)
    return docs

def getTransformations(docs) :
    
    llm_transformations = Groq(model="qwen-2.5-32b", api_key=os.environ["GROQ_API_KEY"])
    text_splitter = SentenceSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)
    title_extractor = TitleExtractor(llm=llm_transformations, nodes=5)
    qa_extractor = QuestionsAnsweredExtractor(llm=llm_transformations, questions=3)

    pipeline = IngestionPipeline(
        transformations=[
            text_splitter,
            title_extractor,
            qa_extractor
        ]
    )

    nodes = pipeline.run(
        documents=docs,
        in_place=True,
        show_progress=True,
    )

    return nodes

def getEmbeddingModel():
    hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return hf_embeddings

def indexData(docs):
    hf_embeddings = getEmbeddingModel()
    nodes = getTransformations(docs)
    index = VectorStoreIndex(nodes, embed_model=hf_embeddings)

    return index

def storeIndex(index):
    index.storage_context.persist(persist_dir="./vectors")
    
def loadIndex():
    storage_context = StorageContext.from_defaults(persist_dir="./vectors")
    hf_embeddings = getEmbeddingModel()
    index_from_storage = load_index_from_storage(storage_context, embed_model=hf_embeddings)
    return index_from_storage

def query(index, query):
    llm_querying = Groq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])
    query_engine = index.as_query_engine(llm=llm_querying)
    response = query_engine.query(query)
    print(response)

def ChromaIndex():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("healthGPT")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create index
    docs = readData()
    nodes = getTransformations(docs)
    hf_embeddings = getEmbeddingModel()
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=hf_embeddings)

    return index



if __name__ == '__main__':
    llama_doc_eg()