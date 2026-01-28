from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader, 
    Docx2txtLoader,
    DirectoryLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class RAGPipeline:

    def __init__(self,
                 documents_dir: str = "../data/company_docs",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 persist_dir: str = "../data/company_docs/chroma_db"):
        self.documents_dir = documents_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = persist_dir

    def load_documents(self, path):
        loaders = [
            DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader),
            DirectoryLoader(path, glob="**/*.docx", loader_cls=Docx2txtLoader),
        ]

        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        return docs

    def chunk_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(docs)


    def build_vectorstore(self, docs, persist_dir):
        embeddings = OpenAIEmbeddings()
        return Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
    
    def rag_ingestion_pipeline(self):
        docs = self.load_documents(self.documents_dir)
        chunks = self.chunk_documents(docs)
        vectorstore = self.build_vectorstore(chunks, self.persist_dir)
