from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

#our first step is to load our data whther it may be text data, web page , json file or anything

loader = TextLoader("sample.txt")
documents = loader.load()

#now we have to split the documents into small chunks 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

#we have to create embeddings for our chunks and then store them in FAISS 

embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")
db = FAISS.from_documents(chunks, embedding_model)

#we will create retriever for this vectorstore in order to retrieve relevant chunks

retriever = db.as_retriever()

#now we will load the LLM 

llm = OllamaLLM(model="deepseek-r1:1.5b")

chain = RetrievalQA.from_chain_type(llm,retriever=retriever)

query = "simplify the new features of an apple iphone"

response = chain.invoke(query)

print(response['result'])