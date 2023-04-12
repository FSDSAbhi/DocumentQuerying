#get_ipython().system('pip install langchain')
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Install package
#get_ipython().system('pip install "unstructured[local-inference]"')

##Loading the data
loader = UnstructuredPDFLoader("/Users/abhilashmarecharla/Desktop/gpt-pinecone/documents/PrinciplesofManagement.pdf")
data = loader.load()

print("The total number of documents are: ",len(data))
print("The total number of characters in the document are:",len(data[0].page_content))


# # Chunking the original document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
texts = text_splitter.split_documents(data)
print("The number of chunks: ",len(texts))

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

'''
#get_ipython().system('pip install pinecone-client')
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "1a522964-dfde-4870-91eb-8a93b2aba674"
'''
PINECONE_API_ENV = "us-east1-gcp"
embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

##Initialize pinecone
pinecone.init(api_key = PINECONE_API_KEY,
             environment = PINECONE_API_ENV
             )
index_name = "testindex1"

docsearch = Pinecone.from_texts([t.page_content for t in texts],embeddings,index_name = index_name)

#get_ipython().system('pip install tiktoken')

query = "What are the basic principles for management?"
docs = docsearch.similarity_search(query, include_metadata = True)

print("The documents from which the answer is obtained are:",len(docs))

## Query the documents

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

llm = OpenAI(temperature=0,openai_api_key = OPENAI_API_KEY)
chain = load_qa_chain(llm,chain_type="stuff")

query = "What are the basics management?"
docs = docsearch.similarity_search(query, include_metadata = False)
chain.run(input_documents=docs,question=query)

query = "What is planning?"
docs = docsearch.similarity_search(query, include_metadata = False)
chain.run(input_documents=docs,question=query)

query = "What is organizing?"
docs = docsearch.similarity_search(query, include_metadata = False)
chain.run(input_documents=docs,question=query)

query = "How are relationships, responsibilities and connections interlinked with eachother"
docs = docsearch.similarity_search(query, include_metadata = False)
chain.run(input_documents=docs,question=query)
