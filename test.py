from langchain_community.embeddings import HuggingFaceEmbeddings  # for embeddings
from langchain_qdrant import QdrantVectorStore  # modern Qdrant integration
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  # updated import for Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

class RAG:
    def __init__(self, qdrant_url, embedding_model, llama_model):
        self.qdrant_url = qdrant_url
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model,
                                           model_kwargs={'device': 'cpu'})
        self.llm = OllamaLLM(model=llama_model)

    def split_text_into_chunks(self, texts):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        chunks = text_splitter.split_text(texts)
        return chunks
    
    def ingest_vector(self, text_array, collection_name):
        doc_array = []
        for txt in text_array:
        # Chunk the texts into smaller ones
            chunks = self.split_text_into_chunks(txt["text"])
        
        # Convert to LangChain Document objects with metadata
            doc_array.extend([
            Document(page_content=chunk, metadata=txt["metadata"])
            for chunk in chunks
        ])

        client = QdrantClient(url=self.qdrant_url)

        qdrant = QdrantVectorStore.from_documents(
            documents=doc_array,
            embedding=self.embeddings,  # must be a LangChain-compatible embedding
            client=client,
            collection_name=collection_name,
            force_recreate=True
        )

        print("Inserted")
        return qdrant

    def answer_question(self, question, collection_name):

        qdrant_vector = QdrantVectorStore.from_existing_collection(collection_name=collection_name, url=self.qdrant_url,
                                                                   embedding=self.embeddings)
        # Defines the search type and number of relevant document to return of the document retriever
        retriever = qdrant_vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # This prompt will be used to query the LLM using the context retrieved from vector store
        prompt = """
        1. Use the following pieces of context to answer the question at the end.
        2. If you don't know the answer, just say that "Cannot answer the question!" but don't make up an answer on your own.\n
        3. If the answer is found, Keep the answer crisp and limited to 3,4 sentences.

        Context: {context}

        Question: {question}

        Helpful Answer:"""

        # Initializes langchain prompt template from text
        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

        # Formats the user input before passing to LLM model.
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=QA_CHAIN_PROMPT,
            callbacks=None,
            verbose=True)

        # Formats the retrieved information from Vector DB before passing them to LLM model as source
        document_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="Content:{page_content}",
        )

        # Combines the retrieved documents into a single documents for the LLM model.
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
            callbacks=None,
        )

        # Combines the document store with all the prompts
        qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            verbose=True,
            retriever=retriever,
            return_source_documents=True,
        )

        # Finally user defined question is given as query and the response is returned.
        res = qa(question)
        return {"answer": res["result"]}
    
qdrant_url = "http://localhost:6333" # The exact line from your file
collection_name = "test_collection_1"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llama_model = "llama3.2:1b"
rag = RAG(qdrant_url=qdrant_url, embedding_model=embedding_model, llama_model = llama_model)

texts = [
    {"text": 'India boasts a diverse landscape with several prominent mountain ranges, including the Himalayas, Karakoram, Western Ghats, Eastern Ghats, and Aravalli, each with unique characteristics and significance.', "metadata":{'source': 'wikipedia.com'}},
    {"text": 'In India, the Himalayas are grouped into three distinct ranges: the Greater Himalaya Range, the Middle Himalaya Range, and the Outer Himalaya Range. All are found in the northern part of the country.', "metadata":{'source': 'study.com'}},
    {"text": 'The Himalayan Mountain Range is an ideal location for mountaineering expeditions. The glaciers are one of the favourite attractions for the mountaineers. When you go for a hiking expedition, you will face some of the most difficult challenges that you have ever come across. The glaciers in Kashmir and Ladakh will test your endurance. The Siachen Glacier is regarded as one of the biggest glaciers away from the Arctic zones.', "metadata": {'source': 'https://www.mapsofindia.com/mountains/'}}
    ]
ans = rag.ingest_vector(text_array=texts, collection_name=collection_name)

ques = "What are the places the people are watching the video from?"
ans = rag.answer_question(question=ques, collection_name=collection_name)