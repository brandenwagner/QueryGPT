#!/usr/bin/env python3

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
import chromadb
import os
import time

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME",'all-MiniLM-L6-v2')
model_path = os.environ.get('MODEL_PATH','models/codellama-7b-instruct.Q2_K.gguf')
persist_directory = os.environ.get('PERSIST_DIRECTORY','db')

verbose = bool(os.environ.get('VERBOSE',False))
streaming = bool(os.environ.get('STREAMING',False))
cite_source = bool(os.environ.get('CITE_SOURCE', False))

max_tokens = int(os.environ.get('MAX_TOKENS', 1000))
model_n_ctx = int(os.environ.get('MODEL_N_CTX', 1024))
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))


# Define the Chroma settings
CHROMA_SETTINGS = chromadb.config.Settings(
        persist_directory=persist_directory,
        anonymized_telemetry=True
)

def main():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if streaming else [StreamingStdOutCallbackHandler()]
    
    # Prepare the LLM
    llm = LlamaCpp(model_path=model_path, max_tokens=max_tokens, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=verbose)
   
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=cite_source)
    
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query == "quit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], res['source_documents'] if cite_source else []
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        if cite_source:
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)

if __name__ == "__main__":
    main()
