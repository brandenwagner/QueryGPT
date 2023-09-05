# Getting Started
Create a virtualenv of your choice (venv or conda)

pip install -r requirements

Model is CodeLLama from facebookresearch
Used LlamaCpp to convert the model and quantize it


[LlamaCpp] (https://github.com/ggerganov/llama.cpp.git)
[CodeLlama] (https://github.com/facebookresearch/codellama.git)

./convert.py {downloaded_model} --outfile {converted_model} --outtype f16
./quantize {coverted_model} {output_model} {quant_type}

Place quantized model in models dir

Find your source documents and copy them into the source_documents folder
Run ingest.py to built the vectordb and embeddings

now you can run ask.py to use CodeLlama offline to query your docs



Notes:
Models to try
TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q2_K.gguf

Use this tokenizer
Trelis/Llama-2-7b-chat-hf-function-calling-GGML

get a pgdump --schema-only and ingest... maybe train on it?