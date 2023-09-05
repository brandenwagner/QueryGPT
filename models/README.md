The models folder is where you should store all of your models.
In general the script has been written to support llama2 models but other models
may work 

The best sample model to run locally on a M1 mac is:
[TheBloke/CodeLlama-7B-Instruct](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF)


```
llm = LlamaCpp(model_path=model_path, max_tokens=max_tokens, n_ctx=model_n_ctx, 
        n_batch=model_n_batch, callbacks=callbacks, verbose=verbose)
```