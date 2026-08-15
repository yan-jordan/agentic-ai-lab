from lab.models import chat_model, small_model, embedding_model

console = Console()

# default model is Qwen , but you can add different model name here
llm = chat_model()

# for waiting in console
with console.status("[bold green]Thinking...", spinner="dots"):
    response = llm.invoke("what is the capital of UnitedState? answer in one word !")

print("--- --- --- --- --- --- --- --- --- --- ---")
print(type(response))
print("--- --- --- --- --- --- --- --- --- --- ---")
print(response)
print("--- --- --- --- --- --- --- --- --- --- ---")
print(response.content)
print("--- --- --- --- --- --- --- --- --- --- ---")
print("model: " , response.response_metadata.get("model"))
print("--- --- --- --- --- --- --- --- --- --- ---")
print("total token : " , response.usage_metadata.get("total_tokens"))
print("--- --- --- --- --- --- --- --- --- --- ---")