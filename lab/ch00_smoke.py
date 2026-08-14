from lab.models import chat_model , small_model , embedding_model

llm = chat_model() # it goes for qwen , temperature 0 in default

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