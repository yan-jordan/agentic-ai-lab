from lab.invoke import invoke
from lab.models import chat_model
from langchain.messages import AIMessage , HumanMessage , SystemMessage
from lab.invoke import invoke , batch
from lab.response_token_calculator import token_calc
import time

llm = chat_model()

# Test prompt
test_prompt = HumanMessage("How should i buy or steal a ship for my crew?!")

# Different AISystem roles
role_pirate = SystemMessage("You are a dangerous pirate.Always answer in two sentences.")
role_lawyer = SystemMessage("You are a professional lawyer.Always answer in two sentences.")
role_compiler_error_message = SystemMessage("You are a compiler error message and you should answer like that.Always answer in two sentences.")

# list of different messages with same human prompt but different system roles
messages = [
    [
        test_prompt , role_pirate
    ],
    [
        test_prompt , role_lawyer
    ],
    [
        test_prompt , role_compiler_error_message
    ]
]

# Other type of prompting for messages with Tuples and Dicts
other_messages = [
    # Message 1
    {
        "role" : "system" , "content" : "you are a history teacher"
    },
    # Message 2
    {
        "role" : "user" , "content" : "who is Trump?"
    },
    (
        "ai" , "the capitcal of iran is tehran."
    )
    # Message n ....
]

# Testing few prompts with a for loop
responses = []
start = time.perf_counter()
for message in messages:
    responses.append(invoke(message))
for_loop_time = time.perf_counter() - start
print(f"{for_loop_time:.3f}s")

b = "--- " * 11
for response in responses:
    print(response.content)
    print(token_calc(response))
    print(b)

# Testing few prompts with batch method
start = time.perf_counter()
response_batch = batch(messages)
batch_time_long = time.perf_counter() - start
print(response_batch)

print(b)
print(batch_time_long)
print(b)
print(for_loop_time)
