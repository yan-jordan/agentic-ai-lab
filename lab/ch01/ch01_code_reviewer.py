from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder , FewShotChatMessagePromptTemplate
from lab.models import chat_model
from lab.response_token_calculator import token_calc

"""
PROMPT TEMPLATE
"""
# messages
messages = [
    ("system", "you are a professional {role} , answer in {language} language , with {tone} tone.Never use emoji on your answer."),
    ("human", "{question}")
]
# Chat prompt template
prompt = ChatPromptTemplate.from_messages(messages)

# calling our model LLM
llm = chat_model()

# creating our LGEL --> langchain expression language
chain = prompt | llm

# invoke the prompt template and get back the answer
response = chain.invoke({
    "role" : "senior python developer",
    "language" : "english",
    "question" : "explain me functions in python very simply in few sentences.",
    "tone" : "funny"
})

# checking out the results
print(response.content)
print(token_calc(response))
print("--- --- --- --- --- --- --- ---")

"""
PLACE HOLDER
"""

# Placeholder which is actually the memory of our agent during the tasks
questions = [
    "my name is Pouyan",
    "what is my name"
]

history = []

prompt = ChatPromptTemplate.from_messages([
    ("system" , "You are my helpful , smart and concise personal assistant."),
    MessagesPlaceholder("history"),
    ("human" , "{question}")
])

chain = prompt | llm

for question in questions:
    answer = chain.invoke(
        {
            "history" : history,
            "question" : question
        }
    )

    print("Q: " , question)
    print("A: " , answer.content)

    history.append(
        {"role" : "user" , "content" : question}
    )
    history.append(
        {"role" : "assistant" , "content" : answer.content}
    )

# checking results
print(history)

"""
Few shot prompting
"""

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human" , "{input}"),
        ("ai" , "{output}")
    ]
)

print("examples_prompt: \n" , example_prompt)

few_shots = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples = [
        {
            "input" : "I hate this city" , "output" : "negative"
        },
        {
            "input" : "WOW ! This food is amazing!" , "output" : "positive"
        },
        {
            "input" : "I dont know , i guess it was not bad" , "output" : "neutrual"
        }
    ]
)

print("few_shots : \n" , few_shots)



few_shot_prompt = ChatPromptTemplate([
    ("system" , "You are a careful and professional semantic classifier which can label sentences with positive , negative , neutral labels! Here see some examples:"),
    few_shots,
    ("human" , "{input}")
])

print(f"few shot prompt : \n{few_shot_prompt}")

chain = few_shot_prompt | llm

answer = chain.batch([
    {
        "input" : "I just arrived in this country and i have a strange feeling which do not know like here or not"
    },
    {
        "input" : "that movie was bull of shit"
    },
    {
        "input" : "fantastic car"
    },
])

print("answer : " , answer)

"""
EXERCISE 2
"""

example_prompt = ChatPromptTemplate([
    ("human"  ,  "{input}"),
    ("ai" , "{output}")
])

examples = [
    {
        "input": """def get_duplicates(items: list[int]) -> list[int]:
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates""",
        "output": """Performance review:

1. O(n^2) nested loop, plus `items[i] not in duplicates` makes it closer to O(n^3) in the worst case. A hash-based count is O(n).
2. `list.append` in a hot loop with a membership check is the main cost driver; a `set` gives O(1) lookups.

Suggested rewrite:

    from collections import Counter

    def get_duplicates(items: list[int]) -> list[int]:
        return [item for item, count in Counter(items).items() if count > 1]

Complexity: O(n^2)/O(n^3) -> O(n) time, O(n) space. On a 10k-element list this is roughly three orders of magnitude faster.""",
    },
    {
        "input": """def build_report(rows: list[dict]) -> str:
    report = ""
    for row in rows:
        report = report + row["name"] + ": " + str(row["value"]) + "\\n"
    return report""",
        "output": """Performance review:

1. Repeated `str` concatenation allocates a new string every iteration, making this O(n^2) in total bytes copied. `str.join` allocates once.
2. Manual `+` concatenation with `str()` calls is also slower than an f-string.

Suggested rewrite:

    def build_report(rows: list[dict]) -> str:
        return "".join(f"{row['name']}: {row['value']}\\n" for row in rows)

Complexity: O(n^2) -> O(n) time. The generator also avoids materialising an intermediate list.""",
    },
    {
        "input": """def filter_active_users(users: list[dict], banned_ids: list[int]) -> list[dict]:
    result = []
    for user in users:
        if user["active"] == True and user["id"] not in banned_ids:
            result.append(user)
    return result""",
        "output": """Performance review:

1. `user["id"] not in banned_ids` is an O(m) list scan executed for every user, giving O(n*m) overall. Converting `banned_ids` to a `set` once makes each lookup O(1).
2. `== True` adds a needless comparison; use the truth value directly.
3. The append loop can be a list comprehension, which avoids repeated attribute lookup on `result.append`.

Suggested rewrite:

    def filter_active_users(users: list[dict], banned_ids: list[int]) -> list[dict]:
        banned = set(banned_ids)
        return [user for user in users if user["active"] and user["id"] not in banned]

Complexity: O(n*m) -> O(n + m) time, O(m) extra space.""",
    },
]

few_shots = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples = examples
)

few_shot_prompt = ChatPromptTemplate(
    [
        ("system" , "You are a professional and experienced senior {programming_language} developer and you are going to consider codes with focus on {focus}. Here are some examples for you :"),
        few_shots,
        ("human" , "{code}"),
    ]
)

chain = few_shot_prompt | llm

result = chain.invoke(
    {
        "programming_language" : "python",
        "focus" : "performance",
        "code" : """def calculate_min_max_sorted_list(sorted_list : list[int]) -> tuple: 
            min = 0
            max = 0
            for number in sorted_lists:
                if number <= min:
                    min = number
                if number >= max:
                    max = number
            return ( min , max )
        """
    }
)

print("result EXERCISE 2 : " , result.content)