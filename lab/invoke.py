from lab.models import chat_model
from rich.console import Console

console = Console()
llm = chat_model()

def invoke(string: str | None = None , messages: list | None = None):
    with console.status("[bold green]Thinking...", spinner="dots"):
        response = llm.invoke(string) if string != None else llm.invoke(messages)
        return response

def batch(messages : list):
    with console.status("[bold green]Thinking..." , spinner="dots"):
        response = llm.batch(messages)
        return response