import uvicorn
import os
import re

from fastapi import FastAPI, HTTPException
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel


app = FastAPI(title="LangChain GitHub Error Resolver", version="1.0")


class GithubSearchRequest(BaseModel):
    search_term: str


def get_llm():
    """
    Creates a LangChain agent with DuckDuckGo search as a tool.
    The agent will use GPT to reason and search GitHub/StackOverflow results.
    """
    
    llm = ChatOpenAI(model="openai/gpt-4.1-nano", 
                   temperature=0.1,
                   max_tokens=5000,
                   timeout=30)
    return llm

llm = get_llm()


def setup_tool():
    github = GitHubAPIWrapper()
    toolkit = GitHubToolkit.from_github_api_wrapper(github)
    tools = toolkit.get_tools()

    for tool in tools:
        tool.name = re.sub(r'[^a-zA-Z0-9_.-]', '_', tool.name)
    
    return tools

tools = setup_tool()

def setup_agent():

    agent = create_agent(
       llm,
       tools=tools,
       checkpointer=InMemorySaver()
    )
    return agent

agent = setup_agent()

def use_agent(user_query):
    SYSTEM_PROMPT = """You are an intelligent agent designed to interact with GitHub repositories.
        Your job is to understand user questions and generate the correct sequence of GitHub API queries 
        (using the GitHubAPIWrapper or similar tools) to retrieve the required information from GitHub.
    """
    system_prompt = SystemMessage(content=SYSTEM_PROMPT)
    human_query = HumanMessage(content=user_query)
    config = {"configurable": {"thread_id": "1"}}

    messages = [system_prompt, human_query]
    for event in agent.stream( {"messages": messages},
             stream_mode="values",
              config = config):
         # Capture the last AI message if present
        if "messages" in event and event["messages"]:
            msg = event["messages"][-1]
            if msg.type == "ai":
                last_ai_message = msg.content

    # Return only the last AI response
    return last_ai_message



@app.post("/search_github/data")
async def search_error(request: GithubSearchRequest):
    """
    Takes an error log or traceback and uses LangChain to find GitHub discussions or fixes.
    """
    try:
        query = request.search_term
        response = use_agent(query)
        return {
            "query": query,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================
# 5. Run server (local)
# =============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
