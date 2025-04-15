
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import tools_condition,ToolNode
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


#tools

Tavily_Search_Tool = TavilySearchResults()

wikipedia_Tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=500))

arxiv_Wrapper = ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=500)

arxiv_Tool = ArxivQueryRun(api_wrapper=arxiv_Wrapper)

tools = [Tavily_Search_Tool,wikipedia_Tool,arxiv_Tool]

#LLM

llm = ChatGroq(model="Gemma2-9b-it")

llm_with_tools = llm.bind_tools(tools=tools)

#response = llm_with_tools.invoke("attention is all you need")

#print(response)
#if we execute this we notice that as as our llm is binded with tools it knows when to call tool and which tool but it can't run that tool.
#let's define our workflow to execute the tools also

#stateschema
#AnyMessage means human message or AI message
#Annotated is for labelling
#add_messages is reducers in langgraph. If we not use this reducer in our application then new messages will overwrite the previous messages.
#But here we want to append the new messages to the existing messages in a list
 
class state(TypedDict):
    messages : Annotated[list[AnyMessage],add_messages]

#now we have to define our node 

def tool_calling_llm(state):
    return {"messages":llm_with_tools.invoke(state["messages"])}


workflow = StateGraph(state)

workflow.add_node("AI Assistant",tool_calling_llm)
workflow.add_node("tools",ToolNode(tools))
workflow.add_edge(START,"AI Assistant")
workflow.add_conditional_edges("AI Assistant",
                               #If the latest message from assistant is a tool call -> then tools_condition route to tools
                               #If the latest message form the assistant is not a tool call -> then tools_condition routes to end,
                               tools_condition
                               )
workflow.add_edge("tools","AI Assistant")
workflow.add_edge("AI Assistant",END)

graph = workflow.compile()

messages = graph.invoke({"messages":"what is pascal's principle"})

for m in messages['messages']:
    m.pretty_print()