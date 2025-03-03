import os
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import getpass
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_mistralai import ChatMistralAI
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, END, START

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")

if "TAVILY_API_KEY" not in os.environ:
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")


# Define graph states
class AgentState(Enum):
    RESEARCH = "research"
    SYNTHESIS = "synthesis"
    ANSWER = "answer"
    COMPLETE = "complete"


# Initialize LLM and search tool
researcher_llm = ChatMistralAI(model_name="mistral-large-latest", temperature=0.3)
synthesizer_llm = ChatMistralAI(model_name="mistral-large-latest", temperature=0.4)
drafter_llm = ChatMistralAI(model_name="mistral-large-latest", temperature=0.7)

search_tool = TavilySearchResults(k=8, include_domains=[], exclude_domains=[])


# Define state schema
class GraphState(Dict):
    query: str
    search_results: Optional[List[Dict[str, Any]]] = None
    research_notes: Optional[str] = None
    sources: Optional[List[Dict[str, str]]] = None
    synthesized_research: Optional[str] = None
    final_answer: Optional[str] = None
    messages: List[Dict[str, Any]] = []
    current_state: AgentState = AgentState.RESEARCH

    def __init__(self, query: str):
        super().__init__()
        self["query"] = query
        self["search_results"] = None
        self["research_notes"] = None
        self["sources"] = []
        self["synthesized_research"] = None
        self["final_answer"] = None
        self["messages"] = []
        self["current_state"] = AgentState.RESEARCH


# Agent prompts
researcher_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are a thorough research agent designed to gather comprehensive information on a given topic.
Your job is to:
1. Break down complex queries into searchable components
2. Plan a research strategy with specific search queries
3. Analyze search results objectively
4. Extract key facts, data, and insights
5. Note contradictions or gaps in information
6. Document all sources meticulously

Be methodical, unbiased, and focused on collecting high-quality information."""
        ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(
            content="""
Research Query: {query}

Previous Search Results: {search_results}

Your task is to:
1. Analyze the search results
2. Identify the most relevant information
3. Note any contradictions or gaps in knowledge
4. Suggest additional specific search queries to fill knowledge gaps
5. Organize findings into structured research notes

Provide your research notes in a detailed, well-structured format, highlighting key facts and insights.
"""
        ),
    ]
)

synthesizer_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are a data synthesis agent designed to organize and connect research findings into a cohesive whole.
Your job is to:
1. Identify patterns and relationships across research notes
2. Reconcile contradictory information
3. Highlight consensus views and notable disagreements
4. Organize information hierarchically by importance and relevance
5. Distinguish between factual information and interpretations
6. Connect related concepts to form a complete picture

Be analytical, thorough, and focused on creating a comprehensive synthesis."""
        ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(
            content="""
Original Query: {query}

Research Notes: {research_notes}

Sources: {sources}

Your task is to synthesize this research into a coherent, well-structured document that:
1. Organizes information logically
2. Highlights key findings and their relationships
3. Identifies consensus views and areas of disagreement
4. Notes limitations and gaps in the current research
5. Prepares the information for drafting a comprehensive answer

Present your synthesis in a clear, well-structured format that will serve as the foundation for drafting the final answer.
"""
        ),
    ]
)

drafter_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are an answer drafting agent designed to create comprehensive, accurate, and well-structured responses based on research.
Your job is to:
1. Transform synthesized research into a clear, engaging answer
2. Maintain accuracy while making complex information accessible
3. Use appropriate structure, headings, and formatting for clarity
4. Include relevant examples, analogies, or explanations
5. Cite sources appropriately throughout
6. Ensure the answer directly addresses the original query

Be precise, thorough, and focused on creating a response that effectively communicates the research findings."""
        ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(
            content="""
Original Query: {query}

Synthesized Research: {synthesized_research}

Sources: {sources}

Your task is to draft a comprehensive final answer that:
1. Directly addresses the original query
2. Presents information clearly and logically
3. Uses appropriate formatting for readability
4. Cites sources appropriately
5. Explains complex concepts in an accessible way
6. Provides a balanced view of the topic

Draft a complete, well-structured answer that effectively communicates the research findings while remaining engaging and accessible.
"""
        ),
    ]
)


# Define research agent function
def run_research(state: GraphState) -> GraphState:
    query = state["query"]

    if not state["search_results"]:
        print(f"Searching for: {query}")
        search_results = search_tool.invoke({"query": query})
        state["search_results"] = search_results

        state["messages"].append(
            {"role": "system", "content": f"Performed initial search for: {query}"}
        )

    researcher_chain = researcher_prompt | researcher_llm | StrOutputParser()

    research_notes = researcher_chain.invoke(
        {
            "query": query,
            "search_results": state["search_results"],
            "messages": state["messages"],
        }
    )

    sources = []
    for result in state["search_results"]:
        sources.append(
            {
                "title": result.get("title", "Unknown Title"),
                "url": result.get("url", "Unknown URL"),
                "published_date": result.get("published_date", "Unknown Date"),
            }
        )

    state["research_notes"] = research_notes
    state["sources"] = sources
    state["messages"].append(
        {"role": "assistant", "content": "Research phase completed."}
    )
    state["current_state"] = AgentState.SYNTHESIS

    return state


# Define synthesis agent function
def run_synthesis(state: GraphState) -> GraphState:
    synthesizer_chain = synthesizer_prompt | synthesizer_llm | StrOutputParser()

    synthesized_research = synthesizer_chain.invoke(
        {
            "query": state["query"],
            "research_notes": state["research_notes"],
            "sources": state["sources"],
            "messages": state["messages"],
        }
    )

    state["synthesized_research"] = synthesized_research
    state["messages"].append(
        {"role": "assistant", "content": "Synthesis phase completed."}
    )
    state["current_state"] = AgentState.ANSWER

    return state


# Define answer drafting agent
def run_answer_drafting(state: GraphState) -> GraphState:
    drafter_chain = drafter_prompt | drafter_llm | StrOutputParser()

    final_answer = drafter_chain.invoke(
        {
            "query": state["query"],
            "synthesized_research": state["synthesized_research"],
            "sources": state["sources"],
            "messages": state["messages"],
        }
    )

    state["final_answer"] = final_answer
    state["messages"].append(
        {"role": "assistant", "content": "Answer drafting phase completed."}
    )
    state["current_state"] = AgentState.COMPLETE

    return state


# Entry point function
def start_workflow(state: GraphState) -> GraphState:
    state["messages"].append(
        {
            "role": "system",
            "content": f"Starting research workflow for: {state['query']}",
        }
    )
    return state


# Router Function
def route_next_step(state: GraphState) -> str:
    current_state = state["current_state"]

    if current_state == AgentState.RESEARCH:
        return "synthesis"
    elif current_state == AgentState.SYNTHESIS:
        return "answer"
    elif current_state == AgentState.ANSWER:
        return "complete"
    else:
        return END


# Construct the workflow graph and compile
workflow = StateGraph(GraphState)
workflow.add_node("start", start_workflow)
workflow.add_node("research", run_research)
workflow.add_node("synthesis", run_synthesis)
workflow.add_node("answer", run_answer_drafting)
workflow.add_edge(START, "start")
workflow.add_edge("start", "research")
workflow.add_conditional_edges("research", route_next_step)
workflow.add_conditional_edges("synthesis", route_next_step)
workflow.add_conditional_edges("answer", route_next_step)
deep_research_system = workflow.compile()


def run_deep_research(query: str) -> Dict[str, Any]:
    print(f"Starting deep research on: {query}")
    print("-" * 50)

    # Initialize the state with the query
    initial_state = GraphState(query)
    result = deep_research_system.invoke(initial_state)

    print("-" * 50)
    print(f"Research completed at {datetime.now()}")

    return {
        "query": result["query"],
        "final_answer": result["final_answer"],
        "sources": result["sources"],
        "workflow_path": [
            msg["content"] for msg in result["messages"] if msg["role"] == "assistant"
        ],
    }


if __name__ == "__main__":
    query = input("Enter your query: ")
    results = run_deep_research(query)

    print("FINAL ANSWER")
    print(results["final_answer"])

    print("SOURCES")
    for i, source in enumerate(results["sources"], 1):
        print(f"{i}. {source['url']}")
