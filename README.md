# Deep Research using Agentic AI
This project helps in deep research by analyzing the web and other resources to generate insights. This system uses LangChain, LangGraph, and Tavily for accurate and structured results. It automates the process of retrieving, synthesizing, and generating well-structured answers to complex queries. This system is optimized for autonomous web research, information synthesis, and report generation with minimal human intervention.

## Features
1. Multi-Agent architecture: The system has three AI agents that focus on retrieving information from the web, synthesizing the data, and generating structured answers.
2. Automated research: Uses Tavily to conduct real-time web searches.
3. Uses Mistral AI for innovative response generation.
4. State Management: Uses LangGraph for structured execution.

## System workflow
<p>
<img src="https://github.com/user-attachments/assets/c3222cac-bed9-4958-98a8-339c955be33e" alt="Model Architecture" width="400">
</p>

A user input is taken as a user query. This is then given to the system, and then the system produces a final answer.

## Models
As of now, three agents are working. The LLM used is from Mistral AI. Each agent has a specific task:
### Researcher:
Searches and extracts key insights, facts, and contradictions from web sources in real-time with the help of Tavily. It summarizes findings, documents sources for reference, and 
prepares structured research notes for the next step.
### Synthesizer:
Analyze, organize, and structure the information the research agent obtains into a logical structure. It looks for patterns, relationships, and contradictions in the research to provide a more factual answer.
### Drafter: 
Converts the research results into well-structured, readable, engaging, and easy-to-understand answers. Use formatting methods like headings, bullet points, and citations if needed. Also, tries to provide examples and analogies for better explanations.

## API Keys
This project uses API keys from:
* [Mistral AI](https://mistral.ai/): LLM for the agents
* [Tavily](https://tavily.com/): For real-time web searching

## Future Improvements
1. Provide a robust user interface so for better interaction.
2. Provide long-term memory for better connection with the user.
3. Integrate Retrieval-Augmented Generation (RAG) to improve responses further.
4. Provide the ability to obtain and show relevant diagrams or images along with search results.
