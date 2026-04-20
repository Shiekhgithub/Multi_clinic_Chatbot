"""
agent.py
--------
Creates three LangChain retriever Tools (one per dataset) and
wires them into a ReAct agent using LangChain's create_react_agent.
"""

import os
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import InMemorySaver  
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent






# ──────────────────────────────────────────────
# LLM factory
# ──────────────────────────────────────────────

def get_llm(provider: str = "groq") -> BaseLanguageModel:
    """
    Return an LLM instance.
    provider: "openai" | "groq"
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=0.1,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'groq'.")


# ──────────────────────────────────────────────
# Retriever tools
# ──────────────────────────────────────────────

def build_tools(stores: dict[str, Chroma], k: int = 5) -> list:
    """
    Build one LangChain retriever Tool for each dataset.
    k = number of documents to retrieve per query.
    """
    heart_retriever = stores["heart_disease"].as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    derm_retriever = stores["dermatology"].as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    diabetes_retriever = stores["diabetes"].as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    heart_tool = create_retriever_tool(
        retriever=heart_retriever,
        name="heart_disease_tool",
        description=(
            "Use this tool when the user asks about heart disease, cardiovascular conditions, "
            "chest pain, blood pressure, cholesterol, ECG, or cardiac symptoms. "
            "Input should be a specific question about heart health."
        ),
    )

    derm_tool = create_retriever_tool(
        retriever=derm_retriever,
        name="dermatology_tool",
        description=(
            "Use this tool when the user asks about skin conditions, dermatology, rashes, "
            "psoriasis, dermatitis, lichen planus, pityriasis, or skin disease classification. "
            "Input should be a specific question about skin health."
        ),
    )

    diabetes_tool = create_retriever_tool(
        retriever=diabetes_retriever,
        name="diabetes_tool",
        description=(
            "Use this tool when the user asks about diabetes, blood glucose, insulin, BMI, "
            "Pakistani diabetes data, or related metabolic conditions. "
            "Input should be a specific question about diabetes."
        ),
    )

    return [heart_tool, derm_tool, diabetes_tool]


# ──────────────────────────────────────────────
# ReAct Agent
# ──────────────────────────────────────────────

def build_agent(stores: dict[str, Chroma], provider: str = "groq", k: int = 5):
    """
    Build and return a LangChain ReAct AgentExecutor with memory.
    Uses the standard hwchase17/react-chat prompt from LangChain Hub.
    """
    llm = get_llm(provider)
    tools = build_tools(stores, k=k)

    prompt = """
        You are a highly intelligent Healthcare AI Assistant.

        Your job is to answer user questions related to:
        - Heart Disease
        - Dermatology (skin diseases)
        - Diabetes

        You have access to the following tools:

        1. heart_disease_tool  
        → Use for: heart-related queries (chest pain, ECG, cholesterol, blood pressure, cardiovascular diseases)

        2. dermatology_tool  
        → Use for: skin-related queries (rashes, psoriasis, dermatitis, skin lesions, classification)

        3. diabetes_tool  
        → Use for: diabetes-related queries (blood sugar, insulin, BMI, glucose levels, Pakistani diabetes dataset)

        --------------------------------------------------

        🔍 TOOL USAGE RULES:

        - ALWAYS use a tool if the question is medical and specific.
        - SELECT the MOST RELEVANT tool based on the user's query.

        🚫 OUT-OF-SCOPE HANDLING:

        - If the user's query is NOT related to healthcare, heart disease, dermatology, or diabetes:
        → Politely respond that you are only designed to assist with healthcare-related queries.
        → Do NOT use any tool in this case.
        → Keep the response short, respectful, and helpful.

        Example:
        "I'm here to help with healthcare-related questions such as heart disease, diabetes, and skin conditions. Please ask a relevant question."
        
    """


    agent_executor = create_agent(llm, system_prompt=prompt,checkpointer=InMemorySaver(),tools=tools)

    return agent_executor
