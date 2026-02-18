"""
cli_chat.py
-----------
Simple terminal chatbot for the Healthcare RAG system.

Run:
    python cli_chat.py
"""

import os
from dotenv import load_dotenv
from vector_store import load_all_stores, stores_exist
from agent import build_agent

load_dotenv()

BANNER = """
╔══════════════════════════════════════════════════════╗
║       Healthcare RAG Chatbot  (CLI mode)             ║
║  Datasets: Heart Disease | Dermatology | Diabetes    ║
║  Type 'quit' or 'exit' to stop                       ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    provider    = os.getenv("LLM_PROVIDER", "groq")

    if not stores_exist(persist_dir):
        print("❌  Vector stores not found. Please run:  python ingest.py")
        return

    print(BANNER)
    print(f"[Agent] Loading vector stores …")
    stores = load_all_stores(persist_dir)

    print(f"[Agent] Initialising ReAct agent (provider={provider}) …\n")
    agent_executor = build_agent(stores, provider=provider)

    print("Ready! Ask anything about heart disease, skin conditions, or diabetes.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        try:
            result = agent_executor.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                {"configurable": {"thread_id": "1"}}
                )
            
            answer = result
            print(f"\nAssistant: {answer}\n")
            print("-" * 60)
        except Exception as e:
            print(f"[Error] {e}\n")


if __name__ == "__main__":
    main()
