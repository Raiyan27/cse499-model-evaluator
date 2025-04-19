import os
import time
import traceback
from typing import Dict, Any

from langchain.schema import HumanMessage, AIMessage
from langgraph.graph.state import StateGraph

import config
from vector_store import get_embedding_model, load_or_create_vector_db, get_retriever
from agents import (
    get_clarification_agent,
    get_agentic_rag_app,
    ClarificationState,
    RAGGraphState,
    try_save_graph_image # Optional image saving
)

# --- Global Variables / Initialization ---
# These could be class members if you prefer an object-oriented structure
embeddings_model = None
vector_db = None
retriever = None
clarification_agent: StateGraph | None = None
rag_agent: StateGraph | None = None
resources_initialized = False

def initialize_resources():
    """Initializes all necessary components (models, DB, agents)."""
    global embeddings_model, vector_db, retriever, clarification_agent, rag_agent, resources_initialized
    if resources_initialized:
        print("Resources already initialized.")
        return True

    print("="*50)
    print("Initializing Legal Assistant Resources...")
    print("="*50)

    start_time = time.time()

    # 1. Embeddings
    embeddings_model = get_embedding_model(config.OPENAI_API_KEY)
    if not embeddings_model:
        print("CRITICAL ERROR: Failed to initialize embedding model. Exiting.")
        return False

    # 2. Vector DB & Retriever
    vector_db = load_or_create_vector_db(embeddings_model)
    if not vector_db:
        print("CRITICAL ERROR: Failed to load or create vector database. Exiting.")
        # Potentially clean up partial DB creation? Risky.
        # if os.path.exists(config.PERSIST_DIR):
        #     print(f"Consider manually deleting {config.PERSIST_DIR} if issues persist.")
        return False

    retriever = get_retriever(vector_db)
    if not retriever:
        print("CRITICAL ERROR: Failed to create retriever. Exiting.")
        return False

    # 3. Clarification Agent
    clarification_agent = get_clarification_agent(config.OPENAI_API_KEY)
    if not clarification_agent:
        print("CRITICAL ERROR: Failed to build clarification agent. Exiting.")
        return False
    # Optionally save graph image after successful compilation
    try_save_graph_image(clarification_agent, config.CLARIFICATION_GRAPH_IMAGE_PATH)


    # 4. RAG Agent
    rag_agent = get_agentic_rag_app(retriever, config.OPENAI_API_KEY, config.TAVILY_API_KEY)
    if not rag_agent:
        print("CRITICAL ERROR: Failed to build RAG agent. Exiting.")
        return False
    # Optionally save graph image
    try_save_graph_image(rag_agent, config.RAG_GRAPH_IMAGE_PATH)


    end_time = time.time()
    print("-" * 50)
    print(f"All resources initialized successfully in {end_time - start_time:.2f} seconds.")
    print("-" * 50)
    resources_initialized = True
    return True

def run_legal_assistant_conversation(initial_query: str, max_clarification_attempts: int = config.MAX_QUERY_CLARIFICATION_TURNS) -> Dict[str, Any]:
    """
    Runs the full clarification and RAG process for a given initial query.
    Handles the clarification loop internally.

    Args:
        initial_query: The user's first question.
        max_clarification_attempts: Override for max clarification turns.

    Returns:
        A dictionary containing:
        - 'final_answer': The generated answer string.
        - 'clarified_query': The query used for the RAG agent.
        - 'status': 'Success' or 'Error'.
        - 'error_message': Details if status is 'Error'.
        - 'clarification_log': A list of messages exchanged during clarification (optional).
    """
    if not resources_initialized:
        print("ERROR: Resources not initialized. Call initialize_resources() first.")
        return {"final_answer": "", "clarified_query": initial_query, "status": "Error", "error_message": "Resources not initialized."}

    print(f"\nStarting Conversation for Initial Query: '{initial_query}'")
    print("-" * 30)

    current_history = [HumanMessage(content=initial_query)]
    clarification_log = [{"role": "user", "content": initial_query}]
    clarified_query = None
    turn = 0

    # --- Clarification Loop ---
    while turn < max_clarification_attempts:
        print(f"\nClarification Turn {turn + 1}/{max_clarification_attempts}")
        input_state = ClarificationState(
            initial_query=initial_query,
            conversation_history=current_history,
            max_turns=max_clarification_attempts,
            current_turn=turn, # Pass the current turn number
            clarified_query=None,
            ask_user_question=None,
            reasoning_for_question=None
        )

        try:
            # Stream or invoke the clarification agent for one step
            output_state = clarification_agent.invoke(input_state) # Invoke should run until END for this turn

            if not output_state:
                 raise ValueError("Clarification agent returned None state.")

            # Update turn count based on agent's output (if it increments it)
            # Or manage it externally like here
            turn = output_state.get('current_turn', turn) # Use agent's turn if provided

            question_to_ask = output_state.get("ask_user_question")
            final_clarified_query = output_state.get("clarified_query")

            if final_clarified_query:
                print(f"--- Clarification Complete or Max Turns Reached ---")
                print(f"Final Clarified Query: {final_clarified_query}")
                clarified_query = final_clarified_query
                if question_to_ask: # If max turns hit, there might be a last question generated but unused
                    print(f"(Note: Max turns reached, ignoring last generated question: '{question_to_ask}')")
                    clarification_log.append({"role": "assistant", "content": f"[Max turns reached. Proceeding with best guess.] {question_to_ask}"}) # Log it
                break # Exit clarification loop

            elif question_to_ask:
                print(f"--- Clarification Needed. Agent asks: {question_to_ask} ---")
                clarification_log.append({"role": "assistant", "content": question_to_ask})
                # *** Non-Interactive Simulation ***
                # In a testing scenario, we don't have a real user.
                # We assume the LLM must figure it out or give up within the turns.
                # For DeepEval, we typically want the end-to-end result from the initial query.
                # Let's simulate a generic user response or just let the loop continue.
                # OPTION 1: Simulate a generic response (might lead agent astray)
                # simulated_response = "Please provide more details specific to my situation."
                # print(f"Simulating User Response: {simulated_response}")
                # current_history.append(AIMessage(content=question_to_ask))
                # current_history.append(HumanMessage(content=simulated_response))
                # clarification_log.append({"role": "user", "content": simulated_response})

                # OPTION 2: Let the agent try again with the same history + its own question.
                # This relies on the agent realizing it asked before or the assessment changing.
                # This seems more realistic for testing the agent's internal reasoning.
                current_history.append(AIMessage(content=question_to_ask)) # Add AI question to history for next turn's context

                # Increment turn counter MANUALLY here since we are looping externally
                turn += 1

            else:
                # Should not happen: agent didn't clarify and didn't ask a question
                print("WARNING: Clarification agent ended turn unexpectedly. Proceeding with current history.")
                clarified_query = f"Clarification ended unexpectedly. Using best guess from history: {initial_query}" # Fallback
                break

        except Exception as e:
            print(f"ERROR during clarification turn {turn + 1}: {e}")
            traceback.print_exc()
            return {
                "final_answer": "",
                "clarified_query": initial_query, # Use initial as fallback
                "status": "Error",
                "error_message": f"Error during clarification: {e}",
                "clarification_log": clarification_log
            }

    # --- Post-Clarification ---
    if not clarified_query:
        # If loop finished without setting a query (e.g., hit max turns but assess node didn't synthesize)
        print(f"WARNING: Max clarification turns ({max_clarification_attempts}) reached, but no final query synthesized by agent. Using initial query as fallback.")
        clarified_query = initial_query
        clarification_log.append({"role": "system", "content": f"[Max turns reached, fallback to initial query: {initial_query}]"})

    print("\n--- Moving to RAG Agent ---")
    print(f"Using Query: {clarified_query}")
    print("-" * 30)

    # --- RAG Execution ---
    rag_input_state = RAGGraphState(
        question=clarified_query, # Start with the clarified query (will be rewritten for DB)
        original_clarified_question=clarified_query, # Store the definitive clarified query
        generation="",
        web_search_needed="No", # Initial default
        documents=[]
    )

    try:
        # Invoke the RAG agent
        final_rag_state = rag_agent.invoke(rag_input_state, config={"recursion_limit": 25})

        final_answer = final_rag_state.get("generation", "Error: No final answer generated by RAG agent.")
        print("\n--- RAG Agent Finished ---")
        print(f"Final Answer:\n{final_answer}")

        return {
            "final_answer": final_answer,
            "clarified_query": clarified_query,
            "status": "Success",
            "error_message": None,
            "clarification_log": clarification_log
        }

    except Exception as e:
        print(f"ERROR during RAG agent execution: {e}")
        traceback.print_exc()
        return {
            "final_answer": "",
            "clarified_query": clarified_query, # Report the query we tried to use
            "status": "Error",
            "error_message": f"Error during RAG execution: {e}",
            "clarification_log": clarification_log
        }


# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure resources are ready
    if not initialize_resources():
        exit(1) # Stop if initialization failed

    # Example Usage:
    # Get query from command line argument or use a default
    import sys
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        # test_query = "Tell me about company registration in Bangladesh."
        test_query = "Examine the nuances of the RTI law in promoting governmental accountability and citizen empowerment."
        # test_query = "ভূমি আইন" # Very vague query to test clarification

    print(f"\nRunning assistant for query: '{test_query}'\n")
    result = run_legal_assistant_conversation(test_query)

    print("\n" + "="*50)
    print("Final Result Summary:")
    print(f"Status: {result['status']}")
    if result['error_message']:
        print(f"Error: {result['error_message']}")
    print(f"Clarified Query Used for RAG: {result['clarified_query']}")
    print("\nFinal Answer:")
    print(result['final_answer'])
    print("="*50)

    # Print clarification log (optional)
    # print("\nClarification Log:")
    # for msg in result.get('clarification_log', []):
    #     print(f"  {msg['role']}: {msg['content']}")