import functools
import traceback
from typing import List, Optional, TypedDict as TypingTypedDict
from typing_extensions import TypedDict
from operator import itemgetter
from enum import Enum

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document, HumanMessage, AIMessage, BaseRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph

import config # Use constants and API keys from config.py


# ==============================================================================
# --- Clarification Agent Components ---
# ==============================================================================

# --- State for Clarification Graph ---
class ClarificationState(TypedDict):
    initial_query: str
    conversation_history: List # List of HumanMessage/AIMessage
    max_turns: int
    current_turn: int
    clarified_query: Optional[str]
    ask_user_question: Optional[str]
    reasoning_for_question: Optional[str]

# --- Pydantic Model for Clarity Assessment ---
class ClarityStatus(str, Enum):
    CLEAR = "CLEAR"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    MAX_TURNS_REACHED = "MAX_TURNS_REACHED"

class ConversationalClarityAssessment(BaseModel):
    """Assess if the conversation provides enough detail for a Bangladeshi Legal Assistant."""
    status: ClarityStatus = Field(description="Enum: 'CLEAR' if sufficient details are present, 'NEEDS_CLARIFICATION' if more info needed, 'MAX_TURNS_REACHED' if stuck.")
    reasoning: str = Field(description="Brief reasoning for the status. If NEEDS_CLARIFICATION, explain what's missing. If CLEAR, confirm understanding. If MAX_TURNS_REACHED, explain why stuck.")
    synthesized_query_if_clear: Optional[str] = Field(description="If status is CLEAR, provide a concise, synthesized query representing the user's final need.", default=None)

# --- Initialize Clarification LLMs ---
def get_clarification_llms(api_key: str) -> tuple[ChatOpenAI | None, ChatOpenAI | None]:
    """Initializes LLMs needed for the clarification agent."""
    try:
        print("Initializing Clarification Agent LLMs...")
        assessment_llm = ChatOpenAI(
            model=config.CLARIFICATION_ASSESSMENT_MODEL,
            temperature=0,
            openai_api_key=api_key
        ).with_structured_output(ConversationalClarityAssessment)

        question_gen_llm = ChatOpenAI(
            model=config.CLARIFICATION_QUESTION_MODEL,
            temperature=0.3, # Allow slight creativity for questions
            openai_api_key=api_key
        )
        print("Clarification Agent LLMs Initialized.")
        return assessment_llm, question_gen_llm
    except Exception as e:
        print(f"ERROR: Failed to initialize Clarification LLMs: {e}")
        traceback.print_exc()
        return None, None

# --- Clarification Graph Nodes ---
def assess_clarity_node(state: ClarificationState, assessment_llm: ChatOpenAI):
    """Assesses if the current conversation history provides enough clarity."""
    print("--- CLARIFICATION NODE: Assess Clarity ---")
    history = state['conversation_history']
    current_turn = state['current_turn']
    max_turns = state['max_turns']

    if current_turn >= max_turns:
        print(f"--- Max clarification turns ({max_turns}) reached. ---")
        # Synthesize based on what we have
        simplified_history = "\n".join([f"{type(m).__name__}: {m.content}" for m in history])
        final_query = f"Could not fully clarify after {max_turns} attempts. Best guess based on conversation: {simplified_history}"
        if len(final_query) > 500: # Avoid overly long generated queries
             final_query = f"Could not fully clarify after {max_turns} attempts. Using initial query: {state['initial_query']}"

        return {
            "ask_user_question": None,
            "clarified_query": final_query,
            "reasoning_for_question": f"Max turns ({max_turns}) reached.",
            # Ensure current_turn is passed through or updated if needed downstream
            "current_turn": current_turn
        }

    ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are evaluating an ongoing conversation with a user asking about Bangladeshi Law.
            The user started with an initial query, and you may have asked clarifying questions.
            Your goal is to determine if the *entire conversation history* now provides enough specific detail (laws, facts, context, desired outcome related to Bangladesh) to be answered effectively by a legal RAG system.

            Analyze the conversation provided in the 'history'.

            Output your assessment using the 'ConversationalClarityAssessment' tool.
            - If CLEAR: Synthesize the user's core need into a concise 'synthesized_query_if_clear'.
            - If NEEDS_CLARIFICATION: Explain *specifically* what information is still missing in the 'reasoning'.
            - If MAX_TURNS_REACHED (use this only if explicitly told max turns reached by the system state): Explain why you couldn't get clarity.

            Initial User Query: {initial_query}
            Conversation History:
            """),
        MessagesPlaceholder(variable_name="history"),
        ("system", "Based *only* on the conversation above, assess the clarity and determine the next step.")
    ])

    assessment_chain = ASSESSMENT_PROMPT | assessment_llm
    try:
        assessment_result: ConversationalClarityAssessment = assessment_chain.invoke({
            "initial_query": state['initial_query'],
            "history": history
        })
        print(f"Clarity Assessment: Status={assessment_result.status}, Reason='{assessment_result.reasoning}'")
    except Exception as e:
        print(f"--- ERROR during assessment LLM call: {e} ---")
        # Fallback: Ask a generic question
        return {
            "ask_user_question": "Sorry, I encountered an error trying to understand our conversation. Could you please restate your query clearly?",
            "clarified_query": None,
            "reasoning_for_question": "Error during assessment.",
            "current_turn": current_turn # Pass turn info
        }

    if assessment_result.status == ClarityStatus.CLEAR:
        return {
            "clarified_query": assessment_result.synthesized_query_if_clear,
            "ask_user_question": None,
            "reasoning_for_question": assessment_result.reasoning, # Keep reasoning for logs
            "current_turn": current_turn
        }
    elif assessment_result.status == ClarityStatus.NEEDS_CLARIFICATION:
        return {
            "reasoning_for_question": assessment_result.reasoning,
            "ask_user_question": None, # Will be generated next
            "clarified_query": None,
            "current_turn": current_turn
        }
    else: # MAX_TURNS_REACHED handled by LLM or other status
        # Synthesize based on reasoning if possible
        final_query = f"Could not fully clarify. Reason: {assessment_result.reasoning}. Using initial query: {state['initial_query']}"
        return {
            "ask_user_question": None,
            "clarified_query": final_query,
            "reasoning_for_question": assessment_result.reasoning,
            "current_turn": current_turn
        }

def generate_question_node(state: ClarificationState, question_gen_llm: ChatOpenAI):
    """Generates the next clarifying question based on the assessment."""
    print("--- CLARIFICATION NODE: Generate Question ---")
    reasoning = state.get("reasoning_for_question")
    history = state['conversation_history']
    current_turn = state.get("current_turn", 0) # Get current turn

    if not reasoning:
         print("WARNING: Reasoning for question missing in state for generate_question_node.")
         # Ask a generic question as fallback
         return {
             "ask_user_question": "Sorry, I got a bit lost. Could you please provide more details about what you'd like to know?",
             "current_turn": current_turn + 1 # Increment turn
         }

    QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are part of a conversational assistant helping a user clarify their legal question about Bangladesh.
        A previous analysis determined the conversation is not yet clear enough. The reason given was:
        '{reasoning}'

        Based on this reason and the conversation history so far, generate a *single, concise, specific follow-up question* to ask the user.
        Focus *only* on eliciting the missing information identified in the reasoning. Do NOT try to answer the original query. Keep the question natural.

        Conversation History:
        """),
        MessagesPlaceholder(variable_name="history"),
        ("system", "Generate the next clarifying question based on the history and the reasoning provided above.")
    ])

    question_chain = QUESTION_GEN_PROMPT | question_gen_llm | StrOutputParser()
    try:
        next_question = question_chain.invoke({
            "reasoning": reasoning,
            "history": history
        })
        print(f"Generated Question: {next_question}")
    except Exception as e:
        print(f"--- ERROR during question generation LLM call: {e} ---")
        next_question = "Sorry, I encountered an error trying to formulate a follow-up question. Could you perhaps rephrase or add more detail?"

    # Important: Increment the turn counter here AFTER successfully generating a question
    return {
        "ask_user_question": next_question,
        "current_turn": current_turn + 1
        # Other state fields like clarified_query remain as they were
    }

# --- Build Clarification Graph ---
def get_clarification_agent(openai_api_key: str):
    """Builds and compiles the clarification agent graph."""
    assessment_llm, question_gen_llm = get_clarification_llms(openai_api_key)
    if not assessment_llm or not question_gen_llm:
        print("ERROR: Cannot build clarification agent: LLMs not available.")
        return None

    print("Building Clarification Agent Graph...")
    clarif_graph = StateGraph(ClarificationState)

    # Bind LLMs to nodes
    bound_assess_clarity = functools.partial(assess_clarity_node, assessment_llm=assessment_llm)
    bound_generate_question = functools.partial(generate_question_node, question_gen_llm=question_gen_llm)

    clarif_graph.add_node("assess_clarity", bound_assess_clarity)
    clarif_graph.add_node("generate_question", bound_generate_question)

    # NOTE: The 'ask_user' node is implicit in this non-UI version.
    # The graph determines *if* a question should be asked. The calling code (`main.py`)
    # will handle the loop and decide whether to proceed based on the state.
    # We don't add an explicit 'ask_user' node that pauses the graph here.

    clarif_graph.set_entry_point("assess_clarity")

    # Conditional Edges: Decide after assessment
    # Based on the state *after* assess_clarity runs
    def decide_next_step(state: ClarificationState):
        if state.get("clarified_query"): # Clarity reached or max turns hit
            return END
        elif state.get("reasoning_for_question"): # Needs clarification, reasoning provided
             return "generate_question"
        else:
            # Should not happen with proper node logic, but default to END
            print("WARNING: Ambiguous state after assessment, ending clarification.")
            return END

    clarif_graph.add_conditional_edges(
        "assess_clarity",
        decide_next_step,
        {
            "generate_question": "generate_question",
            END: END
        }
    )

    # Edge from question generation back to assessment for the next turn
    # NO - The calling loop will handle this. generate_question doesn't loop back automatically.
    # It just updates the state with ask_user_question. The graph should end here for this step.
    # We modify the decide_next_step logic slightly. If generate_question is needed, it runs,
    # and then the graph *should* end for that invocation, returning the state with the question.

    # Let's rethink the flow for non-interactive testing.
    # If the goal is *one* run to get EITHER a clarified query OR the first question, the graph is simpler.
    # If the goal is to *simulate* the multi-turn within the graph (less common for testing), it's more complex.

    # ASSUMPTION: The test framework provides ONE initial query. The clarification agent runs
    # internally. If it needs clarification, the LLM *simulates* getting it (or decides it can't).
    # This means the graph should run until `clarified_query` is set or max turns are hit.

    # REVISED Clarification Graph Logic for non-interactive run:
    clarif_graph = StateGraph(ClarificationState)
    bound_assess_clarity = functools.partial(assess_clarity_node, assessment_llm=assessment_llm)
    # Generate question is NOT directly used if we expect the LLM to handle clarification internally.
    # Instead, the assess_clarity LLM needs to be good enough to synthesize or give up.

    # Let's stick to the original multi-step design, but `main.py` will manage the loop.
    # The graph performs one step: Assess -> maybe Generate Question -> End Step.
    clarif_graph = StateGraph(ClarificationState) # Re-init
    bound_assess_clarity = functools.partial(assess_clarity_node, assessment_llm=assessment_llm)
    bound_generate_question = functools.partial(generate_question_node, question_gen_llm=question_gen_llm)

    clarif_graph.add_node("assess_clarity", bound_assess_clarity)
    clarif_graph.add_node("generate_question", bound_generate_question)

    clarif_graph.set_entry_point("assess_clarity")

    def decide_after_assessment(state: ClarificationState):
        # If assess_clarity determined CLEAR or MAX_TURNS, it sets clarified_query
        if state.get("clarified_query"):
            return END
        # If assess_clarity determined NEEDS_CLARIFICATION, it sets reasoning
        elif state.get("reasoning_for_question"):
             # Check if we came from generate_question node already (which sets ask_user_question)
             # This check is tricky. Let's assume assessment always decides.
             return "generate_question"
        else:
            print("WARNING: Unexpected state after assess_clarity. Ending.")
            return END # Fallback

    clarif_graph.add_conditional_edges(
        "assess_clarity",
        decide_after_assessment,
        {
            "generate_question": "generate_question",
            END: END
        }
    )

    # After generating a question, the graph run should END for this turn.
    # The output state will contain 'ask_user_question'.
    clarif_graph.add_edge("generate_question", END)


    # Compile the graph
    try:
        print("Compiling Clarification Agent Graph...")
        clarification_agent = clarif_graph.compile()
        print("Clarification agent graph compiled successfully.")
        # Optionally save graph image (requires graphviz/mermaid)
        # try_save_graph_image(clarification_agent, config.CLARIFICATION_GRAPH_IMAGE_PATH)
        return clarification_agent
    except Exception as e:
        print(f"ERROR: Failed to compile clarification agent graph: {e}")
        traceback.print_exc()
        return None

# ==============================================================================
# --- Agentic RAG Components ---
# ==============================================================================

# --- Pydantic model for grader ---
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# --- Initialize RAG LLMs and Tools ---
def get_rag_components(openai_api_key: str, tavily_api_key: str):
    try:
        print("Initializing RAG Agent LLMs and Tools...")
        llm = ChatOpenAI(model=config.RAG_LLM_MODEL, temperature=0, openai_api_key=openai_api_key)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        rewriter_llm = ChatOpenAI(model=config.REWRITER_LLM_MODEL, temperature=0, openai_api_key=openai_api_key)
        tv_search = TavilySearchResults(
            max_results=config.TAVILY_MAX_RESULTS,
            search_depth=config.TAVILY_SEARCH_DEPTH,
            max_tokens=config.TAVILY_MAX_TOKENS,
            tavily_api_key=tavily_api_key
        )
        print("RAG Agent LLMs and Tools Initialized.")
        return llm, structured_llm_grader, rewriter_llm, tv_search
    except Exception as e:
        print(f"ERROR: Failed to initialize RAG Agent LLMs or Tools: {e}")
        traceback.print_exc()
        return None, None, None, None

# --- RAG Prompts ---
# Grader Prompt
SYS_PROMPT_GRADER = """You are an expert grader assessing relevance of a retrieved document to a user question about Bangladeshi Law.
                 Follow these instructions for grading:
                   - If the document contains keyword(s) or semantic meaning directly related to the user's core question, grade it as relevant.
                   - Be strict. General mentions are not enough unless they directly address the query's specifics.
                   - Your grade must be strictly 'yes' or 'no' (lowercase)."""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_PROMPT_GRADER),
    ("human", "User question:\n{question}\n\nRetrieved document:\n```\n{document}\n```\n\nIs this document relevant to the question? Answer 'yes' or 'no'."),
])

# RAG Prompt (with source handling)
SYS_PROMPT_RAG = """You are an assistant for question-answering tasks related to Bangladeshi Law.
Use the following pieces of retrieved context to answer the question.
If no context is provided, or if the context does not contain the answer, state that you could not find the information in the provided sources.
Do not make up information or answer outside the provided context.
Provide a detailed and specific answer based *only* on the context provided. Be concise and clear.

**IMPORTANT CITATION INSTRUCTIONS:** For each piece of information you use from the context, cite its origin clearly at the end of the relevant sentence or paragraph.
- If the context header indicates a filename (e.g., 'Context from: my_document.pdf'), cite it like this: `(Source: my_document.pdf)`.
- If the context header indicates a web search URL (e.g., 'Context from: Web Search (URL: https://example.com/page)'), cite the URL like this: `(Source: https://example.com/page)`.
- If the URL is 'N/A' or missing, cite it as `(Source: Web Search)`.
- If the source is 'Unknown Source' or missing, cite it as `(Source: Unknown)`.
Combine information logically, but ensure all distinct pieces of information are traceable to their source. Multiple citations in one sentence are acceptable if needed, like `Sentence content. (Source: doc1.pdf) (Source: https://web.url)`.

Question:
{question}

Context:
{context}

Answer:"""
rag_prompt_template = ChatPromptTemplate.from_template(SYS_PROMPT_RAG)

# Web Search Re-writer Prompt
SYS_WEB_SEARCH_PROMPT = """You are an expert question re-writer. Convert the user's legal question about Bangladesh into an effective, keyword-focused query for a web search engine (like Google or Tavily).
Remove conversational fluff. Extract key legal terms, entities, locations, and actions. Output only the optimized query string.
Example Input: "What are the specific rules for registering a private limited company in Dhaka, Bangladesh, including required documents and fees?"
Example Output: private limited company registration Dhaka Bangladesh required documents fees rules
Example Input: "Tell me about bail procedures for narcotics cases under the NDPS Act in Chittagong."
Example Output: bail procedure narcotics NDPS Act Chittagong Bangladesh"""
re_write_prompt_web = ChatPromptTemplate.from_messages([
    ("system", SYS_WEB_SEARCH_PROMPT),
    ("human", "Initial question:\n{question}\n\nOptimized web search query:"),
])

# Initial Query Re-writer Prompt (for Vector DB)
SYS_INITIAL_QUERY_PROMPT = """You are a query optimizer for a vector database containing Bangladeshi legal documents.
Analyze the input query, which should be a relatively clear legal question about Bangladesh.
Refine this query for optimal semantic retrieval from the vector DB. Focus on extracting key legal terms, specific laws/acts, procedures, and relevant entities (like 'court', 'company', 'land'). Keep it concise. Output only the refined query string.

Example Input: "Procedure for child custody application under Muslim Family Laws Ordinance in Dhaka Family Court"
Example Output: child custody application Muslim Family Laws Ordinance Dhaka Family Court procedure

Example Input: "Requirements for bail in non-bailable offense under NDPS Act Bangladesh"
Example Output: bail requirements non-bailable offense NDPS Act Bangladesh

Input Query:
{question}

Refined query for vector DB:"""
re_write_initial_prompt = ChatPromptTemplate.from_messages([
    ("system", SYS_INITIAL_QUERY_PROMPT),
    ("human", "Refine the query based on the input provided above."),
])


# --- RAG Chains ---
def setup_rag_chains(llm, structured_llm_grader, rewriter_llm):
    """Sets up the various chains required for the RAG agent."""
    doc_grader = (grade_prompt | structured_llm_grader)
    question_rewriter_web = (re_write_prompt_web | rewriter_llm | StrOutputParser())
    initial_query_rewriter = (re_write_initial_prompt | rewriter_llm | StrOutputParser())

    def format_docs_with_sources(docs: List[Document]) -> str:
        """Formats documents for the RAG prompt, including specific sources."""
        if not docs:
            return "No relevant context found."
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            content = doc.page_content.strip()
            if not content: continue # Skip empty documents

            header = f"--- Context from: {source}"
            if source == "Web Search":
                url = doc.metadata.get('url', 'N/A')
                header = f"--- Context from: Web Search (URL: {url})"
            elif source == 'Unknown Source':
                 header = "--- Context from: Unknown Source"

            formatted.append(f"{header} ---\n{content}")
        return "\n\n".join(formatted) if formatted else "No relevant context found."

    qa_rag_chain = (
        {
            "context": itemgetter('context') | RunnableLambda(format_docs_with_sources),
            "question": itemgetter('question')
        }
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )
    return doc_grader, question_rewriter_web, initial_query_rewriter, qa_rag_chain


# --- RAG Graph State ---
class RAGGraphState(TypedDict):
    question: str           # The query currently being processed (can be rewritten)
    original_clarified_question: str # The query as it came from clarification
    generation: str         # Final answer
    web_search_needed: str  # "Yes" or "No"
    documents: List[Document] # Relevant documents found

# --- RAG Graph Nodes ---
# Node functions need access to retriever, chains, tools. We'll pass them via functools.partial.

def rewrite_clarified_query_for_db(state: RAGGraphState, initial_query_rewriter_chain):
    """Rewrites the CLARIFIED query for potentially better DB retrieval."""
    print("--- RAG NODE: Initial Query Rewriter (DB) ---")
    clarified_input = state["original_clarified_question"] # Use the original clarified Q as base
    print(f"Clarified Query Input for DB Rewriting: {clarified_input}")
    try:
        # Apply rewriting
        better_question_for_db = initial_query_rewriter_chain.invoke({"question": clarified_input})
        # Fallback if rewriter returns empty string
        if not better_question_for_db:
            print("WARN: DB query rewriter returned empty string, using original clarified query.")
            better_question_for_db = clarified_input
    except Exception as e:
        print(f"ERROR during DB query rewriting: {e}. Using original clarified query.")
        better_question_for_db = clarified_input

    print(f"Rewritten for DB: {better_question_for_db}")
    # Update 'question' for retrieval, keep 'original_clarified_question' intact
    return {"question": better_question_for_db}

def retrieve_rag(state: RAGGraphState, retriever: BaseRetriever):
    """Retrieves documents from the vector database."""
    print("--- RAG NODE: Retrieve from Vector DB ---")
    question_for_db = state["question"] # Use the potentially DB-rewritten question
    print(f"Retrieving for: {question_for_db}")
    try:
        # Use the passed retriever instance
        documents = retriever.invoke(question_for_db)
        print(f"Retrieved {len(documents)} documents.")
        # Filter out docs with very low scores if thresholding didn't catch them all? (optional)
        return {"documents": documents}
    except Exception as e:
        print(f"---ERROR during RAG retrieval: {e}---")
        # Return empty list but allow process to continue (may trigger web search)
        return {"documents": []}

def grade_documents_rag_node(state: RAGGraphState, doc_grader_chain):
    """Grades retrieved documents for relevance against the ORIGINAL CLARIFIED query."""
    print("--- RAG NODE: Grade Documents ---")
    question_to_grade_against = state["original_clarified_question"]
    documents = state["documents"]
    print(f"Grading {len(documents)} documents against original clarified intent: '{question_to_grade_against}'")

    filtered_docs = []
    web_search_needed = "No" # Default

    if not documents:
         print("---No documents retrieved, triggering web search.---")
         web_search_needed = "Yes"
    else:
        all_irrelevant = True
        for i, d in enumerate(documents):
            doc_content = d.page_content
            doc_source = d.metadata.get("source", "Unknown Source")
            print(f"  Grading doc {i+1}/{len(documents)} (Source: {doc_source})...", end="")
            if not doc_content.strip(): # Skip empty docs
                print(" SKIPPED (empty)")
                continue
            try:
                # Use original clarified query for grading
                score: GradeDocuments = doc_grader_chain.invoke({"question": question_to_grade_against, "document": doc_content})
                grade = score.binary_score.lower().strip()

                if grade == "yes":
                    print(f" RELEVANT")
                    filtered_docs.append(d)
                    all_irrelevant = False
                else:
                    print(f" NOT Relevant")
            except Exception as e:
                print(f" ERROR Grading Document (Source: {doc_source}): {e}---")
                # If grading fails for a doc, should we trigger web search?
                # Let's say yes, as we can't be sure about relevance.
                web_search_needed = "Yes"

        if all_irrelevant and documents: # Only trigger if we had docs but none were relevant
            print("---All retrieved documents were graded irrelevant. Triggering web search.---")
            web_search_needed = "Yes"
        elif not all_irrelevant:
             print(f"---Found {len(filtered_docs)} relevant documents from DB.---")


    print(f"---Web Search Decision: {web_search_needed}---")
    # Update state with filtered docs and decision
    return {"documents": filtered_docs, "web_search_needed": web_search_needed}

def rewrite_query_for_web_rag(state: RAGGraphState, question_rewriter_web_chain):
    """Rewrites the original clarified query for effective web search."""
    print("--- RAG NODE: Rewrite Query for Web Search ---")
    original_clarified = state["original_clarified_question"] # Rewrite based on the clarified intent
    print(f"Original clarified query for Web Rewrite: {original_clarified}")
    try:
        web_query = question_rewriter_web_chain.invoke({"question": original_clarified})
        if not web_query:
            print("WARN: Web query rewriter returned empty string, using original clarified query.")
            web_query = original_clarified
    except Exception as e:
        print(f"ERROR during web query rewriting: {e}. Using original clarified query.")
        web_query = original_clarified

    print(f"Rewritten for Web: {web_query}")
    # Update the 'question' field specifically for the web_search node
    return {"question": web_query}

def web_search_rag(state: RAGGraphState, tavily_tool: TavilySearchResults):
    """Performs web search using Tavily."""
    print("--- RAG NODE: Web Search ---")
    web_query = state["question"] # This is the web-optimized query
    original_documents = state["documents"] # Keep relevant DB docs
    print(f"Searching web for: '{web_query}'")

    web_results_docs = []
    try:
        # Use the passed Tavily tool instance
        docs_dict_list = tavily_tool.invoke(web_query) # Tavily call

        if docs_dict_list and isinstance(docs_dict_list, list):
             print(f"---Web Search raw results count: {len(docs_dict_list)}---")
             for i, doc_dict in enumerate(docs_dict_list):
                 # Limit content length? Tavily's max_tokens might handle this.
                 content = doc_dict.get("content", "").strip()
                 url = doc_dict.get("url", "N/A")
                 if content:
                     metadata = {"source": "Web Search", "url": url}
                     web_results_docs.append(Document(page_content=content, metadata=metadata))
                     print(f"  Added web result {i+1}: {url}")
                 else:
                      print(f"  Skipped web result {i+1} (no content): {url}")


        if web_results_docs:
            print(f"---Web Search added {len(web_results_docs)} formatted results.---")
            combined_documents = original_documents + web_results_docs
        else:
            print("---Web Search returned no usable results. Proceeding with DB documents only.---")
            combined_documents = original_documents # Use only DB docs if web fails

    except Exception as e:
        print(f"---ERROR During Web Search: {e}---")
        # Proceed with only the original DB documents
        combined_documents = original_documents

    # IMPORTANT: Do NOT revert 'question' here. Generation should use the ORIGINAL clarified question.
    # The state update only needs to modify 'documents'.
    return {"documents": combined_documents}

def generate_answer_rag(state: RAGGraphState, qa_rag_chain):
    """Generates the final answer using the gathered context and original clarified query."""
    print("--- RAG NODE: Generate Answer ---")
    # Use the ORIGINAL clarified question for generation, regardless of rewrites
    question_for_llm = state["original_clarified_question"]
    documents = state["documents"] # Use combined DB + Web docs (if any)
    print(f"Generating answer for original clarified intent: '{question_for_llm}'")
    print(f"Using {len(documents)} documents as context.")

    if not documents:
        print("---No documents available for generation.---")
        # Provide a standard response indicating lack of info
        generation = f"I could not find relevant information in the local documents or via web search to answer your question: '{question_for_llm}'"
    else:
         try:
             # Use the passed QA RAG chain instance
             generation = qa_rag_chain.invoke({"context": documents, "question": question_for_llm})
             print("---Generation Complete---")
         except Exception as e:
             print(f"---ERROR During Generation: {e}---")
             generation = f"Sorry, an error occurred while generating the final answer: {e}"

    # Update the 'generation' field in the state
    return {"generation": generation}

# --- RAG Conditional Edges ---
def decide_to_generate_rag(state: RAGGraphState):
    """Decides whether to proceed to generation or initiate web search based on grading."""
    print("--- RAG EDGE: Decide to Generate or Web Search ---")
    web_search_needed = state.get("web_search_needed", "No") # Default to No if not set

    if web_search_needed == "Yes":
        print("---DECISION: Routing to Web Search Path (rewrite_query_web)---")
        return "rewrite_query_web"
    else:
        # If we have relevant documents, or even if we had no documents initially
        # but web search wasn't explicitly triggered by grading (e.g., retrieval error),
        # we proceed to generation with whatever documents we have (which might be none).
        print("---DECISION: Routing to Generate Answer (generate_answer)---")
        return "generate_answer"

# --- Build RAG Graph ---
def get_agentic_rag_app(retriever: BaseRetriever, openai_api_key: str, tavily_api_key: str):
    """Builds and compiles the agentic RAG graph."""
    if not retriever:
       print("ERROR: Retriever not available. Cannot build RAG agent.")
       return None

    # Initialize components
    llm, structured_llm_grader, rewriter_llm, tv_search = get_rag_components(openai_api_key, tavily_api_key)
    if not all([llm, structured_llm_grader, rewriter_llm, tv_search]):
        print("ERROR: Failed to initialize one or more RAG components.")
        return None

    # Setup chains
    doc_grader_chain, question_rewriter_web_chain, initial_query_rewriter_chain, qa_rag_chain = setup_rag_chains(
        llm, structured_llm_grader, rewriter_llm
    )

    print("Building RAG Agent Graph...")
    rag_graph = StateGraph(RAGGraphState)

    # Bind tools/chains to nodes using partial
    bound_rewrite_db = functools.partial(rewrite_clarified_query_for_db, initial_query_rewriter_chain=initial_query_rewriter_chain)
    bound_retrieve = functools.partial(retrieve_rag, retriever=retriever)
    bound_grade = functools.partial(grade_documents_rag_node, doc_grader_chain=doc_grader_chain)
    bound_rewrite_web = functools.partial(rewrite_query_for_web_rag, question_rewriter_web_chain=question_rewriter_web_chain)
    bound_web_search = functools.partial(web_search_rag, tavily_tool=tv_search)
    bound_generate = functools.partial(generate_answer_rag, qa_rag_chain=qa_rag_chain)


    # Add nodes
    rag_graph.add_node("query_rewriter_db", bound_rewrite_db)
    rag_graph.add_node("retrieve", bound_retrieve)
    rag_graph.add_node("grade_documents", bound_grade)
    rag_graph.add_node("rewrite_query_web", bound_rewrite_web)
    rag_graph.add_node("web_search", bound_web_search)
    rag_graph.add_node("generate_answer", bound_generate)

    # Set entry point
    rag_graph.set_entry_point("query_rewriter_db")

    # Add edges
    rag_graph.add_edge("query_rewriter_db", "retrieve")
    rag_graph.add_edge("retrieve", "grade_documents")
    rag_graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate_rag, # Decision function
        {
            "rewrite_query_web": "rewrite_query_web", # If Yes for web search
            "generate_answer": "generate_answer",     # If No for web search
        },
    )
    rag_graph.add_edge("rewrite_query_web", "web_search")
    rag_graph.add_edge("web_search", "generate_answer") # Always generate after web search (even if it failed)
    rag_graph.add_edge("generate_answer", END) # Generation is the final step

    # Compile the graph
    try:
        print("Compiling RAG Agent Graph...")
        agentic_rag_compiled = rag_graph.compile()
        print("RAG Agent graph compiled successfully.")
        # Optionally save graph image
        # try_save_graph_image(agentic_rag_compiled, config.RAG_GRAPH_IMAGE_PATH)
        return agentic_rag_compiled
    except Exception as e:
        print(f"ERROR: Failed to compile RAG agent graph: {e}")
        traceback.print_exc()
        return None


# --- Helper to Save Graph Image (Optional) ---
def try_save_graph_image(graph, path):
    """Attempts to save a graph visualization, requires optional dependencies."""
    if not path: return
    try:
        print(f"Attempting to save graph image to {path}...")
        # Check if the necessary method exists
        if hasattr(graph, 'get_graph') and hasattr(graph.get_graph(), 'draw_mermaid_png'):
            image_bytes = graph.get_graph().draw_mermaid_png()
            with open(path, "wb") as f:
                f.write(image_bytes)
            print(f"Graph visualization saved as {path}")
        else:
             print("INFO: Could not generate graph image: Method 'get_graph().draw_mermaid_png()' not found.")
             print("INFO: Ensure 'pygraphviz' and 'matplotlib' are installed, and playwright dependencies (`playwright install-deps`) if using mermaid.")
    except ImportError as img_e:
        print(f"WARNING: Could not save graph image due to missing dependencies: {img_e}.")
        print("INFO: Try `pip install pygraphviz matplotlib 'langchain[graph]'` or check mermaid installation.")
    except Exception as img_e:
        print(f"WARNING: Could not save graph image: {img_e}")