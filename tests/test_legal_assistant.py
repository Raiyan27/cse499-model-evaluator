import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric # Add more metrics as needed
from deepeval.test_case import LLMTestCase
import os
import sys

# Add the parent directory to sys.path to import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the necessary functions/classes from your project
import config # To ensure config is loaded (e.g., for API keys if metrics need them)
from main import initialize_resources, run_legal_assistant_conversation

# --- Test Data Setup ---
# Store test parameters in a more basic structure first (e.g., list of dicts)
# We will create the LLMTestCase object *inside* the test function.
test_data = [
    {
        "id": "relevance_test",
        "input": "What are the basic requirements to register a private limited company in Bangladesh?",
        "expected_output": None, # Provide golden answer if available for other metrics/checks
        "metrics": [AnswerRelevancyMetric(threshold=0.6, model=config.RAG_LLM_MODEL, include_reason=True)]
    },
    {
        "id": "specific_law_test",
        "input": "Examine the nuances of the RTI law in promoting governmental accountability and citizen empowerment.",
        "expected_output": "The Right to Information (RTI) law plays a critical role in promoting governmental accountability and empowering citizens in Bangladesh. By ensuring maximum disclosure and proactive communication, the RTI law holds authorities accountable for their actions. Citizens can demand information regarding government activities, policies, expenditures, and NGO operations, thereby facilitating transparency.\n\nThis law recognizes that information is a public asset, underscored by the Bangladesh Constitution, which states that the people are the ultimate owners of state power. The RTI law establishes mechanisms for citizens to access information without needing to justify their requests, thereby democratizing information access.\n\nMoreover, the RTI law aids marginalized communities by allowing them to seek assistance from NGOs in requesting information, addressing their specific needs, and promoting their rights. Through this framework, the RTI law not only informs but also empowers citizens to engage actively in governance, ensuring a more participatory democratic process. Thus, the RTI law embodies the principles of accountability and citizen empowerment, essential for good governance.", # Replace with actual expected key points if checking correctness
        "metrics": [
            AnswerRelevancyMetric(threshold=0.7, model=config.RAG_LLM_MODEL, include_reason=True),
            # Add FaithfulnessMetric here *if* you modify run_... to return context
            # FaithfulnessMetric(threshold=0.7, model=config.RAG_LLM_MODEL, include_reason=True)
        ]
    },
    {
        "id": "vague_query_test",
        "input": "What does the RTI law state about citizens' rights to access information from authorities?",
        "expected_output": "The RTI law states that every citizen has the right to information from authorities, which are obligated to provide this information upon request. It ensures transparency and accountability by allowing citizens access to information about government activities, policies, decisions, expenditures, and NGO operations using public funds. The law embodies the principles of maximum information disclosure, proactive disclosure, and limits on exemptions, promoting citizen participation in governance.",
        "metrics": [AnswerRelevancyMetric(threshold=0.5, model=config.RAG_LLM_MODEL, include_reason=True)] # Maybe lower threshold for vague
    },
    # Add more test dictionaries here
]

# --- Initialization ---
# Ensure resources are initialized ONCE before running tests
# Using pytest fixture might be cleaner if tests run parallel or setup is complex
@pytest.fixture(scope="session", autouse=True)
def setup_resources():
    print("\nInitializing resources for testing session...")
    initialized = initialize_resources()
    if not initialized:
        pytest.fail("Failed to initialize necessary resources for testing. Check API keys and paths.", pytrace=False)
    print("Resources initialized.")
    # Add teardown logic here if needed (e.g., close DB connections)
    # yield
    # print("Tearing down resources...")


# --- Test Function ---
# Use parametrize to run the test function for each item in test_data
@pytest.mark.parametrize("test_params", test_data, ids=[item['id'] for item in test_data])
def test_legal_assistant_output(test_params: dict):
    """
    Tests the legal assistant's output for a given input using DeepEval metrics.
    """
    # --- Arrange ---
    test_input = test_params["input"]
    expected_output = test_params.get("expected_output", None) # Optional
    metrics_to_run = test_params["metrics"]

    print(f"\n--- Running Test ID: {test_params['id']} ---")
    print(f"Input: {test_input}")

    # --- Act ---
    # Run the conversation function to get the actual output
    # Modify run_legal_assistant_conversation to return context if needed for Faithfulness
    result = run_legal_assistant_conversation(test_input)

    # Check for errors during the run
    if result['status'] == 'Error':
        pytest.fail(f"Assistant run failed for input '{test_input}'. Error: {result['error_message']}", pytrace=False)

    actual_output = result['final_answer']
    print(f"Actual Output: {actual_output[:200]}...") # Print snippet of output

    # If testing Faithfulness, extract context (requires modification to run_... function)
    # retrieval_context_list = result.get('retrieval_context_for_test', []) # Example key
    # retrieval_context_str_list = [doc.page_content for doc in retrieval_context_list]

    # *** Create the LLMTestCase HERE, now that we have actual_output ***
    test_case = LLMTestCase(
        input=test_input,
        actual_output=actual_output,
        expected_output=expected_output, # Pass if available
        # retrieval_context=retrieval_context_str_list # Pass if available and testing Faithfulness
    )

    # --- Assert ---
    # Run the assertion using the dynamically created test_case and specified metrics
    print(f"Asserting with metrics: {[m.__class__.__name__ for m in metrics_to_run]}")
    assert_test(test_case, metrics_to_run)
    print(f"--- Test ID: {test_params['id']} Passed ---")

# To run these tests:
# pip install pytest deepeval
# pytest tests/test_legal_assistant.py