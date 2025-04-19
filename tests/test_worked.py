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

# Ensure resources are initialized ONCE before running tests
# NOTE: Depending on DeepEval's test runner, this might run per test or per session.
# If it runs per test, initialization might be slow. Consider fixtures if using pytest directly.
print("Initializing resources for testing...")
if not initialize_resources():
    pytest.fail("Failed to initialize necessary resources for testing. Check API keys and paths.", pytrace=False)
    # Note: pytest.fail might not work perfectly outside a test function depending on runner.
    # Raising an exception might be better if this runs early.
    # raise RuntimeError("Failed to initialize necessary resources for testing.")


# Define your test cases
# Each test case needs an input query.
# For metrics like Faithfulness, you also need the retrieval_context (which is tricky here as it's internal).
# For metrics like AnswerRelevancy, you need the input query.
# For correctness/custom checks, you might need an 'expected_output' (golden answer).

# Example Test Case 1: Basic Relevance
test_case_relevance = LLMTestCase(
    input="What are the basic requirements to register a private limited company in Bangladesh?",
    # actual_output will be populated by the test function
)

# Example Test Case 2: Specific Law (requires good RAG)
test_case_specific_law = LLMTestCase(
    input="Explain the procedure for obtaining bail for an offense under Section 19(1) table 9(kha) of the Narcotics Control Act, 2018.",
    # You might have a golden answer for this if possible
    expected_output="The procedure involves applying to the relevant court (often Sessions Judge or Metropolitan Sessions Judge)... [Details based on actual law/context]", # Replace with actual expected key points
    # To test Faithfulness, you would need to somehow capture the context used by the RAG agent during the run_legal_assistant_conversation call.
    # This requires modifying run_legal_assistant_conversation to return the 'documents' used for generation.
    # retrieval_context=["Context snippet 1...", "Context snippet 2 from web..."] # This needs to be dynamically generated
)

# Example Test Case 3: Vague Query (testing clarification indirectly via output quality)
test_case_vague = LLMTestCase(
    input="Tell me about land law."
)

@pytest.mark.parametrize(
    "test_case",
    [
        test_case_relevance,
        test_case_specific_law,
        test_case_vague,
        # Add more test cases here
    ]
)
def test_legal_assistant_output(test_case: LLMTestCase):
    # --- Arrange ---
    # Input is already in test_case.input

    # --- Act ---
    # Run the conversation function to get the actual output
    # Modify run_legal_assistant_conversation to return context if needed for Faithfulness
    result = run_legal_assistant_conversation(test_case.input)

    # Check for errors during the run
    if result['status'] == 'Error':
        pytest.fail(f"Assistant run failed for input '{test_case.input}'. Error: {result['error_message']}", pytrace=False)

    actual_output = result['final_answer']
    test_case.actual_output = actual_output # Assign for metric calculation

    # If testing Faithfulness, extract context (requires modification to run_... function)
    # retrieval_context_list = result.get('retrieval_context_for_test', []) # Example key
    # test_case.retrieval_context = [doc.page_content for doc in retrieval_context_list] # Or formatted string

    # --- Assert ---
    # Define metrics for this specific test run
    # Use AnswerRelevancy by default
    relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=config.RAG_LLM_MODEL, include_reason=True)

    # Use Faithfulness *only* if context is available
    # faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=config.RAG_LLM_MODEL, include_reason=True)
    # metrics_to_run = [relevancy_metric, faithfulness_metric] if test_case.retrieval_context else [relevancy_metric]
    metrics_to_run = [relevancy_metric] # Start simple

    # You might add custom checks or other DeepEval metrics here

    # Run the assertion
    assert_test(test_case, metrics_to_run)

# Example of a custom metric (if needed)
# from deepeval.metrics import BaseMetric
# class ContainsKeywordMetric(BaseMetric):
#     # ... implement __init__, measure, is_successful ...
#     pass

# To run these tests, use pytest:
# pip install pytest
# pytest tests/test_legal_assistant.py