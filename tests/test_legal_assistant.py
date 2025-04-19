import pytest
from deepeval import assert_test
# Import necessary metrics
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval, # General Evaluation metric for comparing against expected output
)
from deepeval.test_case import LLMTestCase
from deepeval.scorer import Scorer # For GEval configuration
import os
import sys
from typing import List

# Add the parent directory to sys.path to import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the necessary functions/classes from your project
import config
from main import initialize_resources, run_legal_assistant_conversation

# --- Configure Evaluation Model (Optional but recommended for consistency) ---
# You can configure the model used by the metrics themselves
# Scorer.set_evaluation_model(config.RAG_LLM_MODEL) # Example: Use the same model as your RAG agent

# --- Test Data Setup ---
# Define metrics instances here for clarity or within the test_data dict
# Note: Ensure thresholds are appropriate for your expectations
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model=config.RAG_LLM_MODEL, include_reason=True)
faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=config.RAG_LLM_MODEL, include_reason=True)
contextual_precision_metric = ContextualPrecisionMetric(threshold=0.7, model=config.RAG_LLM_MODEL, include_reason=True)
contextual_recall_metric = ContextualRecallMetric(threshold=0.7, model=config.RAG_LLM_MODEL, include_reason=True)

# GEval for comparing actual vs expected output based on certain criteria
# Criteria can be: Correctness, Relevance, Coherence, Similarity etc.
# Let's define one for Correctness/Similarity relative to the expected output
correctness_metric = GEval(
    name="Correctness",
    criteria="Correctness - determine if the actual output is factually correct based on the expected output.",
    evaluation_params=["input", "actual_output", "expected_output"], # Specify which fields to use
    model=config.RAG_LLM_MODEL, # Specify model if desired
    threshold=0.6 # Adjust threshold
)
similarity_metric = GEval(
    name="Similarity",
    criteria="Similarity - evaluate how similar the meaning and content of the actual output is to the expected output.",
    evaluation_params=["input", "actual_output", "expected_output"],
    model=config.RAG_LLM_MODEL,
    threshold=0.7
)


test_data = [
    {
        "id": "relevance_test_plc", # More specific ID
        "input": "What are the basic requirements to register a private limited company in Bangladesh?",
        "expected_output": "Registering a private limited company in Bangladesh requires several steps managed by the Registrar of Joint Stock Companies and Firms (RJSC). Key requirements generally include: obtaining name clearance, preparing and submitting the Memorandum and Articles of Association, obtaining necessary digital certificates for directors, filing relevant forms (like Form XII for directors), and paying registration fees. Specific documents needed often include director details (NID/Passport), shareholder information, registered office address, and potentially trade licenses.", # More realistic expected output
        "metrics": [
            answer_relevancy_metric, # Still check relevance to input
            faithfulness_metric,
            contextual_precision_metric,
            contextual_recall_metric,
            correctness_metric,
            similarity_metric
            ]
    },
    {
        "id": "off_topic_test_icc", # More specific ID
        "input": "What is ICC?",
        "expected_output": "Information regarding the International Criminal Court (ICC) or the International Cricket Council (ICC) is outside the scope of Bangladeshi legal documents.", # Expected polite refusal or irrelevant answer
        "metrics": [
            answer_relevancy_metric, # Should ideally score low if it tries to answer legally
             # Faithfulness might pass if it says "I cannot answer" based on empty context
            # correctness_metric, # Compare refusal to expected refusal
            # similarity_metric # Compare refusal to expected refusal
            ]
    },
    {
        "id": "specific_law_test_rti_nuances", # More specific ID
        "input": "Examine the nuances of the RTI law in promoting governmental accountability and citizen empowerment.",
        # Using your provided expected output
        "expected_output": "The Right to Information (RTI) law plays a critical role in promoting governmental accountability and empowering citizens in Bangladesh. By ensuring maximum disclosure and proactive communication, the RTI law holds authorities accountable for their actions. Citizens can demand information regarding government activities, policies, expenditures, and NGO operations, thereby facilitating transparency.\n\nThis law recognizes that information is a public asset, underscored by the Bangladesh Constitution, which states that the people are the ultimate owners of state power. The RTI law establishes mechanisms for citizens to access information without needing to justify their requests, thereby democratizing information access.\n\nMoreover, the RTI law aids marginalized communities by allowing them to seek assistance from NGOs in requesting information, addressing their specific needs, and promoting their rights. Through this framework, the RTI law not only informs but also empowers citizens to engage actively in governance, ensuring a more participatory democratic process. Thus, the RTI law embodies the principles of accountability and citizen empowerment, essential for good governance.",
        "metrics": [
            answer_relevancy_metric,
            faithfulness_metric,
            contextual_precision_metric,
            contextual_recall_metric,
            correctness_metric,
            similarity_metric
            ]
    },
    {
        "id": "vague_query_test_rti_rights", # More specific ID
        "input": "What does the RTI law state about citizens' rights to access information from authorities?",
        # Using your provided expected output
        "expected_output": "The RTI law states that every citizen has the right to information from authorities, which are obligated to provide this information upon request. It ensures transparency and accountability by allowing citizens access to information about government activities, policies, decisions, expenditures, and NGO operations using public funds. The law embodies the principles of maximum information disclosure, proactive disclosure, and limits on exemptions, promoting citizen participation in governance.",
        "metrics": [
            answer_relevancy_metric,
            faithfulness_metric,
            contextual_precision_metric,
            contextual_recall_metric, # See if context contained enough for this summary
            correctness_metric,
            similarity_metric
            ]
    },
    # Add more test dictionaries here
]

# --- Initialization ---
@pytest.fixture(scope="session", autouse=True)
def setup_resources():
    print("\nInitializing resources for testing session...")
    initialized = initialize_resources()
    if not initialized:
        pytest.fail("Failed to initialize necessary resources for testing. Check API keys and paths.", pytrace=False)
    print("Resources initialized.")
    yield
    print("\nTesting Session Finished.")
    # Add teardown logic here if needed


# --- Test Function ---
@pytest.mark.parametrize("test_params", test_data, ids=[item['id'] for item in test_data])
def test_legal_assistant_output(test_params: dict):
    """
    Tests the legal assistant's output against expected results and context quality.
    """
    # --- Arrange ---
    test_input = test_params["input"]
    expected_output = test_params.get("expected_output", None)
    metrics_to_run: List = test_params["metrics"] # List of metric instances

    print(f"\n--- Running Test ID: {test_params['id']} ---")
    print(f"Input: {test_input}")
    if expected_output:
        print(f"Expected Output (Snippet): {expected_output[:100]}...")

    # --- Act ---
    result = run_legal_assistant_conversation(test_input)

    if result['status'] == 'Error':
        pytest.fail(f"Assistant run failed for input '{test_input}'. Error: {result['error_message']}", pytrace=False)

    actual_output = result['final_answer']
    retrieved_docs = result.get('retrieval_context_docs', [])
    # Format context for DeepEval (list of strings)
    retrieval_context_list = [doc.page_content for doc in retrieved_docs]

    print(f"Actual Output (Snippet): {actual_output[:200]}...")
    print(f"Retrieved Context Documents: {len(retrieval_context_list)}")

    # *** Create the LLMTestCase with all available information ***
    test_case = LLMTestCase(
        input=test_input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context_list if retrieval_context_list else None # Pass context if available
    )

    # --- Filter Metrics based on available data ---
    # Some metrics require specific fields (e.g., Faithfulness needs context, Recall needs expected_output)
    final_metrics_to_run = []
    print("Metrics configured for this test:")
    for metric in metrics_to_run:
        can_run = True
        metric_name = metric.__class__.__name__
        if isinstance(metric, FaithfulnessMetric) and not test_case.retrieval_context:
            print(f"  - Skipping {metric_name}: Requires retrieval_context.")
            can_run = False
        if isinstance(metric, (ContextualPrecisionMetric, ContextualRecallMetric)) and not test_case.retrieval_context:
             print(f"  - Skipping {metric_name}: Requires retrieval_context.")
             can_run = False
        if isinstance(metric, ContextualRecallMetric) and not test_case.expected_output:
            print(f"  - Skipping {metric_name}: Requires expected_output.")
            can_run = False
        if isinstance(metric, GEval) and "expected_output" in metric.evaluation_params and not test_case.expected_output:
            print(f"  - Skipping {metric_name} (Criteria: {metric.name}): Requires expected_output.")
            can_run = False
        # Add similar checks for other metrics if they have specific requirements

        if can_run:
            print(f"  + Including {metric_name} (Threshold: {metric.threshold})")
            final_metrics_to_run.append(metric)

    if not final_metrics_to_run:
         pytest.skip("No applicable metrics could be run for this test case based on available data (e.g., missing context or expected output).")


    # --- Assert ---
    print(f"Asserting with {len(final_metrics_to_run)} applicable metrics...")
    assert_test(test_case, final_metrics_to_run) # Use the filtered list
    print(f"--- Test ID: {test_params['id']} Passed ---")

# To run these tests:
# pip install pytest deepeval
# pytest tests/test_legal_assistant.py