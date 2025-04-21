import pytest
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from main import run_legal_assistant_conversation, initialize_resources

# --- Test Data (Load from CSV/JSON for hundreds of cases) ---
test_cases_data = [
    (
        "RTI_Scenario_1",
        "Imagine a scenario where all citizens effectively utilize the RTI Act. How might this empowerment transform civic engagement and government transparency in Bangladesh?",
        "Effective use of the RTI Act by all citizens in Bangladesh would likely lead to significantly increased government accountability and transparency. Citizens could monitor public projects, scrutinize spending, and demand better services. This empowerment fosters active civic participation, potentially reducing corruption and improving governance. It creates a more informed public, strengthening democratic processes.",
    ),
    (
        "Company_Reg_Basic",
        "Tell me about company registration in Bangladesh.",
        "Company registration in Bangladesh is primarily handled by the Registrar of Joint Stock Companies and Firms (RJSC). Key types include Private Limited Companies, Public Limited Companies, and Branch Offices. The process involves name clearance, submitting incorporation documents (Memorandum and Articles of Association), paying fees, and obtaining a certificate of incorporation. Specific requirements depend on the company type.",
    ),
    (
        "Vague_Land_Law",
        "ভূমি আইন", # "Land Law"
        "Bangladesh has numerous laws related to land, including acquisition, ownership, mutation, taxation, and dispute resolution. Key acts include the State Acquisition and Tenancy Act, Non-Agricultural Tenancy Act, and Land Development Tax Ordinance. Specific information depends heavily on the particular aspect of land law you are interested in, such as inheritance, purchase, leasing, or specific regional rules.",
    ),
    # --- Add hundreds more test cases here ---
    # Consider loading from a file:
    # import json
    # with open('test_cases.json', 'r') as f:
    #     test_cases_data = json.load(f)
    # Or pandas for CSV:
    # import pandas as pd
    # df = pd.read_csv('test_cases.csv')
    # test_cases_data = list(df.itertuples(index=False, name=None))

]

@pytest.fixture(scope="session", autouse=True)
def setup_resources():
    """Initializes resources once for the entire test session."""
    print("\nInitializing resources for test session...")
    initialized = initialize_resources()
    if not initialized:
        pytest.fail("CRITICAL: Failed to initialize test resources.")
    print("Test resources initialized successfully.")

# --- Parametrized Test Function ---
@pytest.mark.parametrize("test_id, test_input, expected_output", test_cases_data)
def test_legal_assistant_evaluation(test_id, test_input, expected_output):
    """
    Runs a single test case through the legal assistant and evaluates multiple metrics.
    """
    print(f"\n--- Running Test Case ID: {test_id} ---")
    print(f"Input: '{test_input}'")

    # 1. Run your main conversation function
    result_dict = run_legal_assistant_conversation(test_input)

    # 2. Basic check: Did the conversation function succeed?
    if result_dict['status'] != "Success":
        pytest.fail(f"Test Case ID '{test_id}': run_legal_assistant_conversation failed: {result_dict.get('error_message', 'Unknown error')}")

    # 3. Extract results needed for evaluation
    actual_output = result_dict.get('final_answer', None)
    retrieved_docs = result_dict.get('retrieval_context_docs', [])
    retrieval_context_strings = [doc.page_content for doc in retrieved_docs if hasattr(doc, 'page_content')]

    print(f"Actual Output: {actual_output}")
    print(f"Retrieved Context Snippets ({len(retrieval_context_strings)}): {[s[:100] + '...' for s in retrieval_context_strings[:2]]}")

    if actual_output is None:
         pytest.fail(f"Test Case ID '{test_id}': No 'final_answer' found in result_dict.")
    if not isinstance(retrieval_context_strings, list):
        print(f"Warning: Test Case ID '{test_id}': retrieval_context was not a list, defaulting to empty list.")
        retrieval_context_strings = []


    # --- DeepEval Evaluation Section ---
    # 4. Define all metrics
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' correctly and comprehensively addresses the 'input' query, based on the 'expected output' as a reference guide for a good answer.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model="gpt-4o-mini",
        threshold=0.5
    )
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.6, model="gpt-4o-mini")
    faithfulness_metric = FaithfulnessMetric(threshold=0.6, model="gpt-4o-mini")
    contextual_precision_metric = ContextualPrecisionMetric(threshold=0.6, model="gpt-4o-mini")
    contextual_recall_metric = ContextualRecallMetric(threshold=0.6, model="gpt-4o-mini")
    contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.6, model="gpt-4o-mini")

    # 5. Create the LLMTestCase 
    test_case = LLMTestCase(
        input=test_input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context_strings
    )

    # 6. Define the list of metrics to evaluate
    metrics_to_run = [
        correctness_metric,
        answer_relevancy_metric,
        faithfulness_metric,
        contextual_precision_metric,
        contextual_recall_metric,
        contextual_relevancy_metric,
    ]

    # 7. Run the evaluation using deepeval.evaluate
    print(f"\nEvaluating metrics for Test Case ID: {test_id}...")
    evaluation_results = evaluate(test_cases=[test_case], metrics=metrics_to_run)

    # 8. Process and Print Results (and optionally assert)
    print(f"\n--- Evaluation Results for Test Case ID: {test_id} ---")
    all_metrics_passed = True
    for metric_result in evaluation_results:
        print(f"  Metric: {metric_result.metrics_metadata.metric}") 
        print(f"  Score: {metric_result.metrics_metadata.score:.4f}")
        print(f"  Threshold: {metric_result.metrics_metadata.threshold}")
        print(f"  Reason: {metric_result.metrics_metadata.reason}")
        print(f"  Success: {metric_result.success}") 
        print(f"  --------------------")
        if not metric_result.success:
             all_metrics_passed = False

    # 9. Optional: Fail the pytest test if any metric didn't meet its threshold
    assert all_metrics_passed, f"Test Case ID '{test_id}' failed: One or more metrics did not meet their threshold."

    print(f"--- Finished Test Case ID: {test_id} ---")