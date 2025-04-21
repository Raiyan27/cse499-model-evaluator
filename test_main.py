import pytest
from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric
from main import run_legal_assistant_conversation, initialize_resources

# Imports for context retrival evaluation
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

@pytest.fixture(scope="session", autouse=True) 
def setup_resources():
    print("\nInitializing resources for tests...")
    initialized = initialize_resources()
    if not initialized:
        pytest.fail("CRITICAL: Failed to initialize test resources.")
    print("Test resources initialized successfully.")

# --- Your Test Function ---
def test_correctness():
    test_case_input = "Imagine a scenario where all citizens effectively utilize the RTI Act. How might this empowerment transform civic engagement and government transparency in Bangladesh?"
    test_case_expected_output = "If all citizens effectively utilize the RTI Act in Bangladesh, it could lead to a significant transformation in civic engagement and government transparency. Empowered by their right to access information, citizens would be better equipped to participate in the decision-making processes that affect their lives. This increased engagement could foster a culture of accountability, where government officials are more responsive to the public's needs and concerns.\n\nThe proactive disclosure of information mandated by the RTI Act would ensure transparency, allowing citizens to monitor government activities, expenditures, and public services closely. This could reduce corruption and promote good governance, as authorities would be held accountable for their actions. Furthermore, NGOs and the media could effectively leverage the RTI Act to advocate for marginalized communities and gather crucial data to inform social audits and public service evaluations.\n\nUltimately, widespread utilization of the RTI Act would create a more informed and active citizenry, leading to enhanced governance, stronger democratic practices, and a more equitable society."

    print(f"\nRunning conversation for test input: '{test_case_input}'")
    result_dict = run_legal_assistant_conversation(test_case_input)
    assert result_dict['status'] == "Success", f"run_legal_assistant_conversation failed: {result_dict.get('error_message', 'Unknown error')}"

    actual_output_from_run = result_dict['final_answer']
    print(f"Actual output received: {actual_output_from_run}") 

    # --- DeepEval Section ---
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct and relevant based on the 'expected output', considering the medical nature of the 'input' query.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT], # Include INPUT for better context
        threshold=0.5,# Adjust threshold as needed
        model="gpt-4o-mini",
    )

    test_case = LLMTestCase(
        input=test_case_input,                 
        actual_output=actual_output_from_run,  
        expected_output=test_case_expected_output 

    )
    assert_test(test_case, [correctness_metric])

    # evaluate(
    # test_cases=test_case,
    # metrics=correctness_metric,
    # hyperparameters={"model": "gpt-4o-mini"}
    # )