"""
MLflowlite Evaluation Demo

This example demonstrates how to evaluate GenAI applications using MLflow-style
evaluation with built-in and custom scorers.

Based on: https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/
"""

import os
import mlflowlite as mf
from mlflowlite import evaluate, Correctness, Guidelines, scorer

# Set up environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-api-key")

# Enable MLflow tracking
mf.set_mlflow_tracking(True)
mf.set_experiment_name("Evaluation Demo")


def main():
    print("=" * 60)
    print("MLflowlite Evaluation Demo")
    print("=" * 60)
    
    # Step 1: Define a prediction function
    def qa_predict_fn(question: str) -> str:
        """Simple Q&A prediction function."""
        response = mf.completion(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions concisely.",
                },
                {"role": "user", "content": question},
            ],
        )
        return response.content
    
    # Step 2: Define evaluation dataset
    eval_dataset = [
        {
            "inputs": {"question": "What is the capital of France?"},
            "expectations": {"expected_response": "Paris"},
        },
        {
            "inputs": {"question": "Who was the first person to build an airplane?"},
            "expectations": {"expected_response": "Wright Brothers"},
        },
        {
            "inputs": {"question": "Who wrote Romeo and Juliet?"},
            "expectations": {"expected_response": "William Shakespeare"},
        },
        {
            "inputs": {"question": "What is 2+2?"},
            "expectations": {"expected_response": "4"},
        },
    ]
    
    print(f"\n📊 Created evaluation dataset with {len(eval_dataset)} samples")
    
    # Step 3: Define scorers
    
    # Built-in scorer: Checks factual correctness using LLM-as-a-Judge
    correctness_scorer = Correctness()
    
    # Built-in scorer: Checks if response meets guidelines
    english_scorer = Guidelines(
        name="is_english",
        guidelines="The answer must be in English"
    )
    
    # Custom scorer: Check if response is concise
    @scorer
    def is_concise(outputs: str) -> bool:
        """Check if response is concise (less than 20 words)"""
        return len(outputs.split()) <= 20
    
    # Custom scorer with rationale
    @scorer
    def is_polite(outputs: str) -> tuple:
        """Check if response is polite"""
        polite_words = ["please", "thank you", "kindly", "appreciate"]
        has_polite = any(word in outputs.lower() for word in polite_words)
        return has_polite, "Response contains polite language" if has_polite else "Response lacks polite language"
    
    # Combine all scorers
    scorers = [correctness_scorer, english_scorer, is_concise, is_polite]
    
    print("\n📏 Defined 4 scorers:")
    print("  1. Correctness (LLM-as-a-Judge)")
    print("  2. English Guidelines (LLM-as-a-Judge)")
    print("  3. Conciseness (Custom function)")
    print("  4. Politeness (Custom function)")
    
    # Step 4: Run evaluation
    print("\n🚀 Running evaluation...\n")
    
    results = evaluate(
        data=eval_dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
        experiment_name="Evaluation Demo",
        run_name="qa_evaluation",
        log_to_mlflow=True,
        verbose=True,
    )
    
    # Step 5: Analyze results
    print("\n" + "=" * 60)
    print("📊 EVALUATION COMPLETE")
    print("=" * 60)
    
    # View as DataFrame
    df = results.to_dataframe()
    print("\n📋 Sample Results:")
    print(df[['input_question', 'score_correctness', 'score_is_english', 'score_is_concise']].head(2))
    
    # View aggregate scores
    print("\n🎯 Aggregate Scores:")
    for metric, score in sorted(results.aggregate_scores.items()):
        print(f"  {metric}: {score:.3f}")
    
    print(f"\n✅ Results logged to MLflow run: {results.run_id}")
    print("\n💡 View detailed results in MLflow UI:")
    print("   mlflow ui --backend-store-uri sqlite:///mlruns.db")
    

if __name__ == "__main__":
    main()

