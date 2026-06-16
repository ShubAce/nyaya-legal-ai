import sys
import os
import json
import pandas as pd
from pathlib import Path
from loguru import logger

# Add the backend directory to sys.path so we can import modules correctly
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

import evaluation.benchmark as benchmark

# Mock classes to avoid loading heavy models or requiring Gemini API keys
class MockGeminiFlashSystem(benchmark.GeminiFlashSystem):
    def __init__(self):
        # Override to avoid KeyManager initialization
        self.name = "gemini_flash"
        self.cost_per_1k_input_tokens = 0.000075

    def last_cost(self, input_tokens: int) -> float:
        return (input_tokens / 1000) * self.cost_per_1k_input_tokens

class MockNyayaFinetunedSystem(benchmark.NyayaFinetunedSystem):
    def __init__(self):
        # Override to avoid loading HuggingFace pipeline
        self.name = "nyaya_7b"
        self.cost_per_1k_input_tokens = 0.0

    def last_cost(self, input_tokens: int) -> float:
        return (input_tokens / 1000) * self.cost_per_1k_input_tokens

# Register mock classes
benchmark.SYSTEM_MAP["gemini_flash"] = MockGeminiFlashSystem
benchmark.SYSTEM_MAP["nyaya"] = MockNyayaFinetunedSystem

def main():
    system_names = ["gemini_flash", "nyaya"]
    limit = 50
    run_id = "benchmark"

    logger.info(f"Generating results for: {system_names} from checkpoints")
    
    # Load test set (limit=50)
    samples = benchmark.load_test_set(limit=limit)
    all_results = []

    for name in system_names:
        logger.info(f"\n{'='*40}\nProcessing: {name}\n{'='*40}")
        system = benchmark.SYSTEM_MAP[name]()
        # run_system_benchmark reads from the checkpoint file
        result = benchmark.run_system_benchmark(system, samples, run_id)
        all_results.append(result)

    # Print, save, and plot the aggregated results
    benchmark.print_results_table(all_results)
    benchmark.save_results(all_results, run_id)
    benchmark.plot_benchmark(all_results, benchmark.RESULTS_DIR / f"benchmark_{run_id}.png")
    
    logger.success("Results generation complete. Checkpoint files have been preserved.")

if __name__ == "__main__":
    main()
