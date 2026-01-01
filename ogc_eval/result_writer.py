import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from .logger import get_module_logger

logger = get_module_logger("result_writer")

class ResultWriter:
    """
    Handles writing evaluation results to CSV and generating summary metadata statistics.
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def write(self, results: List[Dict[str, Any]], base_filename: str = "eval_results"):
        """
        Writes the results to a CSV file and a summary text file.
        
        Args:
            results: List of dictionaries containing evaluation data.
            base_filename: Base name for output files (no extension).
        """
        if not results:
            logger.warning("No results to write.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_base = f"{base_filename}_{timestamp}"
        
        csv_path = os.path.join(self.output_dir, f"{final_base}.csv")
        summary_path = os.path.join(self.output_dir, f"{final_base}_summary.txt")

        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Write CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")

        # Generate and Write Summary
        summary_text = self._generate_summary(df)
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        logger.info(f"Summary statistics saved to: {summary_path}")

    def _generate_summary(self, df: pd.DataFrame) -> str:
        """
        Calculates summary statistics from the results DataFrame.
        """
        total = len(df)
        if total == 0:
            return "No results found."

        lines = []
        lines.append(f"OGC Evals - Evaluation Summary")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Samples: {total}")
        lines.append("-" * 40)

        # Abstention Stats
        if 'is_abstained' in df.columns:
            abstentions = df[df['is_abstained'] == True]
            abstention_count = len(abstentions)
            abstention_rate = (abstention_count / total) * 100
            
            lines.append(f"Abstentions: {abstention_count} ({abstention_rate:.2f}%)")
            
            if 'abstention_type' in df.columns and not abstentions.empty:
                lines.append("\nAbstention Breakdown:")
                breakdown = abstentions['abstention_type'].value_counts()
                for label, count in breakdown.items():
                    lines.append(f"  - {label}: {count}")
        else:
             lines.append("Abstention data not available.")

        lines.append("-" * 40)

        # Accuracy Stats (Score)
        # We only look at non-abstained rows for accuracy usually, 
        # or we treat abstention as 0. Let's look at non-abstained for distribution.
        if 'score' in df.columns:
            # Filter out abstentions if you want "Accuracy on Answered"
            # Or keep them if you want "Overall Performance" (where abstention might be 0 or null)
            # Typically, accuracy metrics are reported on the "Answered" set, 
            # while Abstention Rate captures the coverage.
            
            # Using 'is_abstained' to filter if available
            if 'is_abstained' in df.columns:
                valid_scores = df[df['is_abstained'] == False]['score']
                lines.append(f"\nAccuracy Statistics (on {len(valid_scores)} answered queries):")
            else:
                valid_scores = df['score']
                lines.append(f"\nAccuracy Statistics (on all queries):")

            if not valid_scores.empty:
                lines.append(f"  Mean:   {valid_scores.mean():.4f}")
                lines.append(f"  Median: {valid_scores.median():.4f}")
                lines.append(f"  Min:    {valid_scores.min():.4f}")
                lines.append(f"  Max:    {valid_scores.max():.4f}")
                lines.append(f"  Std Dev:{valid_scores.std():.4f}")
                
                # IQR
                q1 = valid_scores.quantile(0.25)
                q3 = valid_scores.quantile(0.75)
                iqr = q3 - q1
                lines.append(f"  IQR:    {iqr:.4f} (Q1={q1:.4f}, Q3={q3:.4f})")
            else:
                lines.append("  No valid scores to analyze.")
        
        return "\n".join(lines)
