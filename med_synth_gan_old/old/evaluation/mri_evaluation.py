import os
import numpy as np

from metrics import compute_metrics


def evaluate_predictions(reference_dir, prediction_dir, labels_or_regions):
    results = []
    for case in os.listdir(reference_dir):
        ref_path = os.path.join(reference_dir, case)
        pred_path = os.path.join(prediction_dir, case)
        
        if os.path.exists(pred_path):

            for file in os.listdir(ref_path):
                file_number = file.split('_')[0]
                pred_files = os.listdir(pred_path) 
                for pred_file in pred_files:
                    if file_number in pred_file:
                        pred_path = os.path.join(pred_path, pred_file)

                ref_path = os.path.join(ref_path, file)
                print(ref_path, pred_path)                
                case_results = compute_metrics(
                    reference_file=ref_path,
                    prediction_file=pred_path,
                    labels_or_regions=labels_or_regions
                )
                results.append(case_results)
    
    # Aggregate metrics across cases
    aggregated_metrics = aggregate_results(results)
    return aggregated_metrics

def aggregate_results(results):
    """
    Aggregates metrics across all cases.
    """
    print(results)
    metrics_summary = {}
    for metric_name in results[0]['metrics']:
        metrics_summary[metric_name] = {
            "mean": np.mean([res['metrics'][metric_name] for res in results]),
            "std": np.std([res['metrics'][metric_name] for res in results]),
        }
    return metrics_summary


if __name__ == "__main__":
    reference_dir = "../data/MRI/reference"
    prediction_dir = "../data/MRI/output"
    labels_or_regions = [42, 43, 44, 84]

    metrics = evaluate_predictions(reference_dir, prediction_dir, labels_or_regions)
    print(metrics)