import os
import numpy as np
import cv2
from random import randint
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

# --- Import functions from other project files ---
from cvpr_computedescriptors import compute_and_save_global_descriptor
from cvpr_evaluation import (
    evaluate_system,
    plot_precision_recall_curve,
    plot_single_pr_curve,          
    plot_confusion_matrix,
    save_results_to_file,
    print_evaluation_summary,
    get_class_from_filename, 
    load_descriptors         
)
from cvpr_compare import cvpr_compare 


def save_example_search_results(descriptor_folder, dataset_folder, distance_method, 
                                save_path, k=19):
    """
    Saves a visual plot of top-k search results for a fixed query.
    Also calculates and returns the PR data for this single query.
    
    Returns:
        pr_data (dict): Precision/Recall data for the query.
        query_basename (str): Filename of the query image.
    """
    
    all_files, all_feat = load_descriptors(descriptor_folder, dataset_folder)
    
    if not all_files:
        print(f"Warning: No descriptors found in {descriptor_folder}. Skipping example search.")
        return None, None

    query_idx = 100 # Use a fixed query index for consistent comparison
    
    # --- Query Setup ---
    query_path = all_files[query_idx]
    query_feat = all_feat[query_idx]
    query_class = get_class_from_filename(query_path)
    query_basename = os.path.basename(query_path)
    
    all_class_labels = [get_class_from_filename(f) for f in all_files]
    total_relevant = all_class_labels.count(query_class) - 1
    
    # --- Distance Calculation ---
    distances = []
    for i in range(len(all_files)):
        if i == query_idx:
            continue # Skip self-comparison
        dist = cvpr_compare(query_feat, all_feat[i], method=distance_method)
        distances.append((dist, all_files[i], all_class_labels[i])) 
    
    distances.sort(key=lambda x: x[0])
    
    # --- Single Query PR Calculation ---
    ranked_paths = [path for dist, path, cls in distances]
    ranked_classes = [cls for dist, path, cls in distances]
    ranked_dists = [dist for dist, path, cls in distances]

    y_true_binary = [1 if cls == query_class else 0 for cls in ranked_classes]
    y_scores = [-d for d in ranked_dists] # Scores = negative distance
    
    ap_score = average_precision_score(y_true_binary, y_scores) if any(y_true_binary) else 0.0
    
    query_p = np.zeros(len(distances))
    query_r = np.zeros(len(distances))
    relevant_found = 0
    
    if total_relevant > 0:
        for k_idx, cls in enumerate(ranked_classes):
            rank = k_idx + 1
            if cls == query_class:
                relevant_found += 1
            query_p[k_idx] = relevant_found / rank # Precision at k
            query_r[k_idx] = relevant_found / total_relevant # Recall at k
    
    pr_data = {
        'precision': query_p,
        'recall': query_r,
        'ap': ap_score
    }

    # --- Visualization ---
    num_to_show = min(k + 1, len(all_files)) # +1 for query
    cols = 5
    rows = int(np.ceil(num_to_show / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.suptitle(f'Example Search: {query_basename} [Class {query_class}], Distance: {distance_method}', 
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Display Query
    query_img = cv2.imread(query_path)
    if query_img is not None:
        axes[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'QUERY\nClass: {query_class}', color='red', fontsize=10)
        axes[0].axis('off')
    
    # Display Top K Results
    for idx in range(1, num_to_show):
        if (idx - 1) >= len(distances):
             break
             
        dist, result_path, result_class = distances[idx - 1] 
        result_img = cv2.imread(result_path)
        
        if result_img is not None:
            axes[idx].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            # Color-code results: green for correct class, black for incorrect
            color = 'green' if result_class == query_class else 'black'
            axes[idx].set_title(f'Rank {idx}\nClass: {result_class}\nDist: {dist:.3f}', 
                               color=color, fontsize=9)
            axes[idx].axis('off')
    
    # Hide unused axes
    for idx in range(num_to_show, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return pr_data, query_basename


def run_complete_experiment(Q_values, distance_methods, dataset_folder, descriptor_base_folder):
    """
    Runs the full experiment for RGB histograms.
    
    Iterates over all Q_values and distance_methods, computes descriptors,
    evaluates them, and saves all results.
    """
    results_dir = "experiment_results_1_new"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    # --- Outer loop: Quantization Level (Q) ---
    for Q in Q_values:
        print(f"\n{'='*60}")
        print(f"Processing Q={Q} (bins: {Q**3})")
        print(f"{'='*60}")
        
        # 1. Compute descriptors for this Q value
        params = {'Q': Q}
        descriptor_folder = compute_and_save_global_descriptor(
            'rgb', 
            params, 
            dataset_folder, 
            descriptor_base_folder
        )
        
        # --- Inner loop: Distance Method ---
        for distance_method in distance_methods:
            print(f"\n--- Distance Method: {distance_method} ---")
            
            experiment_name = f"Q{Q}_{distance_method}"
            experiment_dir = os.path.join(results_dir, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # 2. Load descriptors
            all_files, all_feat = load_descriptors(descriptor_folder, dataset_folder)
            
            if len(all_files) == 0:
                print(f"No descriptors found for Q={Q}. Skipping evaluation.")
                continue

            # 3. Run full evaluation
            results = evaluate_system(
                all_files=all_files,
                all_feat=all_feat,
                distance_method=distance_method,
                verbose=False # Suppress per-query progress
            )
            
            all_results[experiment_name] = results
            
            print_evaluation_summary(results)
            
            # 4. Save all plots and reports
            pr_path = os.path.join(experiment_dir, "precision_recall_curve.png")
            plot_precision_recall_curve(
                results, 
                title=f"Average PR Curve (Q={Q}, {distance_method})",
                save_path=pr_path
            )
            
            cm_path = os.path.join(experiment_dir, "confusion_matrix.png")
            plot_confusion_matrix(
                results,
                title=f"Confusion Matrix (Q={Q}, {distance_method})",
                save_path=cm_path
            )
            
            # Save example search visualization
            viz_path = os.path.join(experiment_dir, "example_search.png")
            example_pr_data, query_name = save_example_search_results(
                descriptor_folder,
                dataset_folder,
                distance_method,
                viz_path,
                k=19
            )
            
            # Save PR curve for that single example query
            if example_pr_data and query_name:
                single_pr_path = os.path.join(experiment_dir, "example_query_pr_curve.png")
                plot_single_pr_curve(
                    example_pr_data,
                    query_name,
                    title=f"PR Curve for Single Query: {query_name}\n(Q={Q}, {distance_method})",
                    save_path=single_pr_path
                )
            
            # Save text summary file
            results_file = os.path.join(experiment_dir, "results.txt")
            save_results_to_file(results, results_file)
    
    # 5. Create final comparison summary
    create_comparison_summary(all_results, results_dir)
    
    return all_results, results_dir


def create_comparison_summary(all_results, results_dir):
    """Saves a text file comparing MAP and Top-1 Acc for all experiments."""
    summary_path = os.path.join(results_dir, "comparison_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        # Sort results by mAP (highest first)
        sorted_results = sorted(all_results.items(), 
                              key=lambda x: x[1]['mean_average_precision'], 
                              reverse=True)
        
        f.write(f"{'Experiment':<25} {'MAP':<10} {'Top-1 Acc':<12}\n")
        f.write("-" * 70 + "\n")
        
        # Write one line per experiment
        for exp_name, results in sorted_results:
            map_score = results['mean_average_precision']
            cm = results['confusion_matrix']
            top1_acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
            
            f.write(f"{exp_name:<25} {map_score:<10.4f} {top1_acc:<12.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        if sorted_results:
            f.write(f"Best performing: {sorted_results[0][0]}\n")
            f.write(f"MAP: {sorted_results[0][1]['mean_average_precision']:.4f}\n")
        else:
            f.write("No results to compare.\n")
    
    print(f"\nComparison summary saved to: {summary_path}")


if __name__ == "__main__":
    # --- Configuration ---
    DATASET_FOLDER = '/scratch/zs00774/MSRC_ObjCategImageDatabase_v2'
    DESCRIPTOR_BASE_FOLDER = 'scratch/zs00774/CV/descriptors_1_new'
    
    # Q values to test for RGB histogram
    Q_VALUES = [2,4,6,8,10,12,16,24,32, 40,55,64 ]
    
    # Distance metrics to test
    DISTANCE_METHODS = ['chi_squared']
    
    print("Starting comprehensive experiment for Requirement 1...")
    print(f"Q values: {Q_VALUES}")
    print(f"Distance methods: {DISTANCE_METHODS}")
    print(f"Total experiments: {len(Q_VALUES) * len(DISTANCE_METHODS)}")
    
    # --- Run Experiment Batch ---
    results, results_dir = run_complete_experiment(
        Q_values=Q_VALUES,
        distance_methods=DISTANCE_METHODS,
        dataset_folder=DATASET_FOLDER,
        descriptor_base_folder=DESCRIPTOR_BASE_FOLDER
    )
    
    print(f"\n{'='*60}")
    print(f" All experiments complete!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*60}")
