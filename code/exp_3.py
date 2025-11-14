import os
import numpy as np
import cv2
from random import randint
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import scipy.io as sio

# --- Import functions from other project files ---
from extractRandom import generate_spatial_grid_feature 
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
    """
    
    all_files, all_feat = load_descriptors(descriptor_folder, dataset_folder)
    
    if (all_files is None) or (len(all_files) == 0):
        print(f"Warning: No descriptors found in {descriptor_folder}. Skipping example search.")
        return None, None

    query_idx = 100 # Use a consistent query for comparison
    if query_idx >= len(all_files):
        query_idx = 0 # Fallback
    
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
            continue
        dist = cvpr_compare(query_feat, all_feat[i], method=distance_method)
        distances.append((dist, all_files[i], all_class_labels[i])) 
    
    distances.sort(key=lambda x: x[0])
    
    # --- Single Query PR Calculation ---
    ranked_classes = [cls for dist, path, cls in distances]
    ranked_dists = [dist for dist, path, cls in distances]

    y_true_binary = [1 if cls == query_class else 0 for cls in ranked_classes]
    y_scores = [-d for d in ranked_dists]
    
    ap_score = average_precision_score(y_true_binary, y_scores) if any(y_true_binary) else 0.0
    
    query_p = np.zeros(len(distances))
    query_r = np.zeros(len(distances))
    relevant_found = 0
    
    if total_relevant > 0:
        for k_idx, cls in enumerate(ranked_classes):
            rank = k_idx + 1
            if cls == query_class:
                relevant_found += 1
            query_p[k_idx] = relevant_found / rank
            query_r[k_idx] = relevant_found / total_relevant
    
    pr_data = {'precision': query_p, 'recall': query_r, 'ap': ap_score}
    
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
        if (idx - 1) >= len(distances): break
        dist, result_path, result_class = distances[idx - 1] 
        result_img = cv2.imread(result_path)
        if result_img is not None:
            axes[idx].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            color = 'green' if result_class == query_class else 'black'
            axes[idx].set_title(f'Rank {idx}\nClass: {result_class}\nDist: {dist:.3f}', 
                               color=color, fontsize=9)
            axes[idx].axis('off')
    
    # Hide unused axes
    for idx in range(num_to_show, len(axes)): axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return pr_data, query_basename

def create_comparison_summary(all_results, results_dir):
    """Saves a text file comparing MAP and Top-1 Acc for all experiments."""
    summary_path = os.path.join(results_dir, "comparison_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        if not all_results:
            f.write("No results to compare.\n")
            print("No results to compare.")
            return

        # Sort results by mAP (highest first)
        sorted_results = sorted(all_results.items(), 
                              key=lambda x: x[1]['mean_average_precision'], 
                              reverse=True)
        
        f.write(f"{'Experiment':<75} {'MAP':<10} {'Top-1 Acc':<12}\n")
        f.write("-" * 100 + "\n")
        
        for exp_name, results in sorted_results:
            map_score = results['mean_average_precision']
            cm = results['confusion_matrix']
            top1_acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
            f.write(f"{exp_name:<75} {map_score:<10.4f} {top1_acc:<12.4f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        if sorted_results:
            f.write(f"Best performing: {sorted_results[0][0]}\n")
            f.write(f"MAP: {sorted_results[0][1]['mean_average_precision']:.4f}\n")
        else:
            f.write("No results to compare.\n")

    print(f"\nComparison summary saved to: {summary_path}")

def run_evaluation_for_distance_metric(
    distance_method, 
    best_q, 
    t_values, 
    grid_rows, 
    grid_cols, 
    dataset_folder, 
    descriptor_base_folder, 
    results_base_dir, 
    all_results
    ):
    """
    Generates and evaluates all spatial grid descriptors for ONE distance method.
    
    This function will:
    1. Define all spatial descriptor combinations.
    2. Generate descriptors (if they don't exist).
    3. Evaluate each descriptor with the specified distance_method.
    4. Save all results and add to the all_results dict.
    """

    print(f"\n{'='*70}")
    print(f"Processing for Distance Method: {distance_method.upper()} (using Q={best_q})")
    print(f"{'='*70}")

    # --- Define all experiments to run for this distance method ---
    experiments_to_run = []
    
    for T in t_values:
        # Combined Color Histogram + Texture (EOH)
        experiments_to_run.append({
            'features': ['color_hist', 'texture'],
            'color_Q': 24, # Using the 'best_q' provided
            'texture_bins': T
        })
        
    # Get list of images to process
    image_dir = os.path.join(dataset_folder, 'Images')
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".bmp")]

    # --- Loop through each experiment, generate descriptors, and evaluate ---
    for exp_params in experiments_to_run:
        
        # --- 1. GENERATE DESCRIPTORS (if needed) ---
        
        # Build a unique folder name for this descriptor set
        feature_str = "_".join(sorted(exp_params['features']))
        folder_name = f"spatial_grid_{grid_rows}x{grid_cols}_{feature_str}"
        if 'color_hist' in exp_params['features']:
            folder_name += f"_Q{exp_params['color_Q']}"
        if 'texture' in exp_params['features']:
            folder_name += f"_T{exp_params['texture_bins']}"
        
        descriptor_folder = os.path.join(descriptor_base_folder, folder_name)
        
        # Check if descriptors already exist
        if not os.path.isdir(descriptor_folder) or len(os.listdir(descriptor_folder)) < len(image_files):
            os.makedirs(descriptor_folder, exist_ok=True)
            print(f"\nGenerating Descriptors: {folder_name}")
            
            # Loop and create descriptor for each image
            for i, filename in enumerate(image_files):
                if (i+1) % 100 == 0:
                    print(f"... processing image {i+1}/{len(image_files)}")
                
                img_path = os.path.join(image_dir, filename)
                img = cv2.imread(img_path)
                if img is None: continue
                
                img_normalized = img.astype(np.float64) / 255.0
                fout = os.path.join(descriptor_folder, filename.replace('.bmp', '.mat'))

                # Call the spatial grid feature generator
                F = generate_spatial_grid_feature(
                    img_normalized,
                    grid_rows=grid_rows,
                    grid_cols=grid_cols,
                    feature_types=exp_params['features'],
                    color_Q=exp_params['color_Q'],
                    texture_bins=exp_params['texture_bins']
                )
                sio.savemat(fout, {'F': F})
            print(f"Finished generating: {folder_name}")
        else:
            print(f"\nDescriptors already exist: {folder_name}")

        
        # --- 2. EVALUATE DESCRIPTORS ---
        
        # Create a unique name for this specific evaluation run
        experiment_name = f"{folder_name} (Dist: {distance_method})"
        
        print(f"--- Evaluating: {experiment_name} ---")
        
        # Create a unique output directory for this evaluation's results
        eval_dir_name = f"EVAL_{distance_method.upper()}"
        experiment_dir = os.path.join(results_base_dir, eval_dir_name, folder_name)
        os.makedirs(experiment_dir, exist_ok=True)
            
        all_files, all_feat = load_descriptors(descriptor_folder, dataset_folder)
        if (all_files is None) or (len(all_files) == 0):
            print(f"Error: No descriptors found in {descriptor_folder}. Skipping.")
            continue

        # Run the full system evaluation
        results = evaluate_system(
            all_files=all_files,
            all_feat=all_feat,
            distance_method=distance_method,
            verbose=False # Suppress per-query output
        )
        
        all_results[experiment_name] = results
        print_evaluation_summary(results)
        
        # --- 3. Save all plots and reports ---
        pr_path = os.path.join(experiment_dir, "precision_recall_curve.png")
        plot_precision_recall_curve(
            results, 
            title=f"Average PR Curve\n({experiment_name})",
            save_path=pr_path
        )
        
        cm_path = os.path.join(experiment_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            results,
            title=f"Confusion Matrix\n({experiment_name})",
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
                title=f"PR Curve for Single Query: {query_name}\n({experiment_name})",
                save_path=single_pr_path
            )
        
        # Save text summary file
        results_file = os.path.join(experiment_dir, "results.txt")
        save_results_to_file(results, results_file)


if __name__ == "__main__":
    # --- Configuration ---
    DATASET_FOLDER = '/scratch/zs00774/MSRC_ObjCategImageDatabase_v2'
    DESCRIPTOR_BASE_FOLDER = '/scratch/zs00774/CV/descriptors_3'
    RESULTS_DIR = "experiment_results_req3_24"
    
    # --- Experiment Parameters ---
    GRID_ROWS = 8
    GRID_COLS = 8
    T_VALUES = [4, 8, 16, 32] # EOH texture bins to test

    # Map distance metric to the best Q value found in experiment 1
    BEST_Q_FOR_METRIC = {
        'chi_squared': 24
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}
    
    print("Starting Requirement 3 Experiments (Orderly Fashion)...")
    
    # --- Main Loop ---
    # Run the full set of spatial grid experiments for each distance method
    for method, q_value in BEST_Q_FOR_METRIC.items():
        run_evaluation_for_distance_metric(
            distance_method=method,
            best_q=q_value,
            t_values=T_VALUES,
            grid_rows=GRID_ROWS,
            grid_cols=GRID_COLS,
            dataset_folder=DATASET_FOLDER,
            descriptor_base_folder=DESCRIPTOR_BASE_FOLDER,
            results_base_dir=RESULTS_DIR,
            all_results=all_results
        )

    # --- Final Summary ---
    print(f"\n{'='*70}")
    print(f" All Requirement 3 experiments complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("Generating final comparison summary...")
    create_comparison_summary(all_results, RESULTS_DIR)
    print(f"{'='*70}")
