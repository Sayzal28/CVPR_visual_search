import os
import numpy as np
import scipy.io as sio
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Import functions from other project files ---
from cvpr_evaluation import (
    load_descriptors, 
    evaluate_system,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    save_results_to_file,
    print_evaluation_summary,
    get_class_from_filename
)

def create_pca_summary(all_results, results_dir):
    """Saves a text file comparing MAP and Top-1 Acc for all PCA experiments."""
    summary_path = os.path.join(results_dir, "comparison_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("PCA & MAHALANOBIS EXPERIMENT SUMMARY (REQ 4)\n")
        f.write("=" * 70 + "\n\n")
        
        if not all_results:
            f.write("No results to compare.\n")
            print("No results to compare.")
            return

        # Sort results by mAP (highest first)
        sorted_results = sorted(all_results.items(), 
                              key=lambda x: x[1]['mean_average_precision'], 
                              reverse=True)
        
        f.write(f"{'Experiment':<60} {'MAP':<10} {'Top-1 Acc':<12}\n")
        f.write("-" * 85 + "\n")
        
        for exp_name, results in sorted_results:
            map_score = results['mean_average_precision']
            cm = results['confusion_matrix']
            top1_acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
            f.write(f"{exp_name:<60} {map_score:<10.4f} {top1_acc:<12.4f}\n")
        
        f.write("\n" + "=" * 85 + "\n")
        if sorted_results:
            f.write(f"Best performing: {sorted_results[0][0]}\n")
            f.write(f"MAP: {sorted_results[0][1]['mean_average_precision']:.4f}\n")
        else:
            f.write("No results found to compare.\n")
    
    print(f"\nComparison summary saved to: {summary_path}")

def run_pca_experiments(descriptor_folder, pca_components, dataset_folder, results_dir):
    """
    Main function to run PCA and Mahalanobis distance experiments.
    
    This will:
    1. Load a base descriptor set.
    2. Scale the data (StandardScaler).
    3. Loop through each 'k' in pca_components:
        a. Fit PCA and transform data to 'k' dimensions.
        b. Calculate and save the inverse covariance matrix for the k-dim data.
        c. Evaluate using Euclidean distance.
        d. Evaluate using Mahalanobis distance.
    4. Save all results and a final summary.
    """
    
    print(f"--- Running Requirement 4 Experiments ---")
    print(f"Loading base descriptors from: {descriptor_folder}")
    
    all_files, all_feat = load_descriptors(descriptor_folder, dataset_folder)
    
    if not all_files:
        print("Error: No descriptors loaded. Exiting.")
        return {}

    descriptor_name = os.path.basename(descriptor_folder)
    all_results = {}

    # --- 1. SCALE DATA ---
    # PCA is sensitive to feature scale, so scaling is crucial.
    print("Scaling data (mean=0, std=1)...")
    scaler = StandardScaler()
    all_feat_scaled = scaler.fit_transform(all_feat)
    print(f"Data scaled. Shape: {all_feat_scaled.shape}")
    
    # Determine max possible components
    max_k_possible = min(all_feat_scaled.shape[0] - 1, all_feat_scaled.shape[1])

    # --- 2. RUN PCA PROJECTION EXPERIMENTS ---
    for k in pca_components:
        
        print("\n" + "="*60)
        print(f"Processing for PCA Projection (k={k})...")
        
        if k > max_k_possible:
            print(f"\nSkipping k={k}, as it's larger than max valid dimension ({max_k_possible})")
            continue
            
        # --- A. Fit PCA and transform data ---
        print(f"Fitting NEW PCA (n_components={k})...")
        start_time = time.time()
        pca = PCA(n_components=k)
        all_feat_pca_k = pca.fit_transform(all_feat_scaled)
        print(f"PCA complete in {time.time() - start_time:.2f}s")
        print(f"New descriptor shape: {all_feat_pca_k.shape}")

        # --- B. Calculate Full Inverse Covariance Matrix (C_inv) ---
        print(f"Calculating full {k}x{k} covariance matrix 'C'...")
        cov_matrix = np.cov(all_feat_pca_k, rowvar=False)
        
        if k == min(pca_components):
            print(f"\n--- Debug: Top 5x5 corner of {k}x{k} Covariance Matrix ---")
            cov_matrix_k_thresh = cov_matrix.copy()
            cov_matrix_k_thresh[np.abs(cov_matrix_k_thresh) < 1e-10] = 0.0
            print(cov_matrix_k_thresh[:5, :5])
            print("--- End Debug ---")

        print(f"Calculating full {k}x{k} inverse covariance 'C_inv'...")
        # Use pseudo-inverse for numerical stability
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        
        # Save the C_inv matrix to be loaded by cvpr_compare
        inv_cov_path = os.path.join(results_dir, f'inv_cov_matrix_k{k}.mat')
        sio.savemat(inv_cov_path, {'inv_cov': inv_cov_matrix})
        
        
        # --- C. Evaluate (Experiment 1: Euclidean on PCA features) ---
        print(f"Evaluating PCA-k{k} with Euclidean distance...")
        exp_name = f"PCA_k{k}_Euclidean_on_{descriptor_name}"
        exp_dir = os.path.join(results_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Evaluate using the new k-dimensional features
        results_euclidean = evaluate_system(all_files, all_feat_pca_k, distance_method='euclidean')
        all_results[exp_name] = results_euclidean
        
        print_evaluation_summary(results_euclidean)
        plot_precision_recall_curve(results_euclidean, title=f"PR Curve - {exp_name}", save_path=os.path.join(exp_dir, "pr_curve.png"))
        plot_confusion_matrix(results_euclidean, title=f"CM - {exp_name}", save_path=os.path.join(exp_dir, "cm.png"))
        
        
        # --- D. Evaluate (Experiment 2: Mahalanobis on PCA features) ---
        print(f"Evaluating PCA-k{k} with Mahalanobis_FULL_PCA distance...")
        exp_name = f"PCA_k{k}_Mahalanobis_on_{descriptor_name}"
        exp_dir = os.path.join(results_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Evaluate using the new k-dim features and the corresponding C_inv
        results_mahalanobis = evaluate_system(
            all_files, 
            all_feat_pca_k, 
            distance_method='mahalanobis_full_pca',
            # Pass the path to the C_inv file using the 'verbose' dict
            verbose={'inv_cov_path': inv_cov_path} 
        )
        all_results[exp_name] = results_mahalanobis
        
        print_evaluation_summary(results_mahalanobis)
        plot_precision_recall_curve(results_mahalanobis, title=f"PR Curve - {exp_name}", save_path=os.path.join(exp_dir, "pr_curve.png"))
        plot_confusion_matrix(results_mahalanobis, title=f"CM - {exp_name}", save_path=os.path.join(exp_dir, "cm.png"))
        
        # Save a combined text file for this 'k'
        results_file_path = os.path.join(results_dir, f"PCA_k{k}_results.txt")
        with open(results_file_path, 'w') as f:
            f.write(f"--- Euclidean Results (k={k}) ---\n")
            f.write(f"MAP: {results_euclidean['mean_average_precision']:.4f}\n\n")
            f.write(f"--- Mahalanobis Results (k={k}) ---\n")
            f.write(f"MAP: {results_mahalanobis['mean_average_precision']:.4f}\n")
            
    return all_results


if __name__ == "__main__":
    
    # --- 1. Configuration ---
    DATASET_FOLDER = '/scratch/zs00774/MSRC_ObjCategImageDatabase_v2'
    
    # Set the path to the high-dimensional descriptor set you want to reduce
    DESCRIPTOR_TO_TEST = '/scratch/zs00774/CV/descriptors_3/spatial_grid_8x8_color_hist_texture_Q24_T16'
    
    RESULTS_DIR = "experiment_results_req4_pca"
    
    # --- 2. Define PCA dimensions (k) to test ---
    PCA_COMPONENTS = [2,4,8,16, 32, 64, 128, 256, 512]
    
    # --- 3. Run All Experiments ---
    print("Starting Requirement 4 (PCA & Mahalanobis) Experiments...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Validate paths before starting
    if not os.path.isdir(DATASET_FOLDER):
        print(f"Error: Dataset folder not found, exiting.")
        print(f"Path not found: {DATASET_FOLDER}")
    elif not os.path.isdir(DESCRIPTOR_TO_TEST):
        print(f"Error: Descriptor folder to test not found, exiting.")
        print(f"Path not found: {DESCRIPTOR_TO_TEST}")
    else:
        all_results = run_pca_experiments(
            descriptor_folder=DESCRIPTOR_TO_TEST,
            pca_components=PCA_COMPONENTS,
            dataset_folder=DATASET_FOLDER,
            results_dir=RESULTS_DIR
        )
        
        # --- 4. CREATE FINAL SUMMARY ---
        print(f"\n{'='*60}")
        print(f" All PCA/Mahalanobis experiments complete!")
        create_pca_summary(all_results, RESULTS_DIR)
        print(f"All results saved in: {RESULTS_DIR}")
        print(f"{'='*60}")
