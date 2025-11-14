import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from cvpr_compare import cvpr_compare

def get_class_from_filename(filename):
    """Extracts class ID (e.g., '2') from a filename."""
    basename = os.path.basename(filename)
    parts = basename.replace('.bmp', '').replace('.mat', '').split('_')
    try:
        # The class ID is the first part of the filename
        return int(parts[0])
    except:
        return -1  # Return -1 for unknown or invalid class formats

def load_descriptors(descriptor_folder, dataset_folder):
    """Loads all .mat descriptors from a folder."""
    all_files = []
    all_feat = []
    
    if not os.path.isdir(descriptor_folder):
        print(f"Error: Descriptor folder not found: {descriptor_folder}")
        return [], []
        
    mat_files = sorted([f for f in os.listdir(descriptor_folder) if f.endswith('.mat')])
    
    n_files = len(mat_files)
    if n_files == 0:
        print(f"Warning: No .mat files found in {descriptor_folder}")
        return [], []
        
    for i, filename in enumerate(mat_files):
        mat_path = os.path.join(descriptor_folder, filename)
        # Construct the path to the corresponding .bmp image
        img_path = os.path.join(dataset_folder, 'Images', filename.replace('.mat', '.bmp'))
        
        mat_data = sio.loadmat(mat_path)
        # Ensure the .mat file is valid
        if 'F' in mat_data and isinstance(mat_data['F'], np.ndarray):
            all_files.append(img_path)
            all_feat.append(mat_data['F'].flatten())
    
    return all_files, np.array(all_feat)


def evaluate_system(all_files, all_feat, distance_method='euclidean', verbose=True):
    """
    Evaluates a descriptor by using every image as a query.
    
    Calculates mAP, P/R curves, and confusion matrix.
    
    Args:
        all_files (list): List of image file paths.
        all_feat (np.array): Array of corresponding feature vectors.
        distance_method (str): Distance metric for cvpr_compare.
        verbose (bool or dict): If True, prints progress.
                                If dict, passes args to cvpr_compare (for Mahalanobis).
    """
    all_class_labels = [get_class_from_filename(f) for f in all_files]
    nimg = len(all_files)
    unique_classes = sorted(list(set(all_class_labels)))
    
    # Check if 'verbose' is a boolean or a dict of extra args
    is_verbose_bool = isinstance(verbose, bool)
    
    if is_verbose_bool and verbose:
        print(f"Evaluating {nimg} images from {len(unique_classes)} classes...")
        print(f"Using distance method: {distance_method}\n")
    
    # Result storage
    all_average_precisions = []
    query_pr_curves_p = []
    query_pr_curves_r = []
    all_true_labels = []
    all_predicted_labels = []
    
    # --- Main Evaluation Loop ---
    # Use every image as a query
    for query_idx in range(nimg):
        
        if is_verbose_bool and verbose and (query_idx + 1) % 50 == 0:
            print(f"Processing query {query_idx + 1}/{nimg}...")
        
        query_class = all_class_labels[query_idx]
        query_feat = all_feat[query_idx]
        
        # Total relevant items (excluding the query itself)
        total_relevant = all_class_labels.count(query_class) - 1
        
        # --- Calculate distances to all other images ---
        distances = []
        for i in range(nimg):
            if i == query_idx:
                continue  # Skip self-comparison
            if all_feat[i].shape == query_feat.shape:
                
                # --- Handle distance calculation ---
                # This 'if' block handles the "trick" of passing extra paths
                # for Mahalanobis distances via the 'verbose' argument.
                if not is_verbose_bool:
                    if 'variance_path' in verbose:
                        dist = cvpr_compare(query_feat, all_feat[i], method=distance_method, variance_path=verbose['variance_path'])
                    elif 'inv_cov_path' in verbose:
                        dist = cvpr_compare(query_feat, all_feat[i], method=distance_method, inv_cov_path=verbose['inv_cov_path'])
                    else:
                        # Fallback if verbose is dict but no paths
                        dist = cvpr_compare(query_feat, all_feat[i], method=distance_method)
                else:
                    # Standard distance call
                    dist = cvpr_compare(query_feat, all_feat[i], method=distance_method)
                
                distances.append((dist, i))
        
        # Sort results by distance
        distances.sort(key=lambda x: x[0])
        
        # --- Calculate metrics for this query ---
        ranked_indices = [idx for dist, idx in distances]
        ranked_dists = [dist for dist, idx in distances]
        retrieved_classes = [all_class_labels[idx] for idx in ranked_indices]
        
        # Binary labels (1=relevant, 0=irrelevant) for AP calculation
        y_true_binary = [1 if cls == query_class else 0 for cls in retrieved_classes]
        y_scores = [-d for d in ranked_dists]  # Scores = negative distance
        
        ap = average_precision_score(y_true_binary, y_scores) if any(y_true_binary) else 0.0
        all_average_precisions.append(ap)
        
        # --- Calculate Precision-Recall curve for this query ---
        query_p = np.zeros(nimg - 1)
        query_r = np.zeros(nimg - 1)
        relevant_found = 0
        
        if total_relevant > 0:
            for k_idx, cls in enumerate(retrieved_classes):
                rank = k_idx + 1
                if cls == query_class:
                    relevant_found += 1
                query_p[k_idx] = relevant_found / rank  # Precision at k
                query_r[k_idx] = relevant_found / total_relevant  # Recall at k
        
        query_pr_curves_p.append(query_p)
        query_pr_curves_r.append(query_r)

        # --- For confusion matrix and Top-1 accuracy ---
        all_true_labels.append(query_class)
        # Get the class of the top-ranked result (Rank 1)
        predicted_class = all_class_labels[ranked_indices[0]] if ranked_indices else -1
        all_predicted_labels.append(predicted_class)
        
    # --- Aggregate results ---
    
    # Mean Average Precision (mAP)
    map_score = np.mean(all_average_precisions) if all_average_precisions else 0.0
    
    # Average P/R curve over all queries
    avg_precision_full = np.mean(query_pr_curves_p, axis=0) if query_pr_curves_p else None
    avg_recall_full = np.mean(query_pr_curves_r, axis=0) if query_pr_curves_r else None
    
    # Build confusion matrix
    confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)))
    for true_label, pred_label in zip(all_true_labels, all_predicted_labels):
        if true_label in unique_classes and pred_label in unique_classes:
            true_idx = unique_classes.index(true_label)
            pred_idx = unique_classes.index(pred_label)
            confusion_matrix[true_idx, pred_idx] += 1
    
    if is_verbose_bool and verbose:
        print(f"\n{'='*50}")
        print(f"Mean Average Precision (MAP): {map_score:.4f}")
        print(f"{'='*50}\n")
    
    # Return all computed metrics
    return {
        'mean_average_precision': map_score,
        'all_aps': all_average_precisions,
        'avg_precision_full': avg_precision_full,
        'avg_recall_full': avg_recall_full,
        'all_precisions': query_pr_curves_p,
        'all_recalls': query_pr_curves_r,
        'confusion_matrix': confusion_matrix,
        'true_labels': all_true_labels,
        'predicted_labels': all_predicted_labels,
        'all_class_labels': unique_classes,
    }


def plot_precision_recall_curve(results, title='Precision-Recall Curve', save_path='pr_curve.png'):
    """Plots the average Precision-Recall curve."""
    plt.figure(figsize=(10, 7))
    
    plt.plot(results['avg_recall_full'], results['avg_precision_full'], 
             'b-', linewidth=3, label='Average PR Curve')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f"{title}\n(MAP: {results['mean_average_precision']:.3f})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_single_pr_curve(pr_data, query_basename, title='PR Curve for Single Query', save_path='single_pr_curve.png'):
    """Plots the Precision-Recall curve for a single query."""
    plt.figure(figsize=(10, 7))
    
    ap_score = pr_data.get('ap', 0.0)
    
    plt.plot(pr_data['recall'], pr_data['precision'], 
             'r-', linewidth=2, 
             label=f"PR for {query_basename} (AP: {ap_score:.3f})")
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(results, title='Confusion Matrix', save_path='confusion_matrix.png'):
    """Plots a normalized confusion matrix."""
    confusion_matrix = results['confusion_matrix']
    class_labels = results['all_class_labels']
    
    # Normalize the matrix by rows (per-class accuracy)
    cm_normalized = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle classes with 0 samples
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='Accuracy')
    
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title(title, fontsize=14)
    
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    
    # Add text annotations if classes are few
    if len(class_labels) <= 20:
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
                plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                        ha='center', va='center', color=color, fontsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_results_to_file(results, file_path='results.txt'):
    """Saves numerical results (mAP, per-class AP, Top-1) to a text file."""
    with open(file_path, 'w') as f:
        f.write("VISUAL SEARCH EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Mean Average Precision (MAP): {results['mean_average_precision']:.4f}\n\n")
        
        # Calculate per-class AP
        class_aps = {}
        for true_label, ap in zip(results['true_labels'], results['all_aps']):
            if true_label not in class_aps:
                class_aps[true_label] = []
            class_aps[true_label].append(ap)
        
        f.write("Average Precision per Class:\n")
        for class_id in sorted(class_aps.keys()):
            mean_ap = np.mean(class_aps[class_id])
            f.write(f"  Class {class_id}: {mean_ap:.4f} (from {len(class_aps[class_id])} queries)\n")
        
        # Calculate Top-1 Accuracy from confusion matrix
        cm = results['confusion_matrix']
        top1_acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        f.write(f"\nTop-1 Classification Accuracy: {top1_acc:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Confusion Matrix (counts):\n")
        f.write(str(results['confusion_matrix'].astype(int)))

def print_evaluation_summary(results):
    """Prints a brief evaluation summary to the console."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Mean Average Precision (MAP): {results['mean_average_precision']:.4f}")
    
    cm = results['confusion_matrix']
    top1_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    print(f"Top-1 Classification Accuracy: {top1_accuracy:.4f}")
    
    if 'all_aps' in results and 'all_class_labels' in results:
        # Calculate and print best/worst class AP
        class_aps = {}
        for true_label, ap in zip(results['true_labels'], results['all_aps']):
            if true_label not in class_aps: class_aps[true_label] = []
            class_aps[true_label].append(ap)
        
        mean_class_aps = [np.mean(class_aps[cid]) for cid in class_aps if class_aps[cid]]
        if mean_class_aps:
            print(f"\nBest class mean AP: {max(mean_class_aps):.4f}")
            print(f"Worst class mean AP: {min(mean_class_aps):.4f}")
    
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # --- Example Usage ---
    DESCRIPTOR_FOLDER = 'descriptors_1/globalRGBhisto_8'
    DATASET_FOLDER = 'path/to/MSRC_dataset'
    DISTANCE_METHOD = 'euclidean'
    OUTPUT_FOLDER = 'results'
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("Loading descriptors...")
    all_files, all_feat = load_descriptors(DESCRIPTOR_FOLDER, DATASET_FOLDER)
    
    if all_files:
        print("Evaluating system...")
        results = evaluate_system(all_files, all_feat, DISTANCE_METHOD, verbose=True)
        
        # Save plots
        plot_precision_recall_curve(
            results, 
            save_path=os.path.join(OUTPUT_FOLDER, 'pr_curve.png')
        )
        
        plot_confusion_matrix(results, save_path=os.path.join(OUTPUT_FOLDER, 'confusion_matrix.png'))
        
        # Save text summary
        results_file_path = os.path.join(OUTPUT_FOLDER, 'results.txt')
        save_results_to_file(results, results_file_path)
        print(f"Saved: {results_file_path}")
        
        # Print summary to console
        print_evaluation_summary(results)
    else:
        print("No descriptors found to evaluate.")
