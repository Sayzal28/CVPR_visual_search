import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import shutil

# --- Import functions from other project files ---
from cvpr_evaluation import load_descriptors, get_class_from_filename
try:
    # Attempt to import a progress bar utility
    from cvpr_evaluation import print_progress_bar
except ImportError:
    # Define a fallback function if it doesn't exist
    def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
        pass


def plot_svm_confusion_matrix(y_true, y_pred, class_labels, save_path, title):
    """
    Generates and saves a normalized confusion matrix plot using seaborn.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    # Normalize by row (true class) to get per-class accuracy
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Handle cases where a class had no samples (NaN)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Create the heatmap plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(title, fontsize=14)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_svm_gridsearch_experiment(exp_config, dataset_folder, results_dir):
    """
    Runs a single SVM experiment using GridSearchCV for hyperparameter tuning.
    
    Args:
        exp_config (dict): Configuration for the experiment.
        dataset_folder (str): Path to the MSRC dataset root.
        results_dir (str): Base directory to save results.

    Returns:
        dict: A summary of the experiment results.
    """
    # --- 0. Setup ---
    exp_name = exp_config['name']
    descriptor_folder = exp_config['descriptor_folder']
    param_grid = exp_config['param_grid']
    estimator_params = exp_config.get('estimator_params', {})

    print(f"\n{'='*60}")
    print(f"Running Experiment: {exp_name}")
    print(f"Descriptor: {os.path.basename(descriptor_folder)}")
    print(f"Parameter Grid: {param_grid}")
    print(f"Base Estimator Params: {estimator_params}")
    print(f"{'='*60}")

    # Create a dedicated directory for this experiment's results
    experiment_dir = os.path.join(results_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # --- 1. Load Data ---
    all_files, all_feat = load_descriptors(descriptor_folder, dataset_folder)
    if not all_files:
        print("Error: No descriptors loaded. Skipping experiment.")
        return {'name': exp_name, 'accuracy': 0, 'train_time': 0, 'error': 'No descriptors found'}

    all_class_labels = [get_class_from_filename(f) for f in all_files]
    unique_labels = sorted(list(set(all_class_labels)))
    X = all_feat
    y = np.array(all_class_labels)

    # --- 2. Split Data ---
    # Stratify by 'y' to ensure balanced class representation in train/test
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X, y, np.array(all_files), test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Scale Features ---
    # Fit the scaler ONLY on the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Apply the same scaling transformation to the test data
    X_test_scaled = scaler.transform(X_test)

    print("Data loaded, split, and scaled.")

    # --- 4. Train with GridSearchCV ---
    print(f"Training SVM with GridSearchCV... (This may take a while)")
    start_time = time.time()

    # Apply base parameters (e.g., class_weight)
    base_params = {'random_state': 42}
    base_params.update(estimator_params)
    base_estimator = SVC(**base_params)

    # Configure the cross-validation grid search
    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        cv=5,       # 5-fold cross-validation
        verbose=2,  # Show progress
        n_jobs=4    # Use 4 parallel cores
    )
    grid_search.fit(X_train_scaled, y_train)

    train_time = time.time() - start_time
    print(f"Training complete in {train_time:.2f} seconds.")

    # --- 5. Evaluate Best Model ---
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Predict on the held-out test set
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=unique_labels, zero_division=0)

    print(f"\nBest parameters: {best_params}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Save text report
    report_path = os.path.join(experiment_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Descriptor: {descriptor_folder}\n\n")
        f.write(f"--- PARAMETER GRID TESTED ---\n{param_grid}\n\n")
        f.write(f"--- BEST PARAMS FOUND ---\n{best_params}\n\n")
        f.write(f"Training Time: {train_time:.2f}s\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("--- Classification Report ---\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")

    # Save confusion matrix plot
    cm_title = f"SVM Confusion Matrix - {exp_name}\n(Accuracy: {accuracy:.3f})"
    cm_path = os.path.join(experiment_dir, "confusion_matrix.png")
    plot_svm_confusion_matrix(y_test, y_pred, unique_labels, cm_path, cm_title)
    print(f"Confusion matrix saved to: {cm_path}")

    # --- 6. Save Sorted Test Images ---
    # Copy test images into folders named by their *predicted* class
    sorted_images_base_dir = os.path.join(experiment_dir, "sorted_test_images")
    if os.path.exists(sorted_images_base_dir):
        shutil.rmtree(sorted_images_base_dir) # Clear previous results
    os.makedirs(sorted_images_base_dir, exist_ok=True)

    num_test_images = len(files_test)
    print_progress_bar(0, num_test_images, prefix='Sorting images:', suffix='Complete', length=50)

    for i in range(num_test_images):
        original_path = files_test[i]
        true_class = y_test[i]
        predicted_class = y_pred[i]

        # Create folder for the predicted class
        predicted_dir = os.path.join(sorted_images_base_dir, f"Predicted_Class_{predicted_class}")
        os.makedirs(predicted_dir, exist_ok=True)
        
        # Filename includes the *true* class for easy verification
        new_filename = f"True_Class_{true_class}__{os.path.basename(original_path)}"
        destination_path = os.path.join(predicted_dir, new_filename)
        try:
            shutil.copy(original_path, destination_path)
        except Exception as e:
            print(f"Warning: Could not copy file {original_path}. Error: {e}")

        # Update progress bar
        if (i + 1) % 50 == 0 or (i + 1) == num_test_images:
            print_progress_bar(i + 1, num_test_images, prefix='Sorting images:', suffix='Complete', length=50)

    # Return summary for comparison
    return {'name': exp_name, 'accuracy': accuracy, 'train_time': train_time, 'best_params': best_params}


def create_svm_summary(all_results, results_dir):
    """
    Creates a single text file summarizing all SVM experiments,
    sorted by accuracy.
    """
    summary_path = os.path.join(results_dir, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("SVM EXPERIMENT COMPARISON SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        if not all_results:
            f.write("No results to compare.\n")
            return
            
        # Sort results by accuracy (highest first)
        sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
        
        # Write table header
        f.write(f"{'Experiment':<35} {'Accuracy':<12} {'Train Time (s)':<15} {'Best Params'}\n")
        f.write("-" * 100 + "\n")
        
        # Write one line per experiment
        for res in sorted_results:
            f.write(f"{res['name']:<35} {res['accuracy']:<12.4f} {res['train_time']:<15.2f} {res.get('best_params', 'N/A')}\n")
            
        # Write overall best
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Best performing: {sorted_results[0]['name']}\n")
        f.write(f"Best Accuracy: {sorted_results[0]['accuracy']:.4f}\n")
        f.write(f"Best Params: {sorted_results[0].get('best_params', 'N/A')}\n")
        
    print(f"\nComparison summary saved to: {summary_path}")


if __name__ == "__main__":

    # --- 1. Configuration ---
    DATASET_FOLDER = '/scratch/zs00774/MSRC_ObjCategImageDatabase_v2'
    DESCRIPTOR_FOLDER_TO_TEST = '//scratch/zs00774/CV/descriptors_3/spatial_grid_8x8_color_hist_texture_Q24_T16'
    RESULTS_DIR = "experiment_results_svm_optimized_13k"

    # --- 2. Hyperparameter Grids ---
    C_VALUES = [0.01, 0.1, 1, 10, 100, 1000]
    GAMMA_VALUES = [1e-4, 1e-5, 1e-6, 'scale']

    # --- 3. Experiment Definitions ---
    EXPERIMENTS_TO_RUN = [
        # Linear SVM
        {
            'name': '1_LinearKernel',
            'descriptor_folder': DESCRIPTOR_FOLDER_TO_TEST,
            'param_grid': { 'C': C_VALUES, 'kernel': ['linear'] }
        },
        # Linear SVM with balanced class weights
        {
            'name': '2_LinearKernel_Balanced',
            'descriptor_folder': DESCRIPTOR_FOLDER_TO_TEST,
            'estimator_params': { 'class_weight': 'balanced' },
            'param_grid': { 'C': C_VALUES, 'kernel': ['linear'] }
        },

        # RBF Kernel SVM
        {
            'name': '3_RBF_HighDim',
            'descriptor_folder': DESCRIPTOR_FOLDER_TO_TEST,
            'param_grid': { 'C': C_VALUES, 'kernel': ['rbf'], 'gamma': GAMMA_VALUES }
        },
        # RBF Kernel SVM with balanced class weights
        {
            'name': '4_RBF_HighDim_Balanced',
            'descriptor_folder': DESCRIPTOR_FOLDER_TO_TEST,
            'estimator_params': { 'class_weight': 'balanced' },
            'param_grid': { 'C': C_VALUES, 'kernel': ['rbf'], 'gamma': GAMMA_VALUES }
        },
    ]

    # --- 4. Run Experiments ---
    print("Starting Optimized SVM Experiments (High-Dimensional Descriptors)...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    # Validate paths before starting
    if not os.path.isdir(DATASET_FOLDER):
        print(f"Error: Dataset folder not found: {DATASET_FOLDER}")
    elif not os.path.isdir(DESCRIPTOR_FOLDER_TO_TEST):
        print(f"Error: Descriptor folder not found: {DESCRIPTOR_FOLDER_TO_TEST}")
    else:
        # Loop through and run each defined experiment
        for exp in EXPERIMENTS_TO_RUN:
            result = run_svm_gridsearch_experiment(exp, DATASET_FOLDER, RESULTS_DIR)
            all_results.append(result)

        # Write the final summary file
        print("\n All optimized experiments complete.")
        create_svm_summary(all_results, RESULTS_DIR)
        print(f"All results saved in: {RESULTS_DIR}")
