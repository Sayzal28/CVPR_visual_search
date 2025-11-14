Computer Vision Project Readme

FILE DESCRIPTIONS

Core Libraries

extractRandom.py: Contains all feature extraction functions (RGB, HSV, LBP, EOH, Spatial Grid).

cvpr_compare.py: Calculates the distance between two feature vectors using metrics like 'euclidean', 'chi_squared', and 'mahalanobis_full_pca'.

cvpr_evaluation.py: Runs full Content-Based Image Retrieval (CBIR) evaluations to get mAP, P/R curves, and Top-1 accuracy.

Experiment Runners

exp_1.py: Tests global 3D RGB histograms at various quantization (Q) levels.

exp_3.py: Tests spatial grid descriptors (e.g., 8x8 grid with Color + Texture).

exp_4.py: Tests PCA dimensionality reduction on a high-dimensional descriptor set.

exp_5.py: Tests global LBP and HSV histograms with various parameters.

svm.py: Runs a classification experiment using GridSearchCV to find the best SVM hyperparameters for a given descriptor set.

Utilities & Generators

cvpr_visualsearch.py: A demo script to visually show search results for a single random query.

concatenate_images.py: A utility to stitch all images in a folder into one large image (horizontally or vertically).

cvpr_computedescriptors.py & cvpr_computedescriptors_3.py: Standalone scripts to generate global and spatial descriptors, respectively, without running an evaluation.

HOW TO RUN EXPERIMENTS

Important: Before running, you MUST edit the script file and update the paths at the bottom in the if __name__ == "__main__": block.

DATASET_FOLDER: Path to your MSRC_ObjCategImageDatabase_v2 directory.

DESCRIPTOR_BASE_FOLDER / DESCRIPTOR_FOLDER_TO_TEST: Folder to save/load descriptor .mat files.

RESULTS_DIR: Folder where all output (plots, summaries) will be saved.

Recommended Workflow:

Run Experiment 1 (RGB Histograms):

Edit paths in exp_1.py.

Run: python exp_1.py

Check experiment_results_1_new/comparison_summary.txt to find your best Q value.

Run Experiment 3 (Spatial Grids):

Edit paths in exp_3.py.

Update the BEST_Q_FOR_METRIC dictionary with the best Q value from step 1.

Run: python exp_3.py

Check experiment_results_req3_.../comparison_summary.txt to find your best spatial descriptor.

Run Experiment 4 (PCA):

Edit paths in exp_4.py.

Set DESCRIPTOR_TO_TEST to the path of your best descriptor from step 2 (e.g., .../descriptors_3/spatial_grid_8x8_color_hist_texture_Q24_T16).

Run: python exp_4.py

Run Experiment 5 (LBP & HSV):

Edit paths in exp_5.py.

Run: python exp_5.py

Run SVM Classification:

Edit paths in svm.py.

Set DESCRIPTOR_FOLDER_TO_TEST to the path of your best descriptor (likely from step 2 or 3).

Run: python svm.py
