import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
import matplotlib.pyplot as plt
import argparse

# --- Placeholder functions (if not imported from other files) ---
def cvpr_compare(f1, f2, method='euclidean'):
    """Placeholder: Calculates distance between two feature vectors."""
    if method == 'euclidean':
        return np.linalg.norm(f1 - f2)
    elif method == 'l1':
        return np.sum(np.abs(f1 - f2))
    elif method == 'chi_squared':
         epsilon = 1e-10
         return 0.5 * np.sum(((f1 - f2) ** 2) / (f1 + f2 + epsilon))
    else:
        raise ValueError(f"Unknown distance method: {method}")

def get_class_from_filename(filename):
    """Placeholder: Extracts class label from filename."""
    basename = os.path.basename(filename).replace('.bmp', '')
    parts = basename.split('_')
    return parts[0] if len(parts) >= 1 and parts[0].isdigit() else 'unknown'
# --- End Placeholder Definitions ---


def perform_visual_search(Q, descriptor_folder, image_folder, distance_method, show):
    """
    Performs a visual search for a random query image and displays results.
    """
    # Construct the path to the specific descriptor set
    descriptor_subfolder = f'globalRGBhisto_{Q}'
    full_descriptor_path = os.path.join(descriptor_folder, descriptor_subfolder)

    if not os.path.isdir(full_descriptor_path):
        print(f"Error: Descriptor folder not found at {full_descriptor_path}")
        return

    print(f"Loading descriptors from {full_descriptor_path}")

    # Load all descriptor files and corresponding image paths
    all_feat, all_files = [], []
    try:
        descriptor_files = sorted([f for f in os.listdir(full_descriptor_path) if f.endswith('.mat')])
    except FileNotFoundError:
        print(f"Error: Cannot list files in descriptor folder: {full_descriptor_path}")
        return

    for filename in descriptor_files:
        mat_path = os.path.join(full_descriptor_path, filename)
        img_filename = filename.replace(".mat", ".bmp")
        img_actual_path = os.path.join(image_folder, 'Images', img_filename)

        # Ensure the original image file exists
        if not os.path.exists(img_actual_path):
            print(f"Warning: Image file not found, skipping descriptor {filename}: {img_actual_path}")
            continue
        try:
            # Load the .mat file
            img_data = sio.loadmat(mat_path)
            if 'F' in img_data and isinstance(img_data['F'], np.ndarray):
                all_files.append(img_actual_path)
                all_feat.append(img_data['F'].flatten())
            else:
                print(f"Warning: Descriptor file {filename} has unexpected format. Skipping.")
        except Exception as e:
            print(f"Warning: Could not load or process descriptor {mat_path}: {e}")

    if not all_files:
        print("Error: No valid descriptors were loaded. Exiting.")
        return

    all_feat = np.array(all_feat)
    print(f"Loaded {all_feat.shape[0]} valid descriptors, each with {all_feat.shape[1]} dimensions.")

    # Select a random query image from the loaded set
    nimg = len(all_files)
    query_idx = randint(0, nimg - 1)
    query_file_path = all_files[query_idx]
    query_class_label = get_class_from_filename(query_file_path)
    print(f"\nQuery index: {query_idx}")
    print(f"Query image: {os.path.basename(query_file_path)} (Class: {query_class_label})")
    print(f"Using distance metric: {distance_method}")

    # Calculate distances from the query to all other images
    distances = []
    query_feat = all_feat[query_idx]
    for i in range(nimg):
        candidate_feat = all_feat[i]
        if query_feat.shape != candidate_feat.shape: continue
        distance = cvpr_compare(query_feat, candidate_feat, method=distance_method)
        distances.append((distance, i))

    # Sort results by distance (ascending)
    distances.sort(key=lambda x: x[0])

    # --- Plot the results ---
    num_results_to_display = min(show + 1, nimg)  # +1 to include query
    cols = 5
    rows = max(1, int(np.ceil(num_results_to_display / cols)))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows + 1.5), squeeze=False)
    fig.suptitle(f'Visual Search Results (Query: {os.path.basename(query_file_path)} [Class {query_class_label}], Distance: {distance_method})',
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    displayed_count = 0
    rank = 0
    for i in range(len(distances)):
        if displayed_count >= num_results_to_display: break

        dist, img_idx = distances[i]
        img_path = all_files[img_idx]
        img_class = get_class_from_filename(img_path)
        img = cv2.imread(img_path)
        ax = axes[displayed_count]

        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            title_str, title_color = "", 'black'
            
            # Highlight the query image
            if img_idx == query_idx:
                title_str = f'QUERY\nClass: {img_class}'
                title_color = 'red'
            else:
                rank += 1
                title_str = f'Rank {rank}\nClass: {img_class}\nDist: {dist:.3f}'
            ax.set_title(title_str, fontsize=9, color=title_color)
            ax.axis('off')
        else:
            # Handle missing image files
            ax.text(0.5, 0.5, f'Image not found\n{os.path.basename(img_path)}', ha='center', va='center', fontsize=8)
            ax.axis('off')
        displayed_count += 1

    # Hide any unused subplots
    for j in range(displayed_count, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Perform visual search and display results.')
    parser.add_argument('--Q', type=int, required=True, help='Quantization level (e.g., 8)')
    parser.add_argument('--distance', type=str, default='euclidean', choices=['euclidean', 'l1', 'chi_squared'], help='Distance metric')
    parser.add_argument('--show', type=int, default=19, help='Number of top results to show (excluding query)')
    parser.add_argument('--descriptors', type=str, default='descriptors', help='Base folder containing descriptor subfolders')
    parser.add_argument('--images', type=str, default='MSRC_ObjCategImageDatabase_v2', help='Base folder containing the "Images" subfolder')
    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.isdir(args.images) or not os.path.isdir(os.path.join(args.images, 'Images')):
         print(f"Error: Valid image folder with 'Images' subfolder not found at {args.images}")
    elif not os.path.isdir(args.descriptors):
         print(f"Error: Descriptor base folder not found at {args.descriptors}")
    else:
        # Run the search
        perform_visual_search(Q=args.Q, descriptor_folder=args.descriptors, image_folder=args.images, distance_method=args.distance, show=args.show)
