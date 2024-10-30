# import types
from typing import Tuple

# required imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion, disk, dilation
from scipy.ndimage import laplace
import cv2
import networkx as nx
from skimage.morphology import skeletonize
from skan import csr
from matplotlib.lines import Line2D


def preprocessing(mask: np.ndarray) -> np.ndarray:  
    """
        This function takes a binary mask as input and returns a preprocessed mask.

        Parameters:
        - mask (np.ndarray): A binary mask of the image.

        Returns:
        - mask_erosion (np.ndarray): The preprocessed mask.
    """
    
    selem = disk(1.001)
    image_dilated = dilation(mask, selem)

    mask_sharpened = image_dilated - 0.7 * laplace(image_dilated)

    # You can manipuilate the value of the disk to get different results
    selem = disk(1.5001)
    mask_erosion = erosion(mask_sharpened, selem)

    return mask_erosion

def get_skeleton_from_mask(mask: np.ndarray) -> np.ndarray:
    """
        This function takes a binary mask as input and returns the skeleton of the mask.

        Parameters:
        - mask (np.ndarray): A binary mask of the image.

        Returns:
        - skeleton (np.ndarray): The skeleton of the mask.
    """
    mask = mask > 0
    skeleton = skeletonize(mask, method='zhang')

    return skeleton



def load_first(dir_image: str, dir_mask: str) -> Tuple[np.ndarray, np.ndarray]:
    """
        This function takes a directory containing images and returns the first image found in the directory.

        Parameters:
        - dir_image (str): The directory containing the images.
        - dir_mask (str): The directory containing the masks.

        Returns:
        - image_name (str): The name of the image.
        - image (np.ndarray): The first image found in the directory.
        - mask (np.ndarray): The first mask found in the directory.
    """
    import os
    for filename in os.listdir(dir_mask):
        if filename.endswith(".png"):
            image = cv2.imread(dir_image + filename)
            mask = cv2.imread(dir_mask + filename)
            return filename.split('.')[0], image, mask
    return None, None, None


def find_keypoints_on_graph(skeleton: np.ndarray) -> Tuple[np.ndarray, list, list, list, list]:
    """
        This function takes a skeletonized image as input and returns the coordinates of the skeleton points, endpoints, crossings, bifurcations, and filtered out points.

        Parameters:
        - skeleton (np.ndarray): The skeletonized image.

        Returns:
        - coordinates (tuple): A tuple containing the coordinates of the skeleton points.
        - endpoints (list): A list of indices representing endpoint points in the skeleton.
        - crossings (list): A list of indices representing crossing points in the skeleton.
        - bifurcations (list): A list of indices representing bifurcation points in the skeleton.
        - filtered_out (list): A list of indices representing filtered out points in the skeleton.
        - crossings_connected (list): A list of tuples representing connected crossings in the skeleton.
    """


    distance_threshold = 6

    # Create a graph from the skeleton
    pixel_graph, coordinates = csr.skeleton_to_csgraph(skeleton)

    G = nx.from_scipy_sparse_array(pixel_graph)

    # draw graph
    endpoints = [node for node, degree in G.degree() if degree == 1]
    branchpoints = [node for node, degree in G.degree() if degree == 3]
    # degree > 3 for crossroads
    crossroads = [node for node, degree in G.degree() if degree > 3] # Not sure if there will be ever any ? 

    # Filter out branchpoints that are close to endpoints
    filtered_branchpoints = []
    filtered_out = []
    for branchpoint in branchpoints:
        try:
            min_distance = min(nx.shortest_path_length(G, branchpoint, endpoint) for endpoint in endpoints if nx.has_path(G, branchpoint, endpoint))
            if min_distance > distance_threshold:
                filtered_branchpoints.append(branchpoint)
            else:
                filtered_out.append(branchpoint)
        except ValueError:
            print(f"No path found from branchpoint {branchpoint} to any endpoint.")

    def bfs_paths(graph, start, goal, branchpoint):
        queue = [(start, [start], 0)]
        visited = set()
        visited.add(branchpoint)
        while queue:
            (vertex, path, depth) = queue.pop(0)
            visited.add(vertex)
            for next in graph[vertex]:
                if next not in visited:
                    if next == goal:
                        yield path + [next]
                    else:
                        # for consideration if i should check next in filtered_out #TODO
                        # new_depth = depth + 1 if next in filtered_branchpoints or next in filtered_out else depth
                        new_depth = depth + 1 if next in filtered_branchpoints else depth # (this is for not checking filtered_out)
                        if new_depth < 2:
                            queue.append((next, path + [next], new_depth))

    # define bfs that will check for that one neighbor and find nearest branchpoint
    def bfs_find_in_distance(graph, start, branchpoints, max_distance):
        queue = [(start, [start], 0)]
        visited = set()
        for branchpoint in branchpoints:
            visited.add(branchpoint)
        while queue:
            (vertex, path, distance) = queue.pop(0)
            visited.add(vertex)
            for next in graph[vertex]:
                if next not in visited:
                    new_distance = distance + 1
                    if next in filtered_branchpoints:
                        return next
                    if new_distance < max_distance:
                        queue.append((next, path + [next], new_distance))
                    else:
                        return None
    
    crossings = []
    crossings_connected = []
    # filtered_branchpoints = [branchpoint for branchpoint in filtered_branchpoints if branchpoint not in crossings] doubt if this is needed
    
    for i, branchpoint1 in enumerate(filtered_branchpoints):
        for branchpoint2 in filtered_branchpoints[i+1:]:
            paths = []
            # paths = list(bfs_paths(G, branchpoint1, branchpoint2))
            for neighbor in G.neighbors(branchpoint1):
                new_paths = list(bfs_paths(G, neighbor, branchpoint2, branchpoint1))
                paths.extend(new_path for new_path in new_paths if not any(set(new_path[1:-1]).intersection(set(existing_path[1:-1])) for existing_path in paths))
      
            if len(paths) > 1:
                neighbors = list(G.neighbors(branchpoint2))
                neighbors_filtered_out = []
                for neighbor in neighbors:
                    for path in paths:
                        if neighbor in path:
                            neighbors_filtered_out.append(neighbor)
                            break
                neighbors = [neighbor for neighbor in neighbors if neighbor not in neighbors_filtered_out]
                if len(neighbors) > 1:
                    crossings.append(branchpoint2)
                    # Not sure if that ever will happen
                else:
                    max_distance = 50
                    nearest_branchpoint = bfs_find_in_distance(G, branchpoint2, neighbors_filtered_out, max_distance)
                    if nearest_branchpoint is not None:
                        crossings_connected.append((branchpoint2, nearest_branchpoint))

    print(f"Possible: {crossings_connected}")

    filtered_branchpoints = [branchpoint for branchpoint in filtered_branchpoints if branchpoint not in crossings]
    filtered_branchpoints = [branchpoint for branchpoint in filtered_branchpoints if branchpoint not in [branchpoint for branchpoint, _ in crossings_connected]]
    filtered_branchpoints = [branchpoint for branchpoint in filtered_branchpoints if branchpoint not in [branchpoint for _, branchpoint in crossings_connected]]
    
    return coordinates, endpoints, crossings, filtered_branchpoints, filtered_out, crossings_connected

def draw_everything_else(image_binary: np.ndarray, skeleton: np.ndarray, coordinates: Tuple[np.ndarray, np.ndarray], endpoints: list, crossings: list, filtered_branchpoints: list, filtered_out: list, crossings_connected: list) -> None:
    """
        This function takes a binary image, a skeletonized image, and the keypoints of the skeleton and displays the images side by side.

        Parameters:
        - image_binary (np.ndarray): The binary image.
        - skeleton (np.ndarray): The skeletonized image.
        - coordinates (tuple): A tuple containing the coordinates of the skeleton points.
        - endpoints (list): A list of indices representing endpoint points in the skeleton.
        - crossings (list): A list of indices representing crossing points in the skeleton.
        - filtered_branchpoints (list): A list of indices representing filtered branch points in the skeleton.
        - filtered_out (list): A list of indices representing filtered out points in the skeleton.
    """
    
    fig1 = plt.figure(figsize=(10, 5))
    gs1 = fig1.add_gridspec(1, 2)

    # First subplot
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.imshow(image_binary, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Obraz DICOM')

    # Second subplot
    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.imshow(skeleton)
    ax2.axis('off')
    ax2.set_title('Szkielet')

    plt.tight_layout()
    plt.show()

    # Second figure for the third image
    fig2 = plt.figure(figsize=(10, 10))
    ax3 = fig2.add_subplot(111)
    ax3.imshow(skeleton, cmap='gray')
    ax3.axis('off')
    ax3.set_title('Szkielet z punktami')

    size = 50
    for endpoint in endpoints:
        ax3.scatter(coordinates[1][endpoint], coordinates[0][endpoint], c='y', s=size)

    # for branchpoint in branchpoints:
    #     plt.scatter(coordinates[1][branchpoint], coordinates[0][branchpoint], c='b', s=10)

    # for branchpoint in crossroads:
    #     plt.scatter(coordinates[1][branchpoint], coordinates[0][branchpoint], c='g', s=10)
    
    for branchpoint in filtered_branchpoints:
        ax3.scatter(coordinates[1][branchpoint], coordinates[0][branchpoint], c='b', s=size)

    for branchpoint in filtered_out:
        ax3.scatter(coordinates[1][branchpoint], coordinates[0][branchpoint], c='g', s=size)
    
    for branchpoint in crossings:
        # candidates for crossroads
        ax3.scatter(coordinates[1][branchpoint], coordinates[0][branchpoint], c='r', s=size)
        
    for branchpoint1, branchpoint2 in crossings_connected:

        ax3.scatter(coordinates[1][branchpoint1], coordinates[0][branchpoint1], c='r', s=size, zorder=5)
        ax3.scatter(coordinates[1][branchpoint2], coordinates[0][branchpoint2], c='r', s=size, zorder=5)

        ax3.plot([coordinates[1][branchpoint1], coordinates[1][branchpoint2]], [coordinates[0][branchpoint1], coordinates[0][branchpoint2]], c='orange', linewidth=4)
        
    # legend blue - bifurcation, red potential crossing, green - filtered out
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Bifurcation', markerfacecolor='b', markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Filtered Out', markerfacecolor='g', markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Potential Crossing', markerfacecolor='r', markersize=15),
                     Line2D([0], [0], color='orange', linewidth=4, label='Connected Crossings')]
    ax3.legend(handles=legend_elements, loc='upper right', prop={'size': 15})

    # plt.subplots_adjust(wspace=0.05, hspace=0.10)
    plt.tight_layout()
    plt.show()

import os

def save_points_to_grayscale_image(image_name: str, skeleton: np.ndarray, coordinates: Tuple[np.ndarray, np.ndarray], bifurcations: list, crossings: list, endpoints: list) -> None:
    """
        This function saves the keypoints to a single-layer grayscale image.

        The grayscale image will have the following values:
        - 0: Background
        - 1: Skeleton
        - 2: Bifurcations (branch points)
        - 3: Crossings (nodes with degree > 3)
        - 4: Endpoints (nodes with degree = 1)

        Parameters:
        - image_name (str): The name of the image.
        - image (np.ndarray): The original RGB image (not used in the grayscale image).
        - skeleton (np.ndarray): The skeletonized version of the image.
        - coordinates (tuple): A tuple containing the coordinates of the skeleton points.
        - bifurcations (list): A list of indices representing bifurcation points in the skeleton.
        - crossings (list): A list of indices representing crossing points in the skeleton.
        - endpoints (list): A list of indices representing endpoint points in the skeleton.

        The function will save the resulting grayscale image inside a folder named "keypoints" with the filename "{image_name}.png".
    """

    if not os.path.exists('keypoints'):
        os.makedirs('keypoints')

    height, width = skeleton.shape[:2]
    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    grayscale_image[skeleton > 0] = 1

    def draw_points(points, value):
        for point in points:
            cv2.circle(grayscale_image, (coordinates[1][point], coordinates[0][point]), radius=2, color=value, thickness=-1)


    draw_points(bifurcations, 2)
    draw_points(crossings, 3)
    draw_points(endpoints, 4)

    cv2.imwrite(f'keypoints/{image_name}.png', grayscale_image)


def keypoints(image_name: str, mask: np.ndarray, with_plot: bool = True) -> None:
    """
        This function takes a binary mask as input and returns the keypoints of the mask.

        Parameters:
        - mask (np.ndarray): A binary mask of the image.
    """
    mask_preprocessed = preprocessing(mask)

    skeleton = get_skeleton_from_mask(mask_preprocessed)

    coordinates, endpoints, crossings, filtered_branchpoints, filtered_out, crossings_connected = find_keypoints_on_graph(skeleton)

    if with_plot:
        draw_everything_else(mask_preprocessed, skeleton, coordinates, endpoints, crossings, filtered_branchpoints, filtered_out, crossings_connected)

    save_points_to_grayscale_image(image_name, skeleton, coordinates, filtered_branchpoints, crossings, endpoints)


def make_first_image():
    """
        This function loads the first image from the dataset and displays the keypoints.
    """
    image_folder = "../images_with_proper_colors/images/"
    mask_folder = "../images_with_proper_colors/bin/"

    image_name, image, mask = load_first(image_folder, mask_folder) # That is not binary image. rememeber!
    print(f'Loaded mask: {mask.shape} with values: {np.unique(mask)}')

    # Convert to binary
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_binary = mask_gray > 0

    keypoints(image_name, mask_binary)


import sys
sys.path.append("../images_with_proper_colors/")
from utils import *

def make_all_images():
    """
        This function loads all images from the dataset and displays the keypoints.
    """
    data_folder = "../images_with_proper_colors/"
    data = pd.read_csv(data_folder + "segmentation_modified.csv", sep=";")

    images_voting, segmentations_voting, filenames_voting, labels_voting = get_data(data, voting=True, images_path=(data_folder + "images"), labeling=True)
    
    print("GENERATING KEYPOINTS...")
    # std_output = sys.stdout
    # sys.stdout = open(os.devnull, 'w') 

    for index in range(len(images_voting)):
        if index < 0: # Skip first 8 images
            continue
        if index > 0: # Skip all images after 8th
            break
        mask = get_mask(images_voting[index], segmentations_voting[index], labels_voting[index], name=filenames_voting[index], folder_name='ground_truth', binary=True, ground_truth=True)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = mask_gray > 0
        keypoints(filenames_voting[index], mask_binary, with_plot=True)

    # sys.stdout = std_output

    print("GENERATION OF KEYPOINTS FINISHED")

if __name__ == '__main__':
    make_all_images() 
    # make_first_image()
 