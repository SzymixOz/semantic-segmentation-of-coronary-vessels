# import types
from typing import Tuple

# required imports
import os
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
    repeat_number = 2

    mask_to_preprocess = mask
    for _ in range(repeat_number):

        selem = disk(1.001)
        image_dilated = dilation(mask_to_preprocess, selem)

        selem = disk(1.5001)
        mask_to_preprocess = erosion(image_dilated, selem)

    return mask_to_preprocess


def get_skeleton_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    This function takes a binary mask as input and returns the skeleton of the mask.

    Parameters:
    - mask (np.ndarray): A binary mask of the image.

    Returns:
    - skeleton (np.ndarray): The skeleton of the mask.
    """
    mask = mask > 0
    skeleton = skeletonize(mask, method="zhang")

    return skeleton


def load_first(dir_mask: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function takes a directory containing images and returns the first image found in the directory.

    Parameters:
    - dir_mask (str): The directory containing the masks.

    Returns:
    - image_name (str): The name of the image.
    - mask (np.ndarray): The first mask found in the directory.
    """
    import os

    for filename in os.listdir(dir_mask):
        if filename.endswith(".png"):
            mask = cv2.imread(dir_mask + filename)
            return filename.split(".")[0], mask
    return None, None


def find_keypoints_on_graph(
    skeleton: np.ndarray
) -> Tuple[np.ndarray, list, set, list, list, list]:
    """
    This function takes a skeletonized image as input and returns the coordinates of the skeleton points, endpoints, crossings, bifurcations, and filtered out points.

    Parameters:
    - skeleton (np.ndarray): The skeletonized image.

    Returns:
    - coordinates (tuple): A tuple containing the coordinates of the skeleton points.
    - endpoints (list): A list of indices representing endpoint points in the skeleton.
    - crossings (set): A set of indices representing crossing points in the skeleton.
    - bifurcations (list): A list of indices representing bifurcation points in the skeleton.
    - filtered_out (list): A list of indices representing filtered out points in the skeleton.
    - crossings_connected (list): A list of tuples representing connected crossings in the skeleton.
    """

    distance_threshold = 6

    pixel_graph, coordinates = csr.skeleton_to_csgraph(skeleton)

    G = nx.from_scipy_sparse_array(pixel_graph)

    endpoints = [node for node, degree in G.degree() if degree == 1]
    branchpoints = [node for node, degree in G.degree() if degree >= 3]
    branchpoints = sorted(branchpoints, key=lambda x: coordinates[0][x] ** 2 + coordinates[1][x] ** 2)

    filtered_branchpoints = []
    filtered_out = []
    for branchpoint in branchpoints:
        try:
            min_distance = min(
                nx.shortest_path_length(G, branchpoint, endpoint)
                for endpoint in endpoints
                if nx.has_path(G, branchpoint, endpoint)
            )
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
                        new_depth = (
                            depth + 1 if next in filtered_branchpoints else depth
                        )  # (this is for not checking filtered_out)
                        if new_depth < 2:
                            queue.append((next, path + [next], new_depth))

    def bfs_find_in_distance(graph, start, max_distance, filtered_branchpoints):
        queue = [(start, [start], 0)]
        visited = set()
        while queue:
            (vertex, path, distance) = queue.pop(0)
            visited.add(vertex)
            for next in graph[vertex]:
                if next not in visited:
                    new_distance = distance + 1
                    if next in filtered_branchpoints:
                        if coordinates[0][start] - coordinates[0][next] < 40:
                            return next, False
                        continue 
                    elif next in endpoints:
                        return next, True
                    
                    if new_distance < max_distance:
                        queue.append((next, path + [next], new_distance))
                    else:
                        return None, False

    crossings = set()
    crossings_connected = []
    # filtered_branchpoints = [branchpoint for branchpoint in filtered_branchpoints if branchpoint not in crossings] doubt if this is needed
    
    for i, branchpoint1 in enumerate(filtered_branchpoints):
        for branchpoint2 in filtered_branchpoints[i + 1 :]:
            paths = []
            # paths = list(bfs_paths(G, branchpoint1, branchpoint2))
            for neighbor in G.neighbors(branchpoint1):
                new_paths = list(bfs_paths(G, neighbor, branchpoint2, branchpoint1))
                paths.extend(
                    new_path
                    for new_path in new_paths
                    if not any(
                        set(new_path[1:-1]).intersection(set(existing_path[1:-1]))
                        for existing_path in paths
                    )
                )
            if len(paths) > 1:
                crossings.add(branchpoint2)

    max_distance = 70
    for crossing in crossings.copy():
        nearest_branchpoint, is_filtered_out = bfs_find_in_distance(G, crossing, max_distance, filtered_branchpoints)
        if nearest_branchpoint is not None and not is_filtered_out:
            crossings_connected.append((crossing, nearest_branchpoint))
            crossings.remove(crossing)
        elif nearest_branchpoint is None:
            crossings.remove(crossing)

    crossings_connected_copy = crossings_connected.copy()
    for i, (cross_begin1, cross_end1) in enumerate(crossings_connected_copy):
        for cross_begin2, cross_end2 in crossings_connected_copy[i + 1:]:
            if cross_begin1 == cross_begin2 or cross_end1 == cross_end2 or cross_begin1 == cross_end2 or cross_end1 == cross_begin2:
                path1 = nx.shortest_path(G, cross_begin1, cross_end1)
                path2 = nx.shortest_path(G, cross_begin2, cross_end2)
                if len(path1) < len(path2):
                    if (cross_begin2, cross_end2) in crossings_connected:
                        crossings_connected.remove((cross_begin2, cross_end2))
                else:
                    if (cross_begin1, cross_end1) in crossings_connected:
                        crossings_connected.remove((cross_begin1, cross_end1))

    # print(f"Possible: {crossings_connected}")

    filtered_branchpoints = [
        branchpoint
        for branchpoint in filtered_branchpoints
        if branchpoint not in crossings
    ]
    filtered_branchpoints = [
        branchpoint
        for branchpoint in filtered_branchpoints
        if branchpoint not in [branchpoint for branchpoint, _ in crossings_connected]
    ]
    filtered_branchpoints = [
        branchpoint
        for branchpoint in filtered_branchpoints
        if branchpoint not in [branchpoint for _, branchpoint in crossings_connected]
    ]

    return (
        coordinates,
        endpoints,
        crossings,
        filtered_branchpoints,
        filtered_out,
        crossings_connected,
    )


def draw_everything_else(
    image_name: str,
    image_binary: np.ndarray,
    skeleton: np.ndarray,
    coordinates: Tuple[np.ndarray, np.ndarray],
    endpoints: list,
    crossings: set,
    filtered_branchpoints: list,
    filtered_out: list,
    crossings_connected: list,
) -> None:
    """
    This function takes a binary image, a skeletonized image, and the keypoints of the skeleton and displays the images side by side.

    Parameters:
    - image_name (str): The name of the image.
    - image_binary (np.ndarray): The binary image.
    - skeleton (np.ndarray): The skeletonized image.
    - coordinates (tuple): A tuple containing the coordinates of the skeleton points.
    - endpoints (list): A list of indices representing endpoint points in the skeleton.
    - crossings (set): A set of indices representing crossing points in the skeleton.
    - filtered_branchpoints (list): A list of indices representing filtered branch points in the skeleton.
    - filtered_out (list): A list of indices representing filtered out points in the skeleton.
    """

    fig1 = plt.figure(figsize=(10, 5))
    gs1 = fig1.add_gridspec(1, 2)

    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.imshow(image_binary, cmap="gray")
    ax1.axis("off")
    ax1.set_title("Obraz DICOM")

    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.imshow(skeleton)
    ax2.axis("off")
    ax2.set_title("Szkielet")

    plt.tight_layout()
    # plt.show()
    plt.close()

    fig2 = plt.figure(figsize=(10, 10))
    ax3 = fig2.add_subplot(111)
    ax3.imshow(skeleton, cmap="gray")
    ax3.axis("off")

    size = 50
    for endpoint in endpoints:
        ax3.scatter(coordinates[1][endpoint], coordinates[0][endpoint], c="y", s=size)

    for branchpoint in filtered_branchpoints:
        ax3.scatter(
            coordinates[1][branchpoint], coordinates[0][branchpoint], c="b", s=size
        )

    for branchpoint in filtered_out:
        ax3.scatter(
            coordinates[1][branchpoint], coordinates[0][branchpoint], c="g", s=size
        )

    for branchpoint in crossings:
        ax3.scatter(
            coordinates[1][branchpoint], coordinates[0][branchpoint], c="r", s=size
        )

    for branchpoint1, branchpoint2 in crossings_connected:

        ax3.scatter(
            coordinates[1][branchpoint1],
            coordinates[0][branchpoint1],
            c="r",
            s=size,
            zorder=5,
        )
        ax3.scatter(
            coordinates[1][branchpoint2],
            coordinates[0][branchpoint2],
            c="r",
            s=size,
            zorder=5,
        )

        ax3.plot(
            [coordinates[1][branchpoint1], coordinates[1][branchpoint2]],
            [coordinates[0][branchpoint1], coordinates[0][branchpoint2]],
            c="orange",
            linewidth=4,
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Bifurcation",
            markerfacecolor="b",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Filtered Out",
            markerfacecolor="g",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Potential Crossing",
            markerfacecolor="r",
            markersize=15,
        ),
        Line2D([0], [0], color="orange", linewidth=4, label="Connected Crossings"),
    ]
    ax3.legend(handles=legend_elements, loc="upper right", prop={"size": 15})
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if not os.path.exists("keypoints_legend"):
        os.makedirs("keypoints_legend")

    plt.savefig(f"keypoints_legend/{image_name}.png", bbox_inches='tight', pad_inches=0)
    # plt.show()
    plt.close()

def save_points_to_grayscale_image(
    image_name: str,
    dir_name: str,
    skeleton: np.ndarray,
    coordinates: Tuple[np.ndarray, np.ndarray],
    bifurcations: list,
    crossings: set,
    crossings_connected: list,
    endpoints: list,
) -> None:
    """
    This function saves the keypoints to a single-layer grayscale image.

    The grayscale image will have the following values:
    - 0: Background
    - 1: Skeleton
    - 2: Bifurcations (branch points)
    - 3: Crossings (nodes calculated as crossings)
    - 4: Endpoints (nodes with degree = 1)

    Parameters:
    - image_name (str): The name of the image.
    - dir_name (str): The directory where the grayscale image will be saved
    - image (np.ndarray): The original RGB image (not used in the grayscale image).
    - skeleton (np.ndarray): The skeletonized version of the image.
    - coordinates (tuple): A tuple containing the coordinates of the skeleton points.
    - bifurcations (list): A list of indices representing bifurcation points in the skeleton.
    - crossings (set): A set of indices representing crossing points in the skeleton.
    - endpoints (list): A list of indices representing endpoint points in the skeleton.

    The function will save the resulting grayscale image inside a folder named "keypoints" with the filename "{image_name}.png".
    """

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    height, width = skeleton.shape[:2]
    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    grayscale_image[skeleton > 0] = 1

    def draw_points(points, value):
        for point in points:
            cv2.circle(
                grayscale_image,
                (coordinates[1][point], coordinates[0][point]),
                radius=2,
                color=value,
                thickness=-1,
            )

    draw_points(bifurcations, 2)

    # For now like this, unless we want to draw path
    for cross_begin, cross_end in crossings_connected:
        crossings.add(cross_begin)
        crossings.add(cross_end)

    draw_points(crossings, 3)
    # draw_points(endpoints, 4) - Not needed

    cv2.imwrite(f"{dir_name}/{image_name}.png", grayscale_image)


def keypoints(image_name: str, dir_name: str, mask: np.ndarray, with_plot: bool = True) -> None:
    """
    This function takes a binary mask as input and returns the keypoints of the mask.

    Parameters:
    - image_name (str): The name of the image.
    - dir_name (str): The directory where the grayscale image will be saved
    - mask (np.ndarray): A binary mask of the image.
    - with_plot (bool): A boolean indicating whether to save the keypoints with legend.
    """
    mask_preprocessed = preprocessing(mask)

    skeleton = get_skeleton_from_mask(mask_preprocessed)

    (
        coordinates,
        endpoints,
        crossings,
        filtered_branchpoints,
        filtered_out,
        crossings_connected,
    ) = find_keypoints_on_graph(skeleton)

    if with_plot:
        draw_everything_else(
            image_name,
            mask_preprocessed,
            skeleton,
            coordinates,
            endpoints,
            crossings,
            filtered_branchpoints,
            filtered_out,
            crossings_connected,
        )

    save_points_to_grayscale_image(
        image_name, dir_name, skeleton, coordinates, filtered_branchpoints, crossings, crossings_connected, endpoints
    )


def make_all_images_from_dir():
    """
    This function loads the first image from the dataset and displays the keypoints.
    """
    dir_mask = "../seg_binary/images_binary/" # You can select correct directory here

    print("GENERATING KEYPOINTS...")
    std_output = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    for filename in os.listdir(dir_mask):
        if filename.endswith(".png"):
            mask = cv2.imread(dir_mask + filename)

            # Convert to binary
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_binary = mask_gray > 0
            keypoints(filename.split(".")[0], "keypoints_from_dir", mask_binary, with_plot=False)

    sys.stdout = std_output
    print("GENERATION OF KEYPOINTS FINISHED")

import sys
sys.path.append("../images_with_proper_colors/")
from utils import *

def make_all_images_from_csv():
    """
    This function loads all images from the dataset and displays the keypoints.
    """
    data_folder = "../images_with_proper_colors/"
    data = pd.read_csv(data_folder + "segmentation_modified.csv", sep=";")

    images_voting, segmentations_voting, filenames_voting, labels_voting = get_data(
        data, voting=True, images_path=(data_folder + "images"), labeling=True
    )

    print("GENERATING KEYPOINTS...")
    std_output = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    for index in range(len(images_voting)):
        mask = get_mask(
            images_voting[index],
            segmentations_voting[index],
            labels_voting[index],
            name=filenames_voting[index],
            folder_name="ground_truth",
            binary=True,
            ground_truth=True,
        )
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = mask_gray > 0
        keypoints(filenames_voting[index], "keypoints_from_csv", mask_binary, with_plot=False)

    sys.stdout = std_output
    print("GENERATION OF KEYPOINTS FINISHED")


def get_keypoints(mask):
    """
    This function takes a binary mask as input and returns the keypoints of the mask.
    """
    image_name = "temp_name"
    dir_name = "./"
    keypoints(image_name, dir_name, mask, with_plot=False)
    keypoints_image = cv2.imread(f"{dir_name}/{image_name}.png", cv2.IMREAD_GRAYSCALE)
    # usuń zdjęcie
    os.remove(f"{dir_name}/{image_name}.png")
    return keypoints_image