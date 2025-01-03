import networkx as nx
from scipy.spatial import Delaunay
import pandas as pd
import multiprocessing
import matplotlib
matplotlib.use('TkAgg')
from skimage.measure import regionprops
import matplotlib
import os
from math import sqrt
from scipy import ndimage
from skimage import measure, morphology, segmentation, feature, filters
from czifile import imread as read_czi
from scipy.ndimage import distance_transform_edt
from skimage.feature import blob_log
from skimage.morphology import binary_erosion, binary_dilation, area_opening, area_closing
matplotlib.use('TkAgg')
import cv2
import numpy as np
import os
import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, morphology, segmentation, filters
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from czifile import imread as read_czi
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, binary_erosion, area_opening
import os
import numpy as np
import cv2
from scipy import ndimage
from skimage import measure, morphology, segmentation, filters
from skimage.color import gray2rgb, rgb2gray
import matplotlib.pyplot as plt
from czifile import imread as read_czi
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, binary_erosion, area_opening


os.chdir('/nfs/cc-filer/home/sabulikailik/images/A1818')
# Function to dilate the mask
def dilate_mask(mask, size):
    selem = morphology.disk(size)
    dilated_mask = morphology.dilation(mask, selem)
    return dilated_mask
def extract_centroids(unet_mask):
    labels = measure.label(unet_mask)
    prop = regionprops(labels)
    centroids = np.array([prop[i].centroid for i in range(len(prop))])
    centroids = np.round(centroids).astype(int)  # Round and convert to integers
    return centroids

# Function to create seed image from centroids
def create_seed_image(centroids, shape):
    seed_image = np.zeros(shape, dtype=np.uint8)
    for centroid in centroids:
        cv2.circle(seed_image, (int(centroid[1]), int(centroid[0])), 2, 255, -1)  # Draw a small circle at each centroid
    return seed_image

def signed_distance_transform(true_mask):
    """
    Compute the signed distance transform of a binary mask.

    Parameters:
    - true_mask: numpy.ndarray, a binary mask where the foreground is 1 and the background is 0.

    Returns:
    - sdt: numpy.ndarray, the signed distance transform of the binary mask.
    """

    # Distance transform on the true mask (foreground)
    dt_foreground = distance_transform_edt(true_mask)

    # Distance transform on the inverted mask (background)
    dt_background = distance_transform_edt(1 - true_mask)

    # Assign negative values to the original foreground
    sdt = dt_background - dt_foreground

    return sdt

# Function to process the data for a single image
def process_image(image_filename, image_dir, mask_dir):
    image_path = os.path.join(image_dir, image_filename)
    mask_filename = image_filename.replace('.czi', '_mask.png')
    mask_path = os.path.join(mask_dir, mask_filename)

    image = read_czi(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


    I1 = image[0, :1024, :1280, 0]
    I2 = image[1, :1024, :1280, 0]
    I3 = image[2, :1024, :1280, 0]
    I4 = image[3, :1024, :1280, 0]
    I5 = image[4, :1024, :1280, 0]
    I6 = image[5, :1024, :1280, 0]

    # Process image layers
    mask12 = np.uint8(mask > 0)
    filled_mask1 = ndimage.binary_fill_holes(mask12).astype(np.uint8)
    filled_mask12 = area_opening(filled_mask1, 100, connectivity=1)
    mask2 = binary_erosion(filled_mask12, footprint=np.ones((3, 3)))

    centroids = extract_centroids(mask2)
    seed_image = create_seed_image(centroids, mask2.shape)

    smooth_mask = filters.gaussian(filled_mask12, sigma=0.2)
    dilated_mask = binary_dilation(smooth_mask, footprint=np.ones((1, 1)))

    distance = distance_transform_edt(mask2)
    sdt = signed_distance_transform(mask2)
    sdt_final = sdt.max() - sdt

    markers = measure.label(seed_image)
    labels = segmentation.watershed(-sdt_final, markers, mask=dilated_mask)



    # Nuclear mask
    nuclear_mask = labels
    props1 = regionprops(nuclear_mask, I1)

    # Process data and calculate mean and median
    image_id = []
    nuclei_id = []
    area = []
    pleo = []
    elong = []
    avg_grad = []
    var_grad = []
    cellu = []
    mask = []
    bbox = []
    mean1 = []
    tot1 = []
    cens = []
    mean_cd3_list = []
    median_cd3_list = []
    mean_psmad_list = []
    median_psmad_list = []
    mean_cd8_list = []
    median_cd8_list = []
    mean_ki67_list = []
    median_ki67_list = []
    mean_caspase_list = []
    median_caspase_list = []
    mean_cd8_nuc = []
    median_cd8_nuc = []
    for i in range(len(props1)):
        cen = props1[i].centroid
        cens.append(cen)

        # Convert cens list to a NumPy array
    cens = np.array(cens)

    if len(cens) == 0:
        print(f"Skipping file '{image_filename}' due to insufficient data.", cens)
        return None

    # Compute Delaunay Triangulation
    tri = Delaunay(cens)

    # Construct the graph
    graph = nx.Graph()
    for i, path in enumerate(tri.simplices):
        nx.add_path(graph, path)

    node_data = {}

    # Append the lengths of all connections for each node
    for node in graph.nodes():
        connections = list(graph.neighbors(node))
        neighbor_names = []
        edge_lengths = []

        for neighbor in connections:
            edge_length = np.linalg.norm(cens[node] - cens[neighbor])
            neighbor_names.append(neighbor)
            edge_lengths.append(round(edge_length, 3))

        node_data[node] = {
            'neighbor_names': neighbor_names,
            'edge_lengths': edge_lengths
        }

    for nid1 in range(len(props1)):
        output_image = np.zeros_like(I1, dtype=np.uint8)
        region_coords = props1[nid1].coords
        output_image[region_coords[:, 0], region_coords[:, 1]] = 1

        dapi_dilated = dilate_mask(output_image, 8)
        propdapi = regionprops(output_image, I1)

        # Create a mask to exclude the inner part
        inner_part_mask = dapi_dilated - output_image
        inner_part_mask = np.uint8(inner_part_mask)

        A = propdapi[0].area
        sol = propdapi[0].solidity
        eccin = propdapi[0].eccentricity
        to1 = propdapi[0].image_intensity
        box = propdapi[0].bbox
        mean11 = np.mean(to1)
        tot11 = np.sum(to1[to1 > 0])
        cd3_roi = inner_part_mask*I2
        psmad_roi = output_image*I3
        cd8_roi_outter = inner_part_mask*I4
        cd8_roi_inner = output_image * I4
        ki67_roi = output_image*I5
        caspase = output_image*I6

        # Calculate mean and median signals
        mean_cd3 = np.mean(cd3_roi[cd3_roi > 0])
        median_cd3 = np.median(cd3_roi[cd3_roi > 0])
        mean_psmad = np.mean(psmad_roi[psmad_roi > 0])
        median_psmad = np.median(psmad_roi[psmad_roi > 0])
        mean_cd8_out = np.mean(cd8_roi_outter[cd8_roi_outter > 0])
        median_cd8_out = np.median(cd8_roi_outter[cd8_roi_outter > 0])
        mean_cd8_in = np.mean(cd8_roi_inner[cd8_roi_inner > 0])
        median_cd8_in = np.median(cd8_roi_inner[cd8_roi_inner > 0])
        mean_ki67 = np.mean(ki67_roi[ki67_roi > 0])
        median_ki67 = np.median(ki67_roi[ki67_roi > 0])
        mean_caspase = np.mean(caspase[caspase > 0])
        median_caspase = np.median(caspase[caspase > 0])

        # Append each result to the respective list
        mean_cd8_nuc.append(mean_cd8_in)
        median_cd8_nuc.append(median_cd8_in)
        mean_cd3_list.append(mean_cd3)
        median_cd3_list.append(median_cd3)
        mean_psmad_list.append(mean_psmad)
        median_psmad_list.append(median_psmad)
        mean_cd8_list.append(mean_cd8_out)
        median_cd8_list.append(median_cd8_out)
        mean_ki67_list.append(mean_ki67)
        median_ki67_list.append(median_ki67)
        mean_caspase_list.append(mean_caspase)
        median_caspase_list.append(median_caspase)
        image_id.append(image_filename)
        nuclei_id.append(nid1)
        area.append(A)
        pleo.append(sol)
        elong.append(eccin)
        mean1.append(mean11)
        tot1.append(tot11)
        cellu.append(node_data[nid1])
        mask.append(to1.tolist())
        bbox.append(box)

    data = {
        'image_id': image_id,
        'nuclei_id': nuclei_id,
        'area': area,
        'pleomorphism': pleo,
        'elongation': elong,
        'mean_intensity_DAPI': mean1,
        'total_intensity_DAPI': tot1,
        'cellularity': cellu,
        'mask': mask,
        'Bounding_Box': bbox,
        'mean_cd3': mean_cd3_list,
        'median_cd3': median_cd3_list,
        'mean_psmad': mean_psmad_list,
        'median_psmad': median_psmad_list,
        'mean_cytoplasmic_cd8': mean_cd8_list,
        'median_cytoplasmic_cd8': median_cd8_list,
        'mean_nuclear_cd8': mean_cd8_nuc,
        'median_nuclear_cd8': median_cd8_nuc,
        'mean_ki67': mean_ki67_list,
        'median_ki67': median_ki67_list,
        'mean_caspase': mean_caspase_list,
        'median_caspase': median_caspase_list
    }

    df = pd.DataFrame(data)
    return df


def main():
    image_dir = '/nfs/cc-filer/home/sabulikailik/images/A1818'
    mask_dir = '/nfs/cc-filer/home/sabulikailik/images/A1818/UNet_mask18/'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.czi')]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tasks = [pool.apply_async(process_image, args=(image_file, image_dir, mask_dir)) for image_file in image_files]
        results = [task.get() for task in tasks]  # Collect results

    # Concatenate all DataFrames
    df_combined = pd.concat(results, ignore_index=True)

    # Save the DataFrame to an Excel file
    df_combined.to_excel('Processed_Images_Data18.xlsx', index=False)
    print("Processing complete. Data saved to 'Processed_Images_Data18.xlsx'.")

if __name__ == "__main__":
    main()