import numpy as np
import glob
import pathlib
import nilearn.datasets
from nilearn.image import load_img
from nilearn.maskers import NiftiMapsMasker, NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
from nilearn.interfaces.fmriprep import load_confounds

# Defining functions for generating FC matrix and testing the validity of the FC matrix from a path to 1 nifty file

def MaskerBuilder(atlas):
    atlas_filename = atlas["maps"]
    labels = atlas["labels"]
    masker = NiftiMapsMasker(
        maps_img=atlas_filename,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory="nilearn_cache",
    )
    return masker

def MaskReportGenerator(masker):
    report = masker.generate_report(displayed_maps=[2, 6, 7, 16, 21])
    return report

def TimeSeriesExtractor(img_path, masker):
    nifti_path = img_path
    fmri_img = load_img(nifti_path)
    confounds_simple, sample_mask = load_confounds(
        nifti_path,
        strategy=["high_pass", "motion", "wm_csf"],
        motion="basic",
        wm_csf="basic",
    )
    time_series = masker.fit_transform(fmri_img, confounds=confounds_simple, sample_mask=sample_mask)
    return time_series

def FCMGenerator(img_path, masker):
    time_series = TimeSeriesExtractor(img_path=img_path, masker=masker)
    correlation_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    np.fill_diagonal(correlation_matrix, 0)
    return correlation_matrix

# Fucntions used for visualization
def PLotFCM(FCM, labels = None, colorbar_True_or_False = True, vmax = 0.8, vmin = -0.8):
    plotting.plot_matrix(
        FCM, labels=labels, colorbar=colorbar_True_or_False, vmax=vmax, vmin=vmin, 
    )
    print(FCM.shape)

def PlotCoordinates(atlas, FCM, dim):
    coordinates = plotting.find_probabilistic_atlas_cut_coords(
        maps_img=atlas.maps
    )

    # plot connectome with 85% edge strength in the connectivity
    plotting.plot_connectome(
        FCM,
        coordinates,
        edge_threshold="85%",
        title=f"DiFuMo with {dim} dimensions (probabilistic)",
    )
    plotting.show()

def PlotWebView(atlas, FCM):
    coordinates = plotting.find_probabilistic_atlas_cut_coords(
        maps_img=atlas.maps
    )
    view = plotting.view_connectome(
        FCM, coordinates, edge_threshold="80%"
    )
    view.save_as_html('Connectome_Webview.html')
    print("Check the folder for file name \'Connectome_Webview.html\' and click on it.")

# Function for loading all nifty files within a directory path
def GetNiftyPaths(dir_path):
    return list(pathlib.Path(dir_path).glob('*.nii.gz'))

# Functions for saving/loading Functional Connectivity Matrix to/from a path
def SaveFCM(output_dir, output_file_name, FCM):
    full_path = output_dir / output_file_name
    print("File will be saved at:", full_path)
    np.savetxt(full_path, FCM)
    b = np.loadtxt(full_path)
    return FCM == b

def LoadFCM(mattrix_path):
    FCM = np.loadtxt(mattrix_path)
    return FCM

