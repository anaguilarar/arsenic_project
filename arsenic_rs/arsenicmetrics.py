import os
import matplotlib.pyplot as plt
import pickle
import math
import pandas as pd
import numpy as np

import geopandas as gpd
import rasterio
from affine import Affine
from rasterio.plot import plotting_extent

from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic

from arsenic_rs import clusterfunctions
from arsenic_rs import gisfunctions
from arsenic_rs import arsenicmetrics

### arsenicrs_metrics

def get_imagesegmentation(rasterdata, nsegments, compactness =50, maskimage = None, 
                          rgb_colnames = ['std', 'mean', 'difference']):
    
    '''
    apply image segmentation algorithm to a composite rasterio image
    however the composite image must have the following structure:
    1) standard deviation, 2) average, 3) maximum 4) minimum,
    '''

    ## identify columns for each channel
    ##
    dict_data = {}
    
    if 'std' in rgb_colnames:
        std_values = np.nanmean(pd.concat([pd.DataFrame(gisfunctions.fromrasterio_toflatten(rasterdata[j],selectbands= [1])) 
                                 for j in range(len(rasterdata))], axis = 1), axis = 1)
        dict_data.update({  'std' : std_values })
    
    if 'mean' in rgb_colnames:
        mean_values =  np.nanmean(pd.concat([pd.DataFrame(gisfunctions.fromrasterio_toflatten(rasterdata[j],selectbands= [2])) 
                                 for j in range(len(rasterdata))], axis = 1), axis = 1)
        dict_data.update({ 'mean' : mean_values })
    
    if ('max' in rgb_colnames) | ('difference' in rgb_colnames):
        max_values =  np.nanmean(pd.concat([pd.DataFrame(gisfunctions.fromrasterio_toflatten(rasterdata[j],selectbands= [3])) 
                                 for j in range(len(rasterdata))], axis = 1), axis = 1)
        dict_data.update({  'max' : max_values })
    
    if ('min' in rgb_colnames) | ('difference' in rgb_colnames):
        min_values =  np.nanmean(pd.concat([pd.DataFrame(gisfunctions.fromrasterio_toflatten(rasterdata[j],selectbands= [4])) 
                                 for j in range(len(rasterdata))], axis = 1), axis = 1)
        dict_data.update({  'min' : min_values })

    if ('difference' in rgb_colnames):
        diff_values = max_values - min_values
        dict_data.update( { 'difference' : diff_values })
    
    imageshape = rasterdata[0].read(1).shape
    ## create rgb composite image
    slic_input = np.array([dict_data[rgb_colnames[i]].reshape(imageshape) for i in range(len(rgb_colnames))])
    if maskimage is not None:
        slic_input = np.array([gisfunctions.maskdata_usingimage(slic_input[i,:],maskimage) 
                               for i in range(slic_input.shape[0])])
    ### reshape the array to X , Y , Channel
    slic_input = slic_input.swapaxes(1,0)
    slic_input = slic_input.swapaxes(1,2)
    
    ## scalet the array and apply the SLIC function from the skimage package
    slic_input = scale_minmax(slic_input)
    slic_input[np.isnan(slic_input)] = 0
    segments = slic(slic_input, n_segments = nsegments, compactness  = compactness)
    
    ## plots
    plt.figure(figsize = [14,10])
    plt.subplot(2, 1, 1)
    plt.imshow(mark_boundaries(scale_minmax(slic_input[:,:,:3]),segments))
    plt.axis("off")
    plt.show()
    
    plt.subplot(2, 1, 2)
    plt.imshow(scale_minmax(slic_input[:,:,:3].astype(np.float32)))
    plt.axis("off")
    
    return segments



def get_healthmetrics(rasterdata, superpixels,maskimage = None,band_number = 3):

    df_raster = pd.concat([pd.DataFrame(gisfunctions.fromrasterio_toflatten(rasterdata[j],selectbands= [band_number])) 
                                     for j in range(len(rasterdata))], axis = 1)

    if maskimage is not None:
        df_raster = gisfunctions.maskdataframe_usingimage(df_raster,maskimage)


    df_raster['spID'] = np.ravel(superpixels)
    groups_supper = df_raster.groupby('spID')
    df_raster = df_raster.drop(['spID'], axis = 1)
    
    storelist = [get_cstv(groups_supper,df_raster, i) for i in range(len(groups_supper.groups))]
    storelist = [x for x in storelist if x is not None]

    dataconcatenate = pd.concat(storelist)

    cstvarray = np.zeros(df_raster.shape[0])
    cstvarray[:] = np.nan
    cstvarray[dataconcatenate[0]] = dataconcatenate.cstv.values

    cst_slope_array = np.zeros(df_raster.shape[0])
    cst_slope_array[:] = np.nan
    cst_slope_array[dataconcatenate[0]] = dataconcatenate.cstv_slope.values

    
    return [cstvarray.reshape(rasterdata[0].read(1).shape) , 
            cst_slope_array.reshape(rasterdata[0].read(1).shape) ]


def calculate_aggregation (df, dataindex):
    
    data = pd.DataFrame(dataindex.T)
    data['cstv'] = np.nan
    data['cstv_slope'] = np.nan
    
    variability = np.mean(df.dropna().values, axis = 0)
    
    cstv =  np.mean(variability)

    data['cstv'] = cstv
  
    slope, intercept, r_value, p_value, std_err = stats.linregress([j for j in range(len(variability))],variability)

    data['cstv_slope'] = slope

    return(data)

def get_cstv(groups_df, dataframe, index):
    
    groupindexes = groups_df.groups[index].values
    data_subsample = dataframe.iloc[groupindexes]
    napercentage = data_subsample.dropna().shape[0]/data_subsample.shape[0]
    
    if(napercentage > 0.4):

        return calculate_aggregation(data_subsample,groupindexes)
    else:
        return None