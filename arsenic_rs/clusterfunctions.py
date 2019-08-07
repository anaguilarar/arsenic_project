from arsenic_rs import gisfunctions
import os
import matplotlib.pyplot as plt
import pickle
import math
import pandas as pd
import numpy as np


def elbowplot(datatocluster ,totalclusters = 12):
        
        Sum_of_squared_distances = []
        K = range(1,totalclusters)
        for k in K:
            print(k)
            km = KMeans(n_clusters=k, random_state=0)
            km = km.fit(datatocluster)
            Sum_of_squared_distances.append(km.inertia_)
        
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

def calculate_kmeans(data_tocluster,  ncluster = 5,robustscalar =True,
                     pca_reduction = False, elbow = False):
    
    
         ## remove NA values
    aux_data = data_tocluster.dropna()
   
    
    if robustscalar:
        print('robustscalar')
        transformer = RobustScaler().fit(aux_data)
        aux_data = transformer.transform(aux_data)

    if pca_reduction:
        print('pca')
        pca = PCA(.90)
        pca = pca.fit(aux_data)
        aux_data = pca.transform(aux_data)
        
    
    ## elbow method
    if elbow:
        elbowplot(aux_data)

    else:
        print('k-means')
        
        km = KMeans(n_clusters=ncluster, random_state=0)
        kmeans = km.fit(aux_data)
        
        if pca:
            return [transformer, pca, kmeans]
        
        elif robustscalar:
            return [transformer,kmeans]
        
        else:
            return [kmeans]


def kmeans_tworegions(rasteriodata,cluster= None, subsample = None,robustscalar=True,pca_reduction=True,elbow= False):

    totaldata = np.array([np.size(rasteriodata[i][1].read(1)) for i in range(len(rasteriodata))]).sum()


    if subsample is not None:
        subsample = int(np.round(totaldata*(subsample/100)))
        print("total pixels to sample: " + str(subsample))
        idx = np.random.randint(totaldata, size=subsample)
    else:
        idx = [i for i in range(totaldata)]


    data_tocluster = pd.concat(
        [pd.concat([pd.DataFrame(gisfunctions.fromrasterio_toflatten(rasteriodata[i][j])) 
                                 for j in range(len(rasteriodata[i]))], axis = 1) 
         for i in range(len(rasteriodata))], axis = 0).iloc[idx]
 

    if elbow:
        calculate_kmeans(data_tocluster,robustscalar= robustscalar,pca_reduction= pca_reduction, elbow=elbow)
        return np.nan

    else:
        return calculate_kmeans(data_tocluster, ncluster=cluster,  robustscalar= robustscalar,pca_reduction= pca_reduction)

    
def fromarraytodataframe(dataarray, colnames = None):
    '''transform rasterio array to pandas dataframe'''
    shape_data = dataarray.shape
    if shape_data[0] > shape_data[2]:
        dataarray = dataarray.swapaxes(1,2)
        dataarray = dataarray.swapaxes(0,1)
        
    dataflat = []
    for i in range(dataarray.shape[0]):
        dataflat.append(dataarray[i,:].ravel())
    
    if colnames is None:
        
        dataflatten = pd.DataFrame(np.array(dataflat).T)
    else:
        dataflatten = pd.DataFrame(np.array(dataflat).T,columns= colnames)
    return dataflatten


def classify_oneregion(rasteriodata,kmeansmodel, robustscale = None,pcareduction = None):
    
    data_tocluster = pd.concat([pd.DataFrame(gisfunctions.fromrasterio_toflatten(rasteriodata[j])) 
                         for j in range(len(rasteriodata))], axis = 1)


    ## transform the matrix using the previous models
    data_tocluster = data_tocluster.dropna()

    indexwithoutna = data_tocluster.index.values
    if robustscale is not None:
        data_tocluster = robustscale.transform(data_tocluster)
        print("robust")
        
    if pcareduction is not None:
        data_tocluster = pcareduction.transform(data_tocluster)
        print("pca")
        
    uc_labels = kmeansmodel.predict(data_tocluster)
    refimage = rasteriodata[0].read(1).copy()
    refimage[:] = 0
    refimage.ravel()[indexwithoutna] = uc_labels+1

    return refimage