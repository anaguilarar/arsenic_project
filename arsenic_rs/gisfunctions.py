### gisfunctions
def export_astiff(arraydata, metadata, path = "", file_name = "temp.tif", d_type = 'uint16'):
        
    import rasterio
    import os
    
    ## check Na values
    if np.sum(np.isnan(arraydata)) > 0:
        arraydata[np.isnan(arraydata)] = 0
    
    ## reshape 
    if len(arraydata.shape) == 2:
        arraydata = arraydata.reshape(1, arraydata.shape[0],arraydata.shape[1])
        nlayers = 1
    else:
        nlayers = arraydata.shape[0]
    
    
    ## update metadata
    
    metadata.update(count = nlayers, 
                       nodata = 0,
                   dtype = d_type)
    if d_type == 'uint16':
        convert = np.uint16
    if d_type == 'float32':
        convert = np.float32
    
    with rasterio.open(os.path.join(path, file_name), "w", **metadata) as dest:
        dest.write(arraydata.astype(convert))
        

def fromrasterio_toflatten(rasterarray, tofloat16 = True, selectbands = None):
    
    if selectbands is not None:
        bandstokeep = selectbands
    else:
        bandstokeep =[i for i in range(1,rasterarray.read().shape[0]+1)]
    
    data_organizedperband = []
    
    for i in bandstokeep:
        if rasterarray.meta['dtype'] == 'float32' and tofloat16:
            dataraster =  (rasterarray.read(i)*1000).astype(np.float16)
        else:
            dataraster = rasterarray.read(i)
        data_organizedperband.append(dataraster.ravel()
                                  )
    
    return(np.array(data_organizedperband).T)

def maskdata_usingimage (dataraster, pathtomask, maskvalue = 0):
    ## read raster
    
    datamask = rasterio.open(pathtomask).read(1)
    ## mask data frame
    dataraster[datamask==maskvalue] = np.nan
    
    return dataraster

def maskdataframe_usingimage (dataframe, pathtomask, maskvalue = 0): 
    
    ## read raster
    datamask = rasterio.open(pathtomask).read(1)
    ## mask data frame
    dataframe.loc[datamask.ravel()==maskvalue] = np.nan
    return dataframe