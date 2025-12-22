import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

Tile_dir = "data/tiles_sentinel" #define dir for path where sentinel imgs are stored
Label_dir = "data/labels" #define dir where labels generated will be saved(output)

os.makedirs(Label_dir, exist_ok=True) #create dir for labels and exist_ok prevents errors if dir exists

Hansen_TC = "data/hansen/treecover2000.tif" #values range from 0-100 representing % of tree cover
Hansen_Loss = "data/hansen/lossyear.tif" #0 if no loss, 1-20 for year of loss(2001-2020)

#load hansen layers

#open tree cover geoTiff file
with rasterio.open(Hansen_TC) as src_tc:
    TC_data = src_tc.read(1) #read first and only band of tree cover data into mem
    TC_meta = src_tc.meta #store metadata for later use(transform, CRS, etc)

#open forest loss year geoTiff file
with rasterio.open(Hansen_Loss) as src_loss:
    loss_data = src_loss.read(1) #read first and only band of loss year into mem
    loss_meta = src_loss.meta #store metadata for later use

def reproject_to_tile(tile_path): #func to reproject hansen data to match specific tile's coords system and extent
    with rasterio.open(tile_path) as tile_src: #open sentinel tile img
        profile = tile_src.profile #get profile of tile including metadata

        tile_shape = (profile["height"], profile["width"]) #extract shape of tile
        
        tc_resampled = np.zeros(tile_shape, dtype=np.float32) #create empty arr to store reprojected tree cover data
        #reproject global hansen tree cover data to match tile coords system
        reproject(
            source = TC_data, #src data: global hansen tree cover arr
            destination=tc_resampled, #destination arr to fill w reprojected data
            src_transform=TC_meta["transform"], #geospatial transform of src data
            src_crs=TC_meta["crs"], # coordinate reference system of src data
            dst_transform=profile["transform"], #geospatial transform of destination (tile)
            dst_crs=profile["crs"], #coordinate reference system of destination(file)
            resampling=Resampling.nearest #use nearest neighbour resampling
        )

        #create empty arr to store reprojected forest loss data
        loss_resampled = np.zeros(tile_shape, dtype=np.float32)

        #reproject global hansen forest loss data to match tile's coords system 
        reproject(
            source = loss_data, #src data: global hansen loss year arr
            destination=loss_resampled, #destination arr to fill w reprojected data
            src_transform=loss_meta["transform"], #geospatial transform of src data
            src_crs=loss_meta["crs"], # coordinate reference system of src data
            dst_transform=profile["transform"], #geospatial transform of destination (tile)
            dst_crs=profile["crs"], #coordinate reference system of destination(file)
            resampling=Resampling.nearest #use nearest neighbour resampling
        )
        return tc_resampled, loss_resampled #return both reprojected arrs

def create_label(tile_path): # def func to create labeled mask for given tile
    name = os.path.splitext(os.path.basename(tile_path))[0] #extract filename w/out extension
    tc, loss = reproject_to_tile(tile_path) #get reprojected tree cover and loss data aligned to this tile
    mask = np.zeros_like(tc, dtype=np.uint8) #create empty mask arr w same shape as tile
    forest = tc>30 #create bool arr indicating pixels w tree cover > 30%. this defines what a "forest" is
    deforested = loss>0 #create bool arr indicating pixels where forest loss occured
    mask[forest] = 1 #set forest pixels to class 1
    mask[deforested] = 2 #set deforested pixels to class 2(overwrites forest class where applicable)

    label_path = os.path.join(Label_dir, f"{name}_label.npy") #construct output path for label file
    np.save(label_path, mask) #save mask arr as .npy file
    print("Created Label: ", label_path) #print confirmation label was made

#check if script is run directly and not imported
if __name__ == "__main__":
    tiles = [f for f in os.listdir(Tile_dir) if f.endswith(".tif")]
    for t in tiles:
        create_label(os.path.join(Tile_dir, t)) #create label mask for each tile by calling create label w the full path