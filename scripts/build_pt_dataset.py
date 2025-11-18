import os
import torch
import numpy as np
import rasterio

Tile_dir = "data/tiles_sentinel"
Label_dir = "data/labels"
Out_dir = "data/dataset_pt"

os.makedirs(Out_dir, exist_ok=True)

#define a func to load and normalize satellite img
def load_tile(path):
    with rasterio.open(path) as src: #read bands from raster file
        img = src.read().astype("float32") / 10000.0 #convert to float32 for neural network compatibility and divide by 10k to normalize sen2 reference values
    
    return img #return normalized img arr w any shape

#define func to load corresponding label mask for a tile
def load_label(tile_name):
    #make path to label file by appending "_label.npy" to the tile name
    label_path = os.path.join(Label_dir, f"{tile_name}_label.npy")
    
    #load np arr from .npy file and convert to int64 for pytorch compatibility
    return np.load(label_path).astype("int64")

#check if script is being run directly and not imported as module
if __name__ == "__main__":
    #get list of all .tif files in tile_dir
    tiles = [f for f in os.listdir(Tile_dir) if f.endswith(".tif")]

    for tile in tiles:
        name = os.path.splitext(tile)[0] #extract filename w/out extention
        img = load_tile(os.path.join(Tile_dir, tile)) #load and normalize satellite img tile, returns arr w shape
        mask = load_label(name) #load corresponding label mask and return arr w shape
        img_tensor = torch.from_numpy(img) #convert np img arr to pytorch tensor. shape remains (C, 512, 512) where C is num of spectral bands
        mask_tensor = torch.from_numpy(mask) #convert np mask arr to pytorch tensor and shape remains a 2D arr of class labels
        out_path = os.path.join(Out_dir, name+".pt") #make output file path w .pt extension(pytorch file format)

        #save both img and mask tensors together in a dictionary format creating a single file containing both input img and its label
        torch.save(
            {"image": img_tensor, "mask": mask_tensor}, #dictionary w 2 keys
            out_path #destination file path
        )

        print("Saved: ", out_path)