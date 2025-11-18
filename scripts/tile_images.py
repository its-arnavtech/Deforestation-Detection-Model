import os
import rasterio
import numpy as np
from rasterio.windows import Window

Raw_Dir = "data/raw_sentinel"  #define dir path where raw sentinel satellite images are stored
Tile_Dir = "data/tiles_sentinel" #define dir path where output tiles are saved
Tile_size = 512 #size of each tile(512x512)

os.makedirs(Tile_Dir, exist_ok=True) #create output dir if it doesn't exist, exist_ok=True prevents errors if they happen

def tile_img(path):
    name = os.path.splitext(os.path.basename(path))[0] #extract filename without extension
    with rasterio.open(path) as src: #open raster file in read mode
        width, height = src.width, src.height #get w and h of src img in pixels
        #loop thru img height in steps of tile size, i represents the y coordinate(row positiion)
        for i in range(0, height, Tile_size):
            #loop thru img width, i reps the x coordinate(coloumn)
            for j in range(0, width, Tile_size):
                window = Window(j, i, Tile_size, Tile_size) #create window object defining rectangular region starting at (j, i) with tile size 
                transform = src.window_transform(window) #calculate geospatial transform for specific tile, maintaining correct geographic coordinates for the tile
                tile = src.read(window=window) #read pixel data from src img within defined window and return np arr w shape(bands, h, w)

                #check if tile has full dimensions and skip if incomplete tiles 
                if tile.shape[1] != Tile_size or tile.shape[2] != Tile_size:
                    continue

                #construct output filename using original filename and coords
                tile_path = os.path.join(Tile_Dir, f"{name}_{i}_{j}.tif")

                #open new raster file to write tile data
                with rasterio.open(
                    tile_path, #path where tile is saved
                    "w", #open in write mode
                    driver="GTiff", #use geotiff format
                    h = Tile_size, #set height to 512
                    w = Tile_size, #same for width
                    count=src.count, #copy num of bands from src img
                    dtype=tile.dtype, #copy dt from src img
                    crs = src.crs, #copy coord reference system from src                    
                    transform=transform #set geospatial transform for this tile
                ) as dst:
                    #write tile data to output file
                    dst.write(tile)

                print("Saved", tile_path) #print confirmation that tile was saved

#check if script is being run directly
if __name__ == "__main__":
    #get list of all files in raw_dir that end w dt=tif
    files = [f for f in os.listdir(Raw_Dir) if f.endswith(".tif")]

    for f in files: #loop thru each .tif file
        tile_img(os.path.join(Raw_Dir, f)) #call tile_img func w full path to each file