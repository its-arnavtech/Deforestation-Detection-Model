import importlib
import sys
import time

# Robust dynamic import for Earth Engine ('ee') with helpful guidance when missing
try:
    ee = importlib.import_module('ee')
except ModuleNotFoundError:
    print("ERROR: Python executable:", sys.executable)
    print("The 'ee' module (earthengine-api) is not installed in this Python environment.")
    print("Install with: pip install earthengine-api")
    print("Or run this script using the conda environment where it's installed.")
    raise

# initialize earth engine (will attempt interactive authentication if needed)
try:
    ee.Initialize(project='deforestation-detection-model')
except Exception as e:
    print("Earth Engine initialization failed:", e)
    print("Attempting interactive authentication now...")
    try:
        ee.Authenticate()
        ee.Initialize()
    except Exception as e2:
        print("Authentication / initialization still failed:", e2)
        print("Please run: earthengine authenticate --quiet or follow: http://goo.gle/ee-auth")
        raise

#user defined params
#region of interest (ROI) for processing is defined, here, it is amazon rainforest
ROI = ee.Geometry.Polygon([
    [[-61.5, -3.5], [-61.5, -3.0], [-61.0, -3.0], [-61.0, -3.5], [-61.5, -3.5]]
])
#included in user defined params
start_date = '2022-01-01'
end_date = '2024-01-15'
max_cloud_percent = 20  #include images with less than 20% cloud cover
#folder and filenames for export
export_folder = 'Deforestation_week1'
export_name = 'nvdi_median_amazon'
export_scale = 10 #resolution in meters per pixel
max_pixels = 1e13 #max numbers of pixels to export 

#cloud masking helper function
def mask_s2_sr(img):  #function that applies cloud mask to sentinel2 surface reflectance images
    scl = img.select('SCL') #scl = scene classification layer, scl has values corresponding to surface features
                            #including clouds and cloud shadows
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)) #creates a mask that includes pixels classified
    #where 4=cloud value, 5=cloud shadow, 6=snow, 7=saturated
    return img.updateMask(mask) #returning mask

# FIX: Earth Engine uses .or(), not .Or()
def mask_s2_sr(img):
    scl = img.select('SCL')
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return img.updateMask(mask)

#loading sentinel-2 Surface Reflectance Collection
collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')  #load sentinel-2 SR imagecollection
    .filterBounds(ROI) #filter images by ROI
    .filterDate(start_date, end_date)  #filter images by date range
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_percent))  #filter images by cloud coverage
    .map(mask_s2_sr)  #apply cloud masking function to image collection
)

#get count of images in filtered section
count = collection.size().getInfo()
print(f"Images we found: {count}")

if count==0:
    raise SystemExit("No images found please ajdust ROI") #exit if no images were found

#image processing
#create median composite image to reduce effect of clouds
median = collection.median().clip(ROI)

#ndvi calculation(normalized difference vegetation index) to get vegetation health
ndvi = median.normalizedDifference(['B8','B4']).rename('NVDI') #using the NIR(B8) and red(B4) bands to calculate nvdi

#thumbnail preview
#params to generate thumbnail
thumb_params = {
    'min' : -1, #min value for thumbnail colourscale
    'max' : 1, #max value of thumbmail colourscale
    'palette' : ['brown', 'yellow', 'green'],
    'region' : ROI, #region to make thumbnail for
    'dimensions' : 512  #size of thumbnail(512px^2)
}

#generate url for nvdi thumbnail
url = ndvi.getThumbURL(thumb_params)
print("\n🌿 NDVI Preview URL:\n", url, "\n(Open this in your browser.)")  #print the URL for the thumbnail preview

#export ndvi to google drive
task = ee.batch.Export.image.toDrive(
    image=ndvi.toFloat(), #convert ndvi img to float data type
    description=export_name, #description of export task
    folder=export_folder, #folder in google drive where img saves
    fileNamePrefix=export_name, #prefix for filename
    region=ROI.bounds().getInfo()['coordinates'], #region to export
    scale=export_scale, #resolution of exported img(in meters per pixel)
    maxPixels=int(max_pixels)  #max pixels allowed per export
)

task.start()
print("🚀 Export started... (this runs on Google servers)")  #print a message that the export has started

#poll task status
#following loop checks status of export task and if it's done
while True:
    status = task.status()
    state = status.get('state')
    print("Task state: ", state)
    if state in ['COMPLETED', 'FAILED', 'CANCELLED']:  # Exit the loop if the task is finished, failed, or canceled
        break
    time.sleep(10) #wait 10 secs before checking task status again

#print final status and result
print("\n Final status:", task.status())  #print the final task status
print(f"If COMPLETED, check your Google Drive folder: {export_folder}")  #notify the user where to find the result in Google Drive
