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

#region of interest (ROI) for processing is defined, here, it is amazon rainforest
ROI = ee.Geometry.Polygon([
    [[-61.5, -3.5], [-61.5, -3.0], [-61.0, -3.0], [-61.0, -3.5], [-61.5, -3.5]]
])

start_date = '2022-01-01'
end_date = '2024-01-15'
max_cloud_percent = 20
export_folder = 'Deforestation_week1'
export_name = 'sentinel10bands_median_amazon'
export_scale = 10
max_pixels = 1e13

#cloud masking helper function
def mask_s2_sr(img):  #function that applies cloud mask to sentinel2 surface reflectance images
    scl = img.select('SCL') #scl = scene classification layer, scl has values corresponding to surface features
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return img.updateMask(mask) #returning mask

#load sentinel-2 Surface Reflectance Collection
collection = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(ROI)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud_percent))
    .map(mask_s2_sr)
)

count = collection.size().getInfo()
print(f"Images we found: {count}")

if count == 0:
    raise SystemExit("No images found please adjust ROI")

#select 11 useful bands (10 spectral + classification)
bands = [
    'B1','B2','B3','B4','B5','B6','B7','B8','B11','B12','SCL'
]

median = collection.median().select(bands).clip(ROI)

#export all bands
task = ee.batch.Export.image.toDrive(
    image=median.toFloat(),
    description=export_name,
    folder=export_folder,
    fileNamePrefix=export_name,
    region=ROI.bounds().getInfo()['coordinates'],
    scale=export_scale,
    maxPixels=int(max_pixels)
)

task.start()
print("🚀 Export started... (running on Google servers)")

while True:
    status = task.status()
    state = status.get('state')
    print("Task state:", state)
    if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
        break
    time.sleep(10)

print("\nFinal status:", task.status())
print(f"If COMPLETED, check your Google Drive folder: {export_folder}")
