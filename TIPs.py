import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from skimage import morphology
from skimage import measure
import argparse
import nibabel as nib
import shutil

parser = argparse.ArgumentParser(description='Process files in a folder and calculate centroid.')

# Add argument for the folder path
parser.add_argument('folder', type=str, help='The path to the folder containing files.')

# Add argument for centroid calculation method
parser.add_argument('-c', action='store_false', help='Disable automatic centroid calculation (default is enabled).')

# Parse the arguments
args = parser.parse_args()
os.environ['nnUNet_results'] = 'nnResults/'
# Extract the folder base name
folder_base_name = args.folder

# Determine the centroid calculation method
manual_input = not args.c
    
 
os.environ['MKL_THREADING_LAYER'] = 'GNU' 
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


dst_dit_list = [folder_base_name+"_resample_teeth_input", folder_base_name+"_resample_teeth_binary",
                folder_base_name+"_resample_pulps_input"]
for path in dst_dit_list:
    if not os.path.exists(path):
        os.makedirs(path)
        
        
# Resample to 0.3
print("Resampling to 0.3")
def resample_to_isotropic(input_path, output_path, new_spacing=(0.3, 0.3, 0.3)):
    image = sitk.ReadImage(input_path)
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resampled_image = resample.Execute(image)
    # sitk.WriteImage(resampled_image, output_path)
    # Check if any dimension of the resampled image is greater than 600
    if all(dim <= 600 for dim in resampled_image.GetSize()):
        # Save the image if all dimensions are within the limit
        sitk.WriteImage(resampled_image, output_path)
        print(f"Resampled file saved at: {output_path}")
    else:
        # Print a message if any dimension exceeds the limit
        
        print(f"Resampled image {output_path} exceeds dimension limits (600). Skipping. Consider to crop the image and remove the unnecessary region.")
        

input_folder = folder_base_name
output_folder = folder_base_name + "_resample_teeth_input"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".nii.gz"):
        input_file = os.path.join(input_folder, filename)
        
        # Handle the filename properly by removing ".nii.gz" and adding "_0000.nii.gz"
        base_filename = filename[:-7]  # Remove the last 7 characters (".nii.gz")
        output_filename = base_filename + "_0000.nii.gz"  # Add the "_0000.nii.gz" suffix
        output_file = os.path.join(output_folder, output_filename)
        
        resample_to_isotropic(input_file, output_file)

# Teeth Segmentation
print("Processing Teeth Segmentation.")
os.system('nnUNetv2_predict  -i  '+folder_base_name+"_resample_teeth_input"+'  -o  '+ folder_base_name+"_resample_teeth_binary"+'  -d    803   -c    3d_fullres    -f   all   -tr  nnUNetTrainerUMambaBot   -chk  checkpoint_best.pth --c  -npp=1 -nps=1 ' )



def calculate_centroid(binary_image):
    indices = np.argwhere(binary_image)
    centroid_x = indices[:, 0].mean()
    return centroid_x

def create_offset_map_x(binary_image, centroid_x):
    offset_map_x = np.zeros(binary_image.shape)

    for x in range(binary_image.shape[0]):
        for y in range(binary_image.shape[1]):
            for z in range(binary_image.shape[2]):
                if binary_image[x, y, z] > 0:
                    offset_map_x[x, y, z] = x - centroid_x

    return offset_map_x

def process_file(file_path, manual_input, output_path):
    img = nib.load(file_path)
    data = img.get_fdata()

    centroid_x = calculate_centroid(data) if not manual_input else float(input(f"Enter the X coordinate of the centroid for {os.path.basename(file_path)}: "))
    offset_map_x = create_offset_map_x(data, centroid_x)

    new_filename = os.path.basename(file_path).replace('.nii.gz', '_0001.nii.gz')
    offset_img = nib.Nifti1Image(offset_map_x, img.affine)
    nib.save(offset_img, os.path.join(output_path, new_filename))

def process_folder(folder_path, manual_input, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(folder_path):
        if filename.endswith('.nii.gz'):
            process_file(os.path.join(folder_path, filename), manual_input, output_path)


print('Generating Prior Maps.')
# Ask the user for centroid calculation method
# centroid_method = input("Do you want to use the automatic centroid calculation? (yes/no): ").strip().lower()
manual_input = not args.c
if manual_input:
    print("Manual centroid calculation selected.")
else:
    print("Automatic centroid calculation selected.")
    

folder_path = folder_base_name+"_resample_teeth_binary" # Replace with your folder path

input_folder = folder_path
distance_maps_folder1 = folder_base_name+"_resample_teeth_input"
distance_maps_folder2 = folder_base_name+"_resample_pulps_input"  
process_folder(folder_path, manual_input, distance_maps_folder1)


def load_image(file_path):

    return sitk.ReadImage(file_path)

def connected_components(image):
    return sitk.ConnectedComponent(image)

def save_image(image, filename):

    sitk.WriteImage(image, filename, True)  # True for compression
    
def convert_to_int(image):

    return sitk.Cast(image, sitk.sitkUInt32)
def erode_image(image, radius):

    return sitk.BinaryErode(image, sitk.VectorUInt32([radius, radius, radius]))
def dilate_image(image, radius):

    return sitk.BinaryDilate(image, sitk.VectorUInt32([radius, radius, radius]))

def open_image(image, radius):

    return sitk.BinaryMorphologicalOpening(image, sitk.VectorUInt32([radius, radius, radius]))
def generate_signed_distance_map(binary_image):

    return sitk.SignedMaurerDistanceMap(binary_image, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)

def threshold_negative_values(image):

    return sitk.Threshold(image, lower=-1e10, upper=0, outsideValue=0)
def convert_negative_to_positive(image):

    return sitk.Abs(image)
def resample_to_reference(image, reference_image):
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)


# if not os.path.exists(distance_maps_folder):
#     os.makedirs(distance_maps_folder)   

for file in os.listdir(input_folder):
    file_id = file.rstrip('.nii.gz')
    print("file_id:",file_id)
    if file.endswith('.nii.gz'):
        file_path = os.path.join(input_folder, file)
        label_image = load_image(file_path)
        label_image_int = convert_to_int(label_image)
        
        for erode_param in [1,2,3]:
            

            eroded_image = erode_image(label_image_int, erode_param)  
            connected_components_image = connected_components(eroded_image)
        
            label_stats = sitk.LabelShapeStatisticsImageFilter()
            label_stats.Execute(connected_components_image)
            labels = label_stats.GetLabels()
            print("components",len(labels))
            if len(labels)>25:
                break

        output_folder = "separated"  
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for label in labels:
            if label == 0:  
                continue
            single_object = sitk.BinaryThreshold(connected_components_image, label, label, 1, 0)
            single_object = dilate_image(single_object, 2) 
            output_filename = os.path.join(output_folder, f"{file.rstrip('.nii.gz')}_object_{label}.nii.gz")
            save_image(single_object, output_filename)
        
       
        accumulator = None
        reference_image = label_image
        
        
        segmented_files = [f for f in os.listdir(output_folder) if f.endswith('.nii.gz')]
        
        for file in segmented_files:
            file_path = os.path.join(output_folder, file)
            binary_image = load_image(file_path)
            distance_map = generate_signed_distance_map(binary_image)
            negative_distance_map = threshold_negative_values(distance_map)
            positive_distance_map = convert_negative_to_positive(negative_distance_map)
            positive_distance_map = resample_to_reference(positive_distance_map, reference_image)                    
            if accumulator is None:
                accumulator = positive_distance_map
            else:
                accumulator += positive_distance_map
        folder_to_delete = "separated"
        if os.path.exists(folder_to_delete):
            shutil.rmtree(folder_to_delete)
        final_map_file = os.path.join(distance_maps_folder2, file_id + "_0001.nii.gz")
        save_image(accumulator, final_map_file)
        
for filename in os.listdir(folder_base_name+"_resample_teeth_input"):
        if filename.endswith("_0000.nii.gz"):
            src_path = os.path.join(folder_base_name+"_resample_teeth_input", filename)
            dst_path = os.path.join(folder_base_name+"_resample_pulps_input" , filename)
            shutil.copy(src_path, dst_path)
   
# os.rename(folder_base_name+"_resample", folder_base_name+"_resample_teeth_input")

    
print("Processing Teeth Instance Segmentation.")
os.system('nnUNetv2_predict -step_size 0.8  -i  '+folder_base_name+"_resample_teeth_input"+'  -o  '+ folder_base_name+"_resample_teeth_instance"+' -d  812    -c    3d_fullres    -f   all  -tr  nnUNetTrainerUMambaBot   -chk  checkpoint_best.pth  -npp=1 -nps=1 --c ' )

print("Renumber the instance")       
for file_name in os.listdir(folder_base_name+"_resample_teeth_instance"):
    if file_name.endswith('.nii.gz'):
        file_path = os.path.join(folder_base_name+"_resample_teeth_instance", file_name)
        img = nib.load(file_path)
        img_data = img.get_fdata()
        img_data = img_data.astype(np.float64)
        img_data[(img_data >= 21) & (img_data <= 28)] += 20
        img_data[(img_data >= 1) & (img_data <= 8)] += 20
        img_data_uint8 = img_data.astype(np.uint8)
        output_file_path = os.path.join(folder_base_name+"_resample_teeth_instance", file_name)
        new_img = nib.Nifti1Image(img_data_uint8, img.affine, img.header)
        nib.save(new_img, output_file_path)
        
print("Processing Pulps Segmentation.")        
os.system('nnUNetv2_predict -step_size  0.8  -i  '+folder_base_name+"_resample_pulps_input"+'  -o  '+ folder_base_name+"_resample_pulps_segmentation"+' -d 810    -c    3d_fullres    -f   all  -tr  nnUNetTrainerUMambaBot   -chk  checkpoint_best.pth -npp=1 -nps=1 --c ' )

# Define the directories
input_dir_pulp = folder_base_name+"_resample_pulps_segmentation"
input_dir_toothlabel = folder_base_name+"_resample_teeth_instance"
output_dir = folder_base_name+"_resample_pulps_instance"
os.makedirs(output_dir, exist_ok=True)

pulp_files = [f for f in os.listdir(input_dir_pulp) if f.endswith('.nii.gz')]


for pulp_file in pulp_files:

    pulp_path = os.path.join(input_dir_pulp, pulp_file)
    toothlabel_path = os.path.join(input_dir_toothlabel, pulp_file)
    output_path = os.path.join(output_dir, pulp_file)
    
    if os.path.exists(toothlabel_path):
        pulp_img = nib.load(pulp_path)
        toothlabel_img = nib.load(toothlabel_path)
        

        pulp_data = pulp_img.get_fdata()
        toothlabel_data = toothlabel_img.get_fdata()
        
        result_data = np.zeros_like(pulp_data)
        result_data[pulp_data == 1] = toothlabel_data[pulp_data == 1]
        result_data = result_data.astype(np.int32)

        result_img = nib.Nifti1Image(result_data, pulp_img.affine, pulp_img.header)
        nib.save(result_img, output_path)
        
        