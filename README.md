# Spherical-Transformer-for-classification-on-sphericalimages

## Dependencies

Tangent_images code requires the installation of my Spherical Distortion Package, [which can be found here](https://github.com/meder411/Spherical-Package). Installation instructions are available in the linked repository.

Then install requirements with:

pip install -r requirements.txt

## Train

python3 train.py --data_dir <path_to_dataset> --dataset <dvsc_or_smnist>  --mode <normal_or_face_or_vertex_or_regular> 

## Test

python3 test.py --data_dir <path_to_dataset> --dataset <dvsc_or_smnist> --resume <path_to_the_model>
  
## Datasets
  
  Dataset can found here.
