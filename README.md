# Spherical-Transformer-for-classification-on-sphericalimages

## Dependencies

Tangent_images code requires the installation of my Spherical Distortion Package, [which can be found here](https://github.com/meder411/Spherical-Package). Installation instructions are available in the linked repository.

Then install requirements with:

pip install -r requirements.txt

## Train

python3 train.py --data_dir <path_to_dataset> --dataset <dvsc or smnist>  --mode <normal or face or vertex or regular> 

## Test

python3 test.py --data_dir <path_to_dataset> --dataset <dvsc or smnist> --resume <path to the model> 
  
## Datasets
  
  Dataset can found here.
