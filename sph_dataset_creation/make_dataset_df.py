import pandas as pd
import sys
import os

from pdb import set_trace as pause


dataset_dir = sys.argv[1]
dataset_name = sys.argv[2]

classes = os.listdir(dataset_dir)

df = pd.DataFrame(columns=['file','class', 'x_rot', 'y_rot'])

for clas in classes:
	files = [os.path.join(clas,file) for file in os.listdir(os.path.join(dataset_dir,clas))]
	x = [float(file.split('__')[1]) for file in files]
	y = [float(file.split('__')[2]) for file in files]

	df = df.append(pd.DataFrame({'file':files, 
                                     'class':[clas]*len(files),
                                     'x_rot':x,
                                     'y_rot':y,}),ignore_index=True)


df.to_csv('{}.csv'.format(dataset_name))



