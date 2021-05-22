clear all
addpath('PanoBasic/Projection')

imagenet_dir = '~/datasets/cats_vs_dogs/train';
proj_imgnet_dir = '~/datasets/cats_vs_dogs/sph_dogs_vs_cats'

classes = dir(imagenet_dir);
classes = classes(3:numel(classes));

fov = pi/1.5;
W = 512;
H = 256;
num_projections = 10

for class_idx = 1:numel(classes)
	image_files = dir(fullfile(imagenet_dir, classes(class_idx).name));
	image_files = image_files(3:numel(image_files));

	mkdir(fullfile( proj_imgnet_dir, classes(class_idx).name))
	
	for file_idx = 1:numel(image_files)
		file_name = fullfile(imagenet_dir, classes(class_idx).name, image_files(file_idx).name);
		img = imread(file_name);
		
		[rows, cols, chs] = size(img);

		X = -pi + (pi + pi) .* rand(1,num_projections);
		Y = -pi/2 + (pi/2 + pi/2) .* rand(1,num_projections);

		for proj_id = 1:num_projections
			x = X(proj_id);
			y = Y(proj_id);

			[sphImg, validMap] = im2Sphere(im2double(img), fov, W, H, x, y);
			projected_image = sphImg .* validMap;

			if rows > 2 * cols
				imshow(projected_image)
			end
			output_imgname = strcat(image_files(file_idx).name(1:numel(image_files(file_idx).name())-4),'__',num2str(x),'__',num2str(y),'__.jpg');
			output_fname = fullfile(proj_imgnet_dir, classes(class_idx).name, output_imgname)

			imwrite(projected_image, output_fname)
		end
		

	end
end

break



%{
filename = 'projection.gif'

for i = 1:length(projected_images)
	[A,map] = rgb2ind(projected_images{i},256);
	if i == 1
		imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',0.2);
	else
		imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',0.2);
	end
end
%}
