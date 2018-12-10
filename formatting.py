from PIL import Image
from array import array
from numpy import *

def imageToMatrix(imagePath, outputName, matrixSize = (28, 28)):
	
	data = zeros(matrixSize)

	image = Image.open(imagePath)
	pixels = image.load()

	(width, height) = image.size

	point_x_min = width
	point_y_min = height
	point_x_max = 0
	point_y_max = 0

	for x in range(width):
		for y in range(height):
			if (pixels[x, y][0] != 255):
				if (x < point_x_min):
					point_x_min = x
				if (y < point_y_min):
					point_y_min = y
				if (x > point_x_max):
					point_x_max = x
				if (y > point_y_max):
					point_y_max = y

	area = (point_x_min, point_y_min, point_x_max, point_y_max)
	print(area)

	new_width = point_x_max - point_x_min
	new_height = point_y_max - point_y_min
	print(new_width)
	print(new_height)

	size = 0

	if (new_width > new_height):
		point_y_min = point_y_min - (new_width - new_height) / 2
		point_y_max = point_y_max + (new_width - new_height) / 2
		size = new_width
	else:
		point_x_min = point_x_min - (new_height - new_width) / 2
		point_x_max = point_x_max + (new_height - new_width) / 2
		size = new_height

	area = (point_x_min, point_y_min, point_x_max, point_y_max)
	print(area)

	image_cropped = image.crop(area).resize(matrixSize, Image.ANTIALIAS)
	pixels = image_cropped.load()

	for x in range(matrixSize[0]):
		for y in range(matrixSize[1]):
			data[x, y] = pixels[y, x][0]

	savetxt(outputName, data, "%4i")
