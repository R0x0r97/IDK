from PIL import Image
from array import array
from numpy import savetxt
from numpy import zeros
from math import floor

def isBlank(imagePath):
	image = Image.open(imagePath)
	pixels = image.load()

	for x in range(image.size[0]):
		for y in range(image.size[1]):
			if (pixels[x, y][0] != 255):
				return False
	return True

def applyAntiAlias(image):
	(width, height) = image.size
	image = image.resize((width * 2, height * 2))
	image = image.resize((width, height), Image.ANTIALIAS)
	return image

def trim(image, borderScale):
	(width, height) = image.size
	pixels = image.load()

	topLeft_x = width
	topLeft_y = height
	bottomRight_x = 0
	bottomRight_y = 0

	for x in range(width):
		for y in range(height):
			if (pixels[x, y][0] != 255):
				if (x < topLeft_x):
					topLeft_x = x
				if (y < topLeft_y):
					topLeft_y = y
				if (x > bottomRight_x):
					bottomRight_x = x
				if (y > bottomRight_y):
					bottomRight_y = y

	new_width = bottomRight_x - topLeft_x
	new_height = bottomRight_y - topLeft_y
	size = 0

	if (new_width > new_height):
		topLeft_y = topLeft_y - (new_width - new_height) / 2
		bottomRight_y = bottomRight_y + (new_width - new_height) / 2
		size = new_width
	else:
		topLeft_x = topLeft_x - (new_height - new_width) / 2
		bottomRight_x = bottomRight_x + (new_height - new_width) / 2
		size = new_height

	border = floor(min(topLeft_x, topLeft_y, width - bottomRight_x, height - bottomRight_y))

	if (border > floor(size / borderScale)):
		border = floor(size / borderScale)

	area = (topLeft_x - border, topLeft_y - border, bottomRight_x + border, bottomRight_y + border)

	return image.crop(area)

def PNGToIDX(imagePath, outputName, matrixSize = (28, 28)):
	
	data = zeros(matrixSize)

	image = trim(applyAntiAlias(Image.open(imagePath)), 10).resize(matrixSize)
	image.show()
	pixels = image.load()

	for x in range(matrixSize[0]):
		for y in range(matrixSize[1]):
			data[x, y] = pixels[y, x][0]

	savetxt(outputName, data, "%4i")