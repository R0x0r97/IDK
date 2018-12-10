from PIL import Image
from array import array
from numpy import *

def imageToMatrix(imagePath, matrixSize = (28, 28)):
	
	data = zeros(matrixSize)

	Im = Image.open(imagePath).resize(matrixSize)
	pixels = Im.load()

	for x in range(matrixSize[0]):
		for y in range(matrixSize[1]):
			data[x, y] = pixels[y, x][0]

	savetxt("matrix.txt", data, "%4i")