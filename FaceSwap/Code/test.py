import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('output-1.avi')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# create a list of first 5 frames
img1 = [cap.read()[1] for i in xrange(length)]
height = img1[0].shape[0]
width = img1[0].shape[1]
print(width,height)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
t="gray"
out = cv2.VideoWriter('output-{}.avi'.format(t),fourcc, 20, (width,height))
print(len(img1))
for k in xrange(len(img1)-5):

	# convert all to grayscale
	img = [img1[j+k] for j in xrange(5)]

	# gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]

	# convert all to float64
	gray = [np.float64(i) for i in img]

	# create a noise of variance 25
	noise = np.random.randn(*gray[1].shape)*10

	# Add this noise to images
	noisy = [i+noise for i in gray]

	# Convert back to uint8
	noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]

	# Denoise 3rd frame considering all the 5 frames
	dst = cv2.fastNlMeansDenoisingColoredMulti(noisy, 2, 5, None, 4, 7, 35)
	# print(dst.shape)
	# dst = cv2.merge((dst,dst,dst))
	# cv2.imshow('gray',dst)
	# cv2.waitKey(100)
	# cv2.destroyAllWindows()
	# plt.figure(1)
	# plt.imshow(dst,cmap='gray')
	# plt.pause(.1)


	out.write(dst)

# plt.subplot(131),plt.imshow(gray[2],'gray')
# plt.subplot(132),plt.imshow(noisy[2],'gray')
# plt.subplot(133),plt.imshow(dst,'gray')
# plt.show()
