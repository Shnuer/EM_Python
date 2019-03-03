import cv2
import numpy as np 

img = cv2.imread("sample.png")

# depth = 'IPL_DEPTH_8U'
# channels = 1
img_h, img_w, img_c = img.shape



tst = np.ones((img_h,img_w,3))
print(tst)

for first in tst:
	for second in first:
		if first.any() <100:
			tst.all([int(first)][int(second)]) = 0
		

while 1:
	cv2.imshow('check',img)
	cv2.imshow('tst',tst)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break