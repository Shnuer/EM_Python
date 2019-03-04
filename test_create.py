import cv2
import numpy as np 

img = cv2.imread("sample.png")

# depth = 'IPL_DEPTH_8U'
# channels = 1
img_h, img_w, img_c = img.shape
print(img.shape)
k=0
o=0
z=0


tst = np.zeros((img_h,img_w,1))
tst = tst.astype(np.uint8)
print(tst.shape)
# print(tst)
for i in tst:
	o=0
	for j in i:
			if o>50:
				tst[k][o] = 255
			o+=1
	k+=1

# np.uint8(tst)

print(tst.shape)
# print(type(tst[1][1][1]))
# tst = tst.astype('np.uint8')
processed = cv2.bitwise_and(img,img, mask = tst)		
print(tst)
while 1:
	cv2.imshow('check',img)
	cv2.imshow('tst',tst)
	cv2.imshow('processed',processed)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break