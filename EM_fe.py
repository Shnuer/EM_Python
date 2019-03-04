import cv2
import numpy as np 

def cheсk_predict(model, value_like):
	print("Test value")
	print(value_like)
	print("Full model")
	print(model.predict(value_like))
	_, k = model.predict(value_like)
	print("Like hood")
	print(k[0][0])
	print()


def training(value_for_training):
	create_model = cv2.ml.EM_create()
	create_model.setClustersNumber(2)
	create_model.trainEM(value_for_training)
	means = create_model.getMeans()
	# plt.scatter(means[:, 0], means[:, 1], c='red', edgecolor ='none')
	return create_model


def preparation_value(example):
	img_h, img_w, img_c = example.shape
	Lab_example = cv2.cvtColor(example, cv2.COLOR_BGR2Lab)
	ab_channels = Lab_example[:, :, 1:3]
	
	value = np.ndarray((img_h * img_w, 2), dtype=np.uint8)

	pxl_idx = 0
	for pixel_line in ab_channels:
		for pixel in pixel_line:
			value[pxl_idx] = pixel
			pxl_idx += 1

	# plt.scatter(value[:, 0], value[:, 1], c='black', edgecolor ='none')

	value_EM = np.array(value)
	return value_EM


def comparison_with_sample(model, frame,img_h,img_w):

	img_h, img_w, img_c = frame.shape
	Lab_example = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
	ab_channels = Lab_example[:, :, 1:3]


	# img_h, img_w, img_c = frame.shape
	mask = np.zeros((img_h,img_w,1))
	mask = mask.astype(np.uint8)

	pixel_str = np.array([[0, 0]], np.float32)
	x = 0
	y=0
	for pixel_line in ab_channels:
		x = 0
		for pixel in pixel_line:
			
			pixel = pixel.astype(np.float32)

			
			pixel_str[0]= pixel
			_, like_hood = model.predict(pixel_str)

			if like_hood[0][1] > 0.90:
				mask[y][x] = 255

			x += 1
		y += 1


	return mask

cap = cv2.VideoCapture(0)
_, frame = cap.read()

# bgr = cv2.imread("sample.png")
bgr = frame
img_h, img_w, img_c = frame.shape

value = preparation_value(bgr)
model = training(value)
mask = comparison_with_sample(model, bgr,img_h,img_w)
processed = cv2.bitwise_and(bgr,bgr, mask = mask)
cv2.imshow('mask', mask)
cv2.imshow('processed', processed)
cv2.imshow('bgr', bgr)

img_h, img_w, img_c = frame.shape

while 1:

	_, frame = cap.read()
	mask = comparison_with_sample(model, frame,img_h,img_w)
	with_mask = cv2.bitwise_and(frame,frame, mask = mask)
	cv2.imshow('with_mask', with_mask)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break