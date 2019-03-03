import cv2
import numpy as np
import matplotlib.pyplot as plt



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
	plt.scatter(means[:, 0], means[:, 1], c='red', edgecolor ='none')
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

	plt.scatter(value[:, 0], value[:, 1], c='black', edgecolor ='none')

	value_EM = np.array(value)
	return value_EM



bgr = cv2.imread("sample.png")
print(bgr.shape)
value = preparation_value(bgr)
model = training(value)

print(value.shape)

	
test_value = np.array([[100.1, 180.1]], np.float32)
plt.scatter(test_value[:, 0], test_value[:, 1], c='green', edgecolor ='none')

test_value1 = np.array([[130.1, 130.1]], np.float32)
plt.scatter(test_value1[:, 0], test_value1[:, 1], c='blue', edgecolor ='none')

# test_value2 = np.array([[110.1, 110.1]], np.float32)
test_value3 = np.array([[103.224, 172.058]], np.float32)
plt.scatter(test_value3[:, 0], test_value3[:, 1], c='yellow', edgecolor ='none')


cheсk_predict(model,test_value)
cheсk_predict(model,test_value1)
# cheсk_predict(model,test_value2)
cheсk_predict(model,test_value3)



plt.show()
