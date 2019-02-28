import cv2
import numpy as np
import matplotlib.pyplot as plt

bgr = cv2.imread("sample.png")
img_h, img_w, img_c = bgr.shape

Lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
ab_channels = Lab[:, :, 1:3]

data = np.ones((10,10), np.float32)

value = np.ndarray((img_h * img_w, 2), dtype=np.uint8)

pxl_idx = 0
for pixel_line in ab_channels:
	for pixel in pixel_line:
		value[pxl_idx] = pixel
		pxl_idx += 1

print(value.shape)

plt.scatter(value[:, 0], value[:, 1], c='black', edgecolor ='none')

value_EM = np.array(value)		
test_value = np.array([[125.1, 125.1]], np.float32)
print(test_value)
create_model = cv2.ml.EM_create()
print(create_model)
create_model.setClustersNumber(2)
print(create_model)
create_model.trainEM(value_EM)

covs = create_model.getCovs()

means = create_model.getMeans()

predict = create_model.predict(test_value)

print()
print(covs)
print()
print(means)
print()
print(predict)

plt.scatter(means[:, 0], means[:, 1], c='red', edgecolor ='none')
# plt.matshow(covs)

plt.show()
