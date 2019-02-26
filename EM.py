import cv2
import numpy as np


bgr = cv2.imread("first.png")
Lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
pixel = Lab[:, :, 1:3]
print(type(pixel))
data = np.ones((10,10), np.float32)
print(type(data))
value = []
print(range(len(pixel)))
for y in range(len(pixel)):
	for x in range(len(pixel[y])):
		
		
		value.append(pixel[y][x])


value_EM = np.array(value)		
test_value = np.array([[90.1, 120.1]], np.float32)
print(test_value)
create_model = cv2.ml.EM_create()
print(create_model)
create_model.setClustersNumber(2)
print(create_model)
create_model.trainEM(value_EM)

covs = create_model.getCovs()

means = create_model.getMeans()

predict = create_model.predict(test_value)

print(covs)
print()

print(means)
print()
print(predict)


