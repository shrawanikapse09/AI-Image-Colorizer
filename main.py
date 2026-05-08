import cv2
import numpy as np
from cv2 import dnn

# Model paths
proto_file = "model/colorization_deploy_v2.prototxt"
model_file = "model/colorization_release_v2.caffemodel"
pts_file = "model/pts_in_hull.npy"

# Load model
net = dnn.readNetFromCaffe(proto_file, model_file)
pts = np.load(pts_file)

# Load cluster centers
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Read image
image = cv2.imread("images/img1.jpg")

# Normalize image
scaled = image.astype("float32") / 255.0

# Convert to LAB
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Resize image
resized = cv2.resize(lab, (224, 224))

# Extract L channel
L = cv2.split(resized)[0]

# Mean centering
L -= 50

# Predict ab channels
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize predicted channels
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# Original L channel
L_original = cv2.split(lab)[0]

# Merge channels
colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)

# Convert LAB to BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

# Clip values
colorized = np.clip(colorized, 0, 1)

# Convert to uint8
colorized = (255 * colorized).astype("uint8")

# Resize images for display
image = cv2.resize(image, (640, 640))
colorized = cv2.resize(colorized, (640, 640))

# Combine images
result = cv2.hconcat([image, colorized])

# Show output
cv2.imshow("Black & White to Color", result)

cv2.waitKey(0)
cv2.destroyAllWindows()