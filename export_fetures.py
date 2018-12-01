from keras.applications.inception_resnet_v2 import InceptionResNetV2
import keras.backend as K
import numpy as np
import cv2
from os.path import expanduser, join

def main():
	net = InceptionResNetV2()
	net.summary()
	
	home = expanduser("~")
	image = cv2.imread(join(home, "Pictures", "cat.jpg"))
	image = cv2.resize(image, (299,299))
	image = image[np.newaxis,...]
	print image.shape
	im_data = image / 255.
	
	get_features = K.function([net.layers[0].input, K.learning_phase()], [net.get_layer("avg_pool").output])
	features = get_features([im_data,0])[0]
	np.savez_compressed("cat.npz",**{"f1":features})
	
if __name__ == "__main__":
	main()