
#-=-=-=-=-=-=-=-=-=-=-#
#       IMPORT        #
#-=-=-=-=-=-=-=-=-=-=-#

import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import h5py
from PIL import Image
from glob import glob

#-=-=-=-=-=-=-=-=-=-=-#
#   CHARGER MODÈLE    #
#-=-=-=-=-=-=-=-=-=-=-#

model = load_model("model.h5")

#-=-=-=-=-=-=-=-=-=-=-#
#        DATA         #
#-=-=-=-=-=-=-=-=-=-=-#

touteimages = glob("../Data/test/*.jpg")
print(touteimages)

#-=-=-=-=-=-=-=-=-=-=-=-=-#
# PREDICTION SUR L'IMAGE  #
#-=-=-=-=-=-=-=-=-=-=-=-=-#

for img in touteimages:
	image2 = Image.open(img)
	image2 = image2.resize((150,150))
	image2 = image2.convert("RGB")
	tab = np.array(image2)
	tab = tab.astype("float32")
	tab /= 255
	tab = tab.reshape((150,150,3))
	tab= np.expand_dims(tab,axis=0)

	prediction = model.predict(tab)
	prediction = np.argmax(prediction)
	#print("la predicton est:"+ str(prediction))
	input("")

	if prediction == 0:
		print("HOMER !")
		image2.show()

	elif prediction == 1:
		print("MARGE !")
		image2.show()

	else:
		print("BART !")
		image2.show()
