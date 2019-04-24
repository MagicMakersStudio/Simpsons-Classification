
#-=-=-=-=-=-=-=-=-=-=-#
#       IMPORT        #
#-=-=-=-=-=-=-=-=-=-=-#

from PIL import Image
import numpy
from glob import glob
import numpy as np
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import h5py

#-=-=-=-=-=-=-=-=-=-=-#
#        DATA         #
#-=-=-=-=-=-=-=-=-=-=-#

images_pretes = []
label = []
X_train = []
Y_train = []
X_test = []
Y_test = []


##### HOMER

touteimages = glob("../Data/dataset simpsons/homer_simpson/*.jpg")

for img in tqdm(touteimages):
	homer = Image.open(img)
	homer = homer.resize((150,150))
	homer = np.array(homer)
	homer = homer.astype("float32")
	homer /= 255
	homer = homer.reshape(150,150, 3)
	images_pretes.append(homer)
	label.append(0)

##### MARGE

touteimages = glob("../Data/dataset simpsons/marge_simpson/*.jpg")

for img in tqdm(touteimages):
	marge = Image.open(img)
	marge = marge.resize((150,150))
	marge = np.array(marge)
	marge = marge.astype("float32")
	marge /= 255
	marge = marge.reshape(150,150, 3)
	images_pretes.append(marge)
	label.append(1)

##### BART

touteimages = glob("../Data/dataset simpsons/bart_simpson/*.jpg")

for img in tqdm(touteimages):
	bart = Image.open(img)
	bart = bart.resize((150,150))
	bart = np.array(bart)
	bart = bart.astype("float32")
	bart /= 255
	bart = bart.reshape(150,150, 3)
	images_pretes.append(bart)
	label.append(2)

label = to_categorical(label,3)

#print(images_pretes)

label = np.array(label)
images_pretes = np.array(images_pretes)

X_train,X_test,Y_train,Y_test = train_test_split(images_pretes,label,test_size =0.5)

print(label)
print(images_pretes[0])


#-=-=-=-=-=-=-=-=-=-=-#
#       MODÈLE        #
#-=-=-=-=-=-=-=-=-=-=-#

model = Sequential()
model.add(Conv2D(24, (5,5),strides=(2,2), activation="relu", input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(32, (5,5),strides=(2,2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(64, (5,5),strides=(2,2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(100, activation="relu"))

model.add(Dense(3, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",
				optimizer=Adam(),
				metrics= ["accuracy"])

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#       TRAIN - ENTRAÎNEMENT        #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

model.fit(images_pretes,label,
		epochs=100,
		batch_size=100,
		validation_data=(X_test, Y_test))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#       SAUVEGARDER MODÈLE        #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

model.save("model.h5")
