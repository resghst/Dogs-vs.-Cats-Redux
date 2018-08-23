import os, cv2, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND']='tensorflow'

from tqdm import tqdm    #Helps in vis
from random import shuffle 

train_dir = './data/train/'
test_dir = './data/test/'

img_size = 224
print('start')
short_list_train = os.listdir(train_dir) #using a subset of data as resouces as limited. 
short_list_test = os.listdir(test_dir)

def label_img(img):
	word_label = img.split('.')[-3] # conversion to one-hot array [cat,dog]
	if word_label == 'cat': return [1,0]   # [much cat, no dog]
	elif word_label == 'dog': return [0,1] # [no cat, very doggo]

def create_train_data():
	training_data = []
	for img in tqdm(short_list_train):
		label = label_img(img)
		path = os.path.join(train_dir,img)
		img = cv2.imread(path, cv2.IMREAD_COLOR)
		img = cv2.resize(img, (img_size, img_size))
		training_data.append([np.array(img),np.array(label)])
	shuffle(training_data)
	return training_data

def process_test_data():
	testing_data = []
	for img in tqdm(os.listdir(test_dir)):
		path = os.path.join(test_dir,img)
		img_num = img.split('.')[0]
		img = cv2.imread(path,cv2.IMREAD_COLOR)
		img = cv2.resize(img, (img_size,img_size))
		testing_data.append([np.array(img), img_num])
	shuffle(testing_data)
	return testing_data
	
dd = 0
cc = 0
labels = []
for i in short_list_train:
	if 'dog' in i:  labels.append(1)
	else:  labels.append(0)
sns.countplot(labels)
plt.title('Cats and Dogs')
train = create_train_data()
X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,3)
Y = np.array([i[1] for i in train])

from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
print('\n-----------------------------------------------------------\n')
NUM_CLASSES = 2
# RESNET_WEIGHTS_PATH = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'input_shape=(img_size,img_size,3),
model = Sequential()
WEIGHTS_PATH = './premodel/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model.add(ResNet50(include_top=False, pooling='max', weights=WEIGHTS_PATH))
model.add(Dropout(rate=0.25))
model.add(Dense(NUM_CLASSES, activation='softmax'))
# Say not to train first layer (ResNet) model. It is already trained
model.layers[0].trainable = True
print(model.summary())
print('\n-----------------------------------------------------------\n')

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

print('\n-----------------------------------------------------------\n')
import gc 
del X
del Y
del train
gc.collect()
print( "gc.collect" + str(gc.collect()))

short_list_train = os.listdir(train_dir)[-5000:]
train = create_train_data()
X = np.array([i[0] for i in train]).reshape(-1,img_size,img_size,3)
Y = np.array([i[1] for i in train])
history = model.fit(X, Y, validation_split=0.5, epochs=4, batch_size=1)
model.save('./savemodel/my_model.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

aalll = input("inter")
prob = []
img_list = []
test_data =  process_test_data()
for data in tqdm(test_data):
	img_num = data[1]
	img_data = data[0]
	orig = img_data
	data = img_data.reshape(-1,img_size,img_size,3)
	model_out = model.predict([data])[0]
	img_list.append(img_num)
	prob.append(model_out[1])

submission = pd.DataFrame({'id':img_list , 'label':prob})
submission.head()
submission.to_csv("./submit.csv", index=False)







