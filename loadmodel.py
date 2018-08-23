
import os, cv2, random
from random import shuffle 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from tqdm import tqdm    #Helps in vis

img_size = 224
img_size = 224
test_dir = './data/test/'
model = load_model('./savemodel/my_model.h5')

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
submission.to_csv("./ll.csv", index=False)
