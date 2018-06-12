from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
# from keras.applications.mobilenet import MobileNet, preprocess_input
import numpy as np
from glob import glob
from collections import defaultdict
import os
import pickle
import sys

image_path = sys.argv[1]
result_path = sys.argv[2]

model = ResNet50(weights='imagenet')

broken_image = []
pred_result = defaultdict(list)

image_files = glob(os.path.join(image_path, '*.jpg'))
print(len(image_files))
len_file = len(image_files)
for i, f in enumerate(image_files):
    try:
        img_id = os.path.splitext(os.path.basename(f))[0]
        # print(img_id)
        img = image.load_img(f, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        result = decode_predictions(preds, top=3)[0]
        for r in result:
            pred_result[img_id].append(r[1])
        # print('Predicted:', result)
    except:
        broken_image.append(f)
    print("{}/{}".format(i, len_file), end='\r')

with open(result_path, 'wb') as f:
    pickle.dump(pred_result, f)

