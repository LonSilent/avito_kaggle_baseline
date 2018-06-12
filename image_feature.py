from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
#import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 
import sys

#from IPython.core.display import HTML 
#from IPython.display import Image

images_path = sys.argv[1]
imgs = os.listdir(images_path)[:]

def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1

    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]

    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

def perform_color_analysis(img, flag):
    path = images_path + img 
    im = IMG.open(path) #.convert("RGB")

    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 

    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None

def average_pixel_width(img):
    path = images_path + img 
    im = IMG.open(path)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100
                                                                                      
def get_dominant_color(img):
    path = images_path + img 
    img = cv2.imread(path)
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color

def get_average_color(img):
    path = images_path + img 
    img = cv2.imread(path)
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color

def getSize(filename):
    filename = images_path + filename
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    filename = images_path + filename
    img_size = IMG.open(filename).size
    return img_size 

def get_blurrness_score(image):
    path =  images_path + image 
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm

if __name__ == '__main__':
    features = pd.DataFrame()
    features['image'] = imgs
    idx = sys.argv[1][-2]
    print (idx)
    print ('number of figures:', len(imgs))
    print ('process dullness')
    features['dullness'] = features['image'].apply(lambda x : perform_color_analysis(x, 'black'))
    print ('process whiteness')
    features['whiteness'] = features['image'].apply(lambda x : perform_color_analysis(x, 'white'))
    print ('process average_pixel_width')
    features['average_pixel_width'] = features['image'].apply(average_pixel_width)
    print ('process dominant_color')
    features['dominant_color'] = features['image'].apply(get_dominant_color)
    print ('process dominant_red')
    features['dominant_red'] = features['dominant_color'].apply(lambda x: x[0]) / 255
    print ('process dominant_green')
    features['dominant_green'] = features['dominant_color'].apply(lambda x: x[1]) / 255
    print ('process dominant_blue')
    features['dominant_blue'] = features['dominant_color'].apply(lambda x: x[2]) / 255
    print ('process average_color')
    features['average_color'] = features['image'].apply(get_average_color)
    print ('process average_red')
    features['average_red'] = features['average_color'].apply(lambda x: x[0]) / 255
    print ('process average_green')
    features['average_green'] = features['average_color'].apply(lambda x: x[1]) / 255
    print ('process average_blue')
    features['average_blue'] = features['average_color'].apply(lambda x: x[2]) / 255
    print ('process image_size')
    features['image_size'] = features['image'].apply(getSize)
    print ('process temp_size')
    features['temp_size'] = features['image'].apply(getDimensions)
    print ('process width')
    features['width'] = features['temp_size'].apply(lambda x : x[0])
    print ('process height')
    features['height'] = features['temp_size'].apply(lambda x : x[1])
    print ('process drop')
    features = features.drop(['temp_size', 'average_color', 'dominant_color'], axis=1)
    print ('process blurrness')
    features['blurrness'] = features['image'].apply(get_blurrness_score)
    fname = './features/' + idx + '.csv'
    print ('save to %s' % (fname))
    features.to_csv(fname, index=False)


