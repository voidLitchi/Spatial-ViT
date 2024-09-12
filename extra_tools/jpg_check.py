import os
import imghdr
from tqdm import tqdm

# path = '/data/yuzhi/Spatial-ViT-main/datasets/voc/masks'
path = '/data/VOCdevkit/VOC2012/JPEGImages/'
original_images = []
for root, dirs, filenames in os.walk(path):
    for filename in filenames:
        original_images.append(os.path.join(root, filename))

original_images = sorted(original_images)
print('num:', len(original_images))
f = open('check_error.txt', 'w+')

error_images = []

for filename in tqdm(original_images):
    check = imghdr.what(filename)
    if check == None:
        f.write(filename)
        f.write('\n')
        error_images.append(filename)

print(len(error_images))

f.seek(0)

for s in f:
    print(s)

f.close()