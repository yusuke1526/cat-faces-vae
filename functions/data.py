import cv2

def fileRead(filename, image_w=64, image_h=64):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255
    return img

    