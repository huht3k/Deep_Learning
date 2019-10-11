from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt


def main1():
    xml_file = 'G00001.XML'
    classes = ('__background__',  # always index 0
               'cancer')
    bboxes = getBboxes_from_xml_file(xml_file, classes)
    print(bboxes)
    print(type(bboxes))
    print(bboxes.shape)

    img = cv2.imread("001.jpg")
    plt.figure()
    cv2.imshow('cancer', draw_rect(img, bboxes))
    cv2.waitKey(0)

    img, bboxes = RandomHorizontalFlip(1)(img, bboxes)
    plt.figure()
    cv2.imshow('horizontalFlip', draw_rect(img, bboxes))
    cv2.waitKey(0)



def main():
    img = cv2.imread("messi.jpg")[:,:,::-1] #OpenCV uses BGR channels
    bboxes = pkl.load(open("messi_ann.pkl", "rb"))

    plt.figure()
    #draw_rect(img, bboxes)
    plt.imshow(draw_rect(img, bboxes))

    """
    transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.2, diff = True), RandomRotate(10)])

    img, bboxes = transforms(img, bboxes)
    """

    #img, bboxes = RandomHorizontalFlip(1)(img, bboxes)

    #img, bboxes = RandomScale(0.8, diff = True)(img, bboxes)
    #img, bboxes = Scale(-0.2)(img, bboxes)

    #img, bboxes = Rotate(50)(img, bboxes)
    """
    angle = 40
    img = rotate_im(img, angle)
    """
    #img, bboxes = Translate(-0.2, -0.5)(img, bboxes)

    #img, bboxes = RandomTranslate((-0.2, 0.3))(img, bboxes)
    bboxes1 = bboxes.copy()
    img1, bboxes1 = HorizontalFlip()(img, bboxes1)
    img, bboxes = VerticalFlip()(img, bboxes)

    #img, bboxes = RandomVerticalFlip()(img, bboxes)

    #img, bboxes = CrossFlip()(img, bboxes)

    #img, bboxes = RandomCrossFlip(0.8, 0.6)(img, bboxes)

    #img, bboxes = Shear(0.5)(img, bboxes)

    #img, bboxes = RandomShear((-0.5, 0.5))(img, bboxes)

    #img, bboxes = VerticalShear(0.8)(img, bboxes)

    #img, bboxes = RandomVerticalShear((-0.5, 0.5))(img, bboxes)

    #img, bboxes = CrossShear(0.5, 0.5)(img, bboxes)

    #img, bboxes = RandomCrossShear(0.5, 0.5)(img, bboxes)

    #img, bboxes = Resize(400)(img, bboxes)

    #img, bboxes = RandomHSV(10, 10, 10)(img, bboxes)

    #img, bboxes = CrossScale(-1, -2)(img, bboxes)

    plt.figure()
    plt.imshow(draw_rect(img, bboxes))

    plt.show()

if __name__ == '__main__':
    main()