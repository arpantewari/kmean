import cv2
import numpy as np
class Partition:
    def __init__(self,partition=4):
        self.partition=partition
    def kmeansClustering(self,SegImage):
        SegImage=cv2.GaussianBlur(SegImage,(7,7),0)
        vectorized=SegImage.reshape(-1,3)
        vectorized=np.float32(vectorized)
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
        ret,label,center=cv2.kmeans(vectorized,self.partition,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        res=center[label.flatten()]
        segmented_image=res.reshape((SegImage.shape))
        return label.reshape((SegImage.shape[0],SegImage.shape[1])),segmented_image.astype(np.uint8)
    def extractComponent(self,SegImage,labelImage,label):
        component=np.zeros(SegImage.shape,np.uint8)
        component[labelImage==label]=SegImage[labelImage==label]
        return component
if __name__=="__main__":
    import argparse
    import sys
    SegImage=cv2.imread(r'C:\Users\Lenovo\Desktop\lena.jpg')
    
    
    part=Partition()
    print("Hello World")
    label,output=part.kmeansClustering(SegImage)
    label,output=part.kmeansClustering(SegImage)
    cv2.imshow("Original Image",SegImage)
    cv2.imshow("Segmented Image",output)
    output=part.extractComponent(SegImage,label,3)
    cv2.imshow("Extracted image",output)
