import sys
import os
import time
import cv2
import csv
from sys import platform
import argparse
import datetime
import joblib
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
detect=joblib.load("detecter3.pkl")


windowName = "Buidling environment is bad for your health"
#Target FPS
frameRate = 30
#take a picture every {samplePeriod} sec.
samplePeriod = 2

windowName = "Buidling environment is bad for your health"
#Target FPS
frameRate = 30
#take a picture every {samplePeriod} sec.
samplePeriod = 2
whatdo=""
def getFocusBodyIndex(keypoints, bodyCount):
    focusIndex = 0
    maxRadiusSquared = 0
    for bodyIndex in range(bodyCount):
        dx = keypoints[bodyIndex][5][0] - keypoints[bodyIndex][2][0]
        dy = keypoints[bodyIndex][5][1] - keypoints[bodyIndex][2][1]
        radiusSquared = dx*dx + dy*dy
        if maxRadiusSquared < radiusSquared:
            maxRadiusSquared = radiusSquared
            focusIndex = bodyIndex
    return focusIndex

def main():
    buildPath = os.path.dirname(os.path.realpath(__file__))
    
    sys.path.append(buildPath + "/python/openpose/Release");
    os.environ['PATH']  += ';' + (buildPath + "/x64/Release") + ';' + (buildPath + "/bin;")
    import pyopenpose as op
    
    params = dict()
    params["model_folder"] = "../models/"
    params["model_pose"] = "BODY_25"
    params["net_resolution"] = "-1x224"
    params["hand"] = True
    params["hand_detector"] = 0
    params["hand_net_resolution"] = "224x224"
    openposeHandle = op.WrapperPython()
    openposeHandle.configure(params)
    openposeHandle.start()
    
    
    datum = op.Datum()
    cam = cv2.VideoCapture(0)
    
    frameElapsedTime = 1
    totalTimeElapsed = 0
    sampleIndex = 0
    OutputStr = ""
    while True:
        frameStart = time.time()
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        datum.cvInputData = frame
        openposeHandle.emplaceAndPop(op.VectorDatum([datum]))
        if type(datum.poseKeypoints) != type(None):
            if totalTimeElapsed > samplePeriod:
                totalTimeElapsed -= samplePeriod
                focusIndex = getFocusBodyIndex(datum.poseKeypoints, len(datum.poseKeypoints))
                body = datum.poseKeypoints[focusIndex]
                leftHand = datum.handKeypoints[0][focusIndex]
                rightHand = datum.handKeypoints[1][focusIndex]
                            
                X=[body[0][0], body[0][1], body[1][0], body[1][1], body[2][0], body[2][1], body[3][0], body[3][1],body[4][0], 
                body[4][1], body[5][0], body[5][1], body[6][0], body[6][1], body[7][0], body[7][1],body[15][0],
                body[15][1], body[16][0], body[16][1], body[17][0], body[17][1], body[18][0], body[18][1],leftHand[0][0], leftHand[0][1]]
                Xdict={'a':X}
                X= pd.DataFrame(Xdict)
                
                
                #X=preprocessing.scale(X)
                #print(X)
                preX=detect.predict(X.T)
                max_a = np.argmax(preX, axis=1)
                
                if max_a==0:
                    OutputStr = "studying"
                if max_a==1:
                    OutputStr = "using smartphone"
                if max_a==2:
                   OutputStr = "sleeping"
        
        datum.cvOutputData = cv2.putText(datum.cvOutputData, OutputStr, (2, 34), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
        datum.cvOutputData = cv2.putText(datum.cvOutputData,OutputStr, (0, 32), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        #datum.cvOutputData = cv2.putText(datum.cvOutputData, "FPS: %.02f Target FPS: %.02f" % (1.0 / frameElapsedTime, frameRate), (2, 34), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
        #datum.cvOutputData = cv2.putText(datum.cvOutputData,"FPS: %.02f Target FPS: %.02f" % (1.0 / frameElapsedTime, frameRate), (0, 32), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        cv2.imshow(windowName, datum.cvOutputData)
        frameEnd = time.time()
        remainTime = (1.0 / frameRate) - (frameEnd - frameStart)
        if remainTime > 0:
            key = cv2.waitKey(int(remainTime*1000) & 0xffffffff)
        else:
            key = cv2.waitKey(1)
        frameEnd = time.time()
        frameElapsedTime = frameEnd - frameStart
        totalTimeElapsed += frameElapsedTime
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_ASPECT_RATIO) == -1: break
    cam.release()

main()