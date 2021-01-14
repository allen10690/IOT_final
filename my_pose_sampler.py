import sys
import os
import time
import cv2
import csv
from sys import platform
import argparse
import datetime

windowName = "Buidling environment is bad for your health"
#Target FPS
frameRate = 30
#take a picture every {samplePeriod} sec.
samplePeriod = 2

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
	t = datetime.datetime.now()
	samplePath = buildPath + "/my_samples_%04d_%02d_%02d_%02d_%02d_%02d" % (t.year, t.month, t.day, t.hour, t.minute, t.second)
	
	sys.path.append(buildPath + "/python/openpose/Release")
	os.environ['PATH']  += ';' + (buildPath + "/x64/Release") + ';' + (buildPath + "/bin;")
	import pyopenpose as op
	params = dict()
	params["model_folder"] = "../models/"
	params["model_pose"] = "BODY_25"
	params["net_resolution"] = "-1x160"
	params["hand"] = True
	params["hand_detector"] = 0
	params["hand_net_resolution"] = "224x224"
	openposeHandle = op.WrapperPython()
	openposeHandle.configure(params)
	openposeHandle.start()
	os.makedirs(samplePath)
	
	datum = op.Datum()
	cam = cv2.VideoCapture(0)
	with open(samplePath + "/sample_coords.csv", 'w', newline='') as csvFile:
		csvWriter = csv.writer(csvFile)
		csvWriter.writerow(["focusing", 
			"nose_x", "nose_y", "neck_x", "neck_y", "rShoulder_x", "rShoulder_y", "rElbow_x", "rElbow_y", 
			"rWrist_x", "rWrist_y", "lShoulder_x", "lShoulder_y", "lElbow_x", "lElbow_y", "lWrist_x", "lWrist_y", 
			"rEye_x", "rEye_y", "lEye_x", "lEye_y", "rEar_x", "rEar_y", "lEar_x", "lEar_y", 
			"lHand_x", "lHand_y", 
			"lThumb0_x", "lThumb0_y", "lThumb1_x", "lThumb1_y", "lThumb2_x", "lThumb2_y", "lThumb3_x", "lThumb3_y", 
			"lIndexFinger0_x", "lIndexFinger0_y", "lIndexFinger1_x", "lIndexFinger1_y", "lIndexFinger2_x", "lIndexFinger2_y", "lIndexFinger3_x", "lIndexFinger3_y", 
			"lMiddleFinger0_x", "lMiddleFinger0_y", "lMiddleFinger1_x", "lMiddleFinger1_y", "lMiddleFinger2_x", "lMiddleFinger2_y", "lMiddleFinger3_x", "lMiddleFinger3_y", 
			"lRingFinger0_x", "lRingFinger0_y", "lRingFinger1_x", "lRingFinger1_y", "lRingFinger2_x", "lRingFinger2_y", "lRingFinger3_x", "lRingFinger3_y", 
			"lLittleFinger0_x", "lLittleFinger0_y", "lLittleFinger1_x", "lLittleFinger1_y", "lLittleFinger2_x", "lLittleFinger2_y", "lLittleFinger3_x", "lLittleFinger3_y", 
			"rHand_x", "rHand_y", 
			"rThumb0_x", "rThumb0_y", "rThumb1_x", "rThumb1_y", "rThumb2_x", "rThumb2_y", "rThumb3_x", "rThumb3_y", 
			"rIndexFinger0_x", "rIndexFinger0_y", "rIndexFinger1_x", "rIndexFinger1_y", "rIndexFinger2_x", "rIndexFinger2_y", "rIndexFinger3_x", "rIndexFinger3_y", 
			"rMiddleFinger0_x", "rMiddleFinger0_y", "rMiddleFinger1_x", "rMiddleFinger1_y", "rMiddleFinger2_x", "rMiddleFinger2_y", "rMiddleFinger3_x", "rMiddleFinger3_y", 
			"rRingFinger0_x", "rRingFinger0_y", "rRingFinger1_x", "rRingFinger1_y", "rRingFinger2_x", "rRingFinger2_y", "rRingFinger3_x", "rRingFinger3_y", 
			"rLittleFinger0_x", "rLittleFinger0_y", "rLittleFinger1_x", "rLittleFinger1_y", "rLittleFinger2_x", "rLittleFinger2_y", "rLittleFinger3_x", "rLittleFinger3_y"])
		
		frameElapsedTime = 1
		totalTimeElapsed = 0
		sampleIndex = 0
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
					cv2.imwrite(samplePath + "/%08d.jpg" % sampleIndex, datum.cvOutputData)
					csvWriter.writerow([0, 
						body[0][0], body[0][1], body[1][0], body[1][1], body[2][0], body[2][1], body[3][0], body[3][1], 
						body[4][0], body[4][1], body[5][0], body[5][1], body[6][0], body[6][1], body[7][0], body[7][1], 
						body[15][0], body[15][1], body[16][0], body[16][1], body[17][0], body[17][1], body[18][0], body[18][1], 
						leftHand[0][0], leftHand[0][1], 
						leftHand[ 1][0], leftHand[ 1][1], leftHand[ 2][0], leftHand[ 2][1], leftHand[ 3][0], leftHand[ 3][1], leftHand[ 4][0], leftHand[ 4][1], 
						leftHand[ 5][0], leftHand[ 5][1], leftHand[ 6][0], leftHand[ 6][1], leftHand[ 7][0], leftHand[ 7][1], leftHand[ 8][0], leftHand[ 8][1], 
						leftHand[ 9][0], leftHand[ 9][1], leftHand[10][0], leftHand[10][1], leftHand[11][0], leftHand[11][1], leftHand[12][0], leftHand[12][1], 
						leftHand[13][0], leftHand[13][1], leftHand[14][0], leftHand[14][1], leftHand[15][0], leftHand[15][1], leftHand[16][0], leftHand[16][1], 
						leftHand[17][0], leftHand[17][1], leftHand[18][0], leftHand[18][1], leftHand[19][0], leftHand[19][1], leftHand[20][0], leftHand[20][1], 
						rightHand[0], rightHand[0][1], 
						rightHand[ 1][0], rightHand[ 1][1], rightHand[ 2][0], rightHand[ 2][1], rightHand[ 3][0], rightHand[ 3][1], rightHand[ 4][0], rightHand[ 4][1], 
						rightHand[ 5][0], rightHand[ 5][1], rightHand[ 6][0], rightHand[ 6][1], rightHand[ 7][0], rightHand[ 7][1], rightHand[ 8][0], rightHand[ 8][1], 
						rightHand[ 9][0], rightHand[ 9][1], rightHand[10][0], rightHand[10][1], rightHand[11][0], rightHand[11][1], rightHand[12][0], rightHand[12][1], 
						rightHand[13][0], rightHand[13][1], rightHand[14][0], rightHand[14][1], rightHand[15][0], rightHand[15][1], rightHand[16][0], rightHand[16][1], 
						rightHand[17][0], rightHand[17][1], rightHand[18][0], rightHand[18][1], rightHand[19][0], rightHand[19][1], rightHand[20][0], rightHand[20][1]])
					print("sample %08d saved." % sampleIndex)
					sampleIndex += 1
			
			datum.cvOutputData = cv2.putText(datum.cvOutputData, "FPS: %.02f Target FPS: %.02f" % (1.0 / frameElapsedTime, frameRate), (2, 34), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))
			datum.cvOutputData = cv2.putText(datum.cvOutputData, "FPS: %.02f Target FPS: %.02f" % (1.0 / frameElapsedTime, frameRate), (0, 32), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
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