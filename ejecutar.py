from maxMin import maxAndMin
from conv2Net import ConvNet
import face_recognition
import cv2 as cv
import torch
import torch.nn as nn
import pyautogui
import os
import numpy as np
import copy
import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pyautogui.FAILSAFE=False


def process(im):
    eye = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    eye = cv.resize(eye, dsize=(100, 50))

    # Display the image - DEBUGGING ONLY
    #cv.imshow('frame', left_eye)

    top = max([max(x) for x in eye])
    eye = (torch.tensor([[eye]]).to(dtype=torch.float,
                                              device=device)) / top
    return eye

def dataLoad(path, want = 0):
    totalHolder = []
    dims = [1600,900]

    im = cv.cvtColor(cv.imread(path + "/" + "465.750.60.jpg"), cv.COLOR_BGR2GRAY)
    top = max([max(x) for x in im])
    totalHolder.append(((torch.tensor([[im]]).to(dtype=torch.float,device=device))/top,torch.tensor([[int(("465.621.60.jpg".split("."))[want])/dims[want]]]).to(dtype=torch.float,device=device)))

    # print(totalHolder)
    return totalHolder

#pruebaderecho = dataLoad("pruebaderecho")
#pruebaizquierdo = dataLoad("pruebaizquierdo")


def eyetrack(xshift = 70, yshift=60, frameShrink = 0.15):
    modelx= ConvNet().to(device)
    modelx.load_state_dict(torch.load("xModels/kernel_7_36_x.plt",map_location=device))
    modelx.eval()

    modely= ConvNet().to(device)
    modely.load_state_dict(torch.load("xModels/Kernel_7_33_y.plt",map_location=device))
    modely.eval()

    alpha = 0.1 

    smoothed_x = None
    smoothed_y = None

    webcam = cv.VideoCapture(0)
    
    while True:

        ret, frame = webcam.read()
        smallframe = cv.resize(copy.deepcopy(frame), (0, 0), fy=frameShrink, fx=frameShrink)
        smallframe = cv.cvtColor(smallframe, cv.COLOR_BGR2GRAY)
        feats = face_recognition.face_landmarks(smallframe)
   
        if len(feats) > 0:

            leBds, leCenter = maxAndMin(feats[0]['left_eye'], mult=1/frameShrink)
            reBds, leCenter = maxAndMin(feats[0]['right_eye'], mult=1/frameShrink)

            right_eye = frame[leBds[1]:leBds[3], leBds[0]:leBds[2]]
            left_eye=frame[reBds[1]:reBds[3], reBds[0]:reBds[2]]



            right_eye = process(right_eye)
            left_eye = process(left_eye)

            x=modelx(right_eye,left_eye)
            y=modely(right_eye,left_eye)


            x=x.item()*1600
            y=y.item()*900

            if smoothed_x is None:
                smoothed_x = x
                smoothed_y = y
            else:
                smoothed_x = alpha * x + (1 - alpha) * smoothed_x
                smoothed_y = alpha * y + (1 - alpha) * smoothed_y
            


            pyautogui.moveTo(smoothed_x, smoothed_y)
            



            cv.imshow("Frame", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break



eyetrack()