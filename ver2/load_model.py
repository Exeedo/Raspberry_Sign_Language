# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:09:40 2022

@author: H
"""

import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import arabic_reshaper
import warnings
warnings.filterwarnings('ignore')
arabic = lambda x: arabic_reshaper.reshape(x)[::-1]

device = "cpu"

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5),
    transforms.Grayscale(num_output_channels=1)
])


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
     
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(8 * 8 * 64, 1000),

            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),

            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 32),

         
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x
    

classToLtetter = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa',  'la', 'laam', 'meem', 'nun',  'ra', 'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']
ARABIC_LETTERS = {
    'aleff' : ('أ', 'ألف'),
    'bb'    : ('ب', 'باء'),
    'taa'   : ('ت', 'تاء'),
    'thaa'  : ('ث', 'ثاء'),
    'jeem'  : ('ج', 'جيم'),
    'haa'   : ('ح', 'حاء'),
    'khaa'  : ('خ', 'خاء'),
    'dal'   : ('د', 'دال'),
    'thal'  : ('ذ', 'ذال'),
    'ra'    : ('ر', 'راء'),
    'zay'   : ('ز', 'زاي'),
    'seen'  : ('س', 'سين'),
    'sheen' : ('ش', 'شين'),
    'saad'  : ('ص', 'صاد'),
    'dhad'  : ('ض', 'ضاد'),
    'ta'    : ('ط', 'طاء'),
    'dha'   : ('ظ', 'ظاء'),
    'ain'   : ('ع', 'عين'),
    'ghain' : ('غ', 'غين'),
    'fa'    : ('ف', 'فاء'),
    'gaaf'  : ('ق', 'قاف'),
    'kaaf'  : ('ك', 'كاف'),
    'laam'  : ('ل', 'لام'),
    'meem'  : ('م', 'ميم'),
    'nun'   : ('ن', 'نون'),
    'ha'    : ('هـ', 'هاء'),
    'waw'   : ('و', 'واو'),
    'ya'    : ('ي', 'ياء'),
    'toot'  : ('ة', 'تاء مربوطة'),
    'al'    : ('ال', 'ال التعريف'),
    'la'    : ('لا', '(لام-ألف)'),
    'yaa'   : ('ئ', 'همزة على نبرة'),
}

        
model_path="ver2/project_signLanguage.pt"       
model = CNN().to(device)
model.load_state_dict(torch.load(model_path),strict=False)
model.eval()

#complete code read from camera retrun the class (the letter)
cam = cv2.VideoCapture(0)
if cam.isOpened():
    result, frame = cam.read()
else:
    result = False

while result:
    cv2.imshow("Camera-Live", frame)
    rval, frame = cam.read()
    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        break
    if key != ord('s'):
        continue
    cv2.imshow("Camera-Saved", frame)
    image = image_transform(frame)
    x, y, z = image.shape
    image = image.reshape(1, x, y, z)
    with torch.no_grad():
        inputs = image.to(device)
        outputs = model(inputs).cpu()
        prediction = torch.max(outputs, 1)[1]
        letter=classToLtetter[prediction]
        print(f"Prediction: {letter} - ({arabic(ARABIC_LETTERS[letter][0])}) - {arabic(ARABIC_LETTERS[letter][1])}")
        # print(ARABIC_LETTERS[letter])
cam.release()
cv2.destroyAllWindows()
