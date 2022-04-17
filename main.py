from Scripts.RecogonizeFace import recogonizeFace
from Scripts.RegisterNewFace import registerNewFace
from Scripts.Train import train
import warnings
import os
from numpy.core.fromnumeric import argmax
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db = os.path.join(BASE_DIR, "DataBase")

def main():
    print("Program Has Started.")
    while(True):
        c = int(input("1 - RecogonizeFace | 2 - RegisterFace: "))
        if c==1:
            recogonizeFace(db)
        elif c==2:
            registerNewFace(db)
            print("Training Started")
            train(db)
        else:
            print("Invalid Input")

if __name__ == "__main__":
    main()