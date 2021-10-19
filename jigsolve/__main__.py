import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run with a test image')
    parser.add_argument('path')
    args = parser.parse_args()

    img = cv2.imread(args.path)
    cv2.imshow('test', img)
    cv2.waitKey(0)

if __name__ == '__main__': main()
