import cv2 as cv


def main():
    img = cv.imread('filtered.png')
    # cv.imshow('img', img)

    # blur = cv.bilateralFilter(img, 15, 75, 75)
    blur = cv.GaussianBlur(img, (25, 25), cv.BORDER_DEFAULT)

    canny = cv.Canny(blur, 125, 175, 4)
    cv.imshow('canny', canny)

    cv.waitKey(0)


if __name__ == '__main__':
    main()