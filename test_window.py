import cv2


def click_event(event, c, r, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = c * width / img_dim[1], r * height / img_dim[0]
        print(f"({c}, {r}) -> ({round(x, 4)}\", {round(y, 4)}\") -> ")

        # move robot arm to desired coordinates


def main():
    global width, height, img_dim # r and y are measured from top of image
    width, height = 24.0, 18.0 # inches

    img = cv2.imread("img/out/transformed.png")
    img_dim = img.shape[:2]
    print(f"{img.shape=}")
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()