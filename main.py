#!/usr/bin/python3

import cv2
import numpy as np
import sys

if len(sys.argv) != 2:
    print("인자로 하나의 파일 명을 입력하세요.")
    sys.exit()

filename = sys.argv[1]

BLUE, GREEN, RED, BLACK, WHITE = (255, 0.0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)
DRAW_BG = {'color': BLACK, 'val': 0}
DRAW_FG = {'color': WHITE, 'val': 1}

rect = (0, 0, 1, 1)
drawing = False
rectangle = False
rect_over = False
rect_or_mask = 100
value = DRAW_FG
thickness = 3

def onMouse(event, x, y, flags, param):
    global ix, iy, img, img2, drawing, value, mask, rectangle
    global rect, rect_or_mask, rect_over

    if event == cv2.EVENT_LBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle:
            img = img2.copy()
            cv2.rectangle(img, (ix, iy), (x, y), RED, 2)
            rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
            rect_or_mask = 0

    elif event == cv2.EVENT_LBUTTONUP:
        rectangle = False
        rect_over = True

        cv2.rectangle(img, (ix, iy), (x, y), RED, 2)
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        rect_or_mask = 0
        print('n : 적용하기')

    if event == cv2.EVENT_RBUTTONDOWN:
        if not rect_over:
            print('마우스 오른쪽 버튼을 누른 채로 전경이 되는 부분을 선택하세요.')
        else:
            drawing = True
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_RBUTTONUP:
        if drawing:
            drawing = False
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    return

def grabcut():
    global ix, iy, img, img2, drawing, value, mask, mask2, rectangle, output
    global rect, rect_or_mask, rect_over

    img = cv2.imread(filename)

    if np.all(img == None):
            print("존재하지 않는 파일이거나, 파일 형식이 올바르지 않습니다.(JPG 파일을 입력해 주세요.)")
            sys.exit()

    img2 = img.copy()

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    output = np.zeros(img.shape, np.uint8)

    cv2.namedWindow('input')
    cv2.namedWindow('output')
    cv2.setMouseCallback('input', onMouse, param=(img, img2))
    cv2.moveWindow('input', img.shape[1] + 10, 90)

    print('왼쪽 마우스 버튼을 누르고 인물이 있는 영역을 지정한 후 n을 누르세요.')

    while True:
        cv2.imshow('output', output)
        cv2.imshow('input', img)

        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break

        if k == ord('0'):
            print('오른쪽 마우스로 제거할 부분을 표시한 후 n을 누르세요.')
            value = DRAW_BG
        elif k == ord('1'):
            print('오른쪽 마우스로 복원할 부분을 표시한 후 n을 누르세요.')
            value = DRAW_FG
        elif k == ord('r'):
            print('리셋합니다')
            rect = (0, 0, 1, 1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            output = np.zeros(img.shape, np.uint8)
            print('0:제거배경선택 1:복원전경선택 n:적용하기 r:리셋\n영역 지정이 완료되었다면 esc키를 누르세요.')
        elif k == ord('n'):
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            if rect_or_mask == 0:
                cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
                rect_or_mask = 1
            elif rect_or_mask == 1:
                cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

            print('0:제거배경선택 1:복원전경선택 n:적용하기 r:리셋\n영역 지정이 완료되었다면 esc키를 누르세요.')
        
        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')        
        output = cv2.bitwise_and(img2, img2, mask=mask2)

    cv2.destroyAllWindows()

def filtering():
    mask3 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    dst = cv2.subtract(img2, mask3)
    cv2.imshow('output', output)

    print('1부터 5까지의 숫자 중 하나를 입력하세요. : ')

    while True:

        z = cv2.waitKey(1) & 0xFF

        if z == 27:
            print('필터를 선택하지 않았습니다. 프로그램을 종료합니다.')
            sys.exit()
            break

        if z == ord('1'):
            print('필터 1번 적용. esc키를 눌러 종료하세요.')
            dst2 = cv2.Laplacian(dst, cv2.CV_64F)
            break

        if z == ord('2'):
            print('필터 2번 적용. esc키를 눌러 종료하세요.')
            dst2 = cv2.Sobel(dst, cv2.CV_64F, 0, 1, ksize=3)
            break

        if z == ord('3'):
            print('필터 3번 적용. esc키를 눌러 종료하세요.')
            dst2 = cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=3)
            break

        if z == ord('4'):
            print('필터 4번 적용. esc키를 눌러 종료하세요.')
            filter = np.array([[0, -1, -3], [1, 0, -1], [3, 1, 0]])
            dst2 = cv2.filter2D(dst, -1, filter)
            break

        if z == ord('5'):
            print('필터 5번 적용. esc키를 눌러 종료하세요.')
            dst0 = cv2.GaussianBlur(dst, (0, 0), 2.0)
            dst2 = cv2.Laplacian(dst0, cv2.CV_64F)
            break

    dst3 = np.uint8(dst2)
    dst4 = cv2.add(output, dst3)
    cv2.imshow('filtered', dst4)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('filtered_'+filename, dst4)

grabcut()
filtering()
