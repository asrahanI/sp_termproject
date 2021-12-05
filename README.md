# 사진 왜곡 프로그램

프로젝트 멤버 : 콘텐츠IT학과 20175316 안준용(1인) / 담당 파트 : 프로젝트 전체

***

## 프로젝트 및 개발 내용 소개

인물이 있는 사진에서, 인물을 제외한 배경 부분만 왜곡시켜 색다른 느낌의 사진을 만들어 내는 프로그램을 제작하였습니다.

프로그래밍 언어로는 파이썬을 사용했고, opencv를 통해 영상처리를 진행하였습니다.

입력받은 이미지에서 인물을 추출한 후, 인물을 제외한 배경 부분에 필터를 적용하는 원리입니다.

이미지에서 인물 부분을 추출하기 위해 grabcut 알고리즘을 사용하였습니다.

## 다이어그램

![image](https://user-images.githubusercontent.com/92137084/144738241-51f9916f-196f-4a5f-abe3-b12244bf4b3d.png)

## 프로젝트 개발 결과물 소개

아래 사진과 같이, 인물이 있는 사진을 프로그램에 입력하면 배경 부분의 왜곡을 통해 다섯 종류의 색다른 사진을 만들어 낼 수 있습니다.

![image](https://user-images.githubusercontent.com/92137084/144738578-4d0535cb-5b57-4ea3-8303-e5a7ce5903b3.png)

```python

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

```

상기의 코드를 통해 구현하였으며, 파이썬의 opencv를 베이스로 하여 제작하였습니다.

이미지를 입력받은 후, 마우스 조작을 통해 이미지에서 인물 영역을 지정하고 배경 부분에 필터를 적용하는 프로그램입니다.

## 개발 결과물 사용법 소개

리눅스 터미널을 통해 사용됩니다. 

./'실행 프로그램명' '파일명' 을 입력하면 해당 이미지를 편집하게 됩니다.

만일 파일명을 입력하지 않았거나, 하나 이상의 파일명을 입력하거나, 존재하지 않는 파일을 입력하는 등 잘못된 방식으로 사용 시 올바른 사용법을 출력합니다.

![image](https://user-images.githubusercontent.com/92137084/144740212-5fbafcc5-456a-4c6e-b4d3-ac08344798b4.png)

올바른 방식으로 사용한 경우, 사용법과 함께 사진의 영역 지정 창이 생성됩니다.

![image](https://user-images.githubusercontent.com/92137084/144740237-c5301ed9-39c5-470b-80e9-5d7c8a4d2e6c.png)

사용법에 따라 영역을 지정한 후 필터를 지정하면 필터가 적용된 파일이 저장됩니다.

![image](https://user-images.githubusercontent.com/92137084/144740371-dcda6840-f204-4755-902f-424bb274218c.png)

![image](https://user-images.githubusercontent.com/92137084/144740504-6866c5df-a83e-409b-af1d-607241965375.png)

![image](https://user-images.githubusercontent.com/92137084/144740546-2f39f6f2-a125-49a7-a676-256f6eb5dc84.png)

## 개발 결과물 필요성 및 활용방안

현대 사회에 들어, 빠른 기술의 발달로 인해 많은 사람들이 스마트폰이라는 이름의 개인 카메라를 하나씩은 가지고 있는 세상이 되었습니다.

카메라가 많아진 만큼, 우리 모두는 우리의 추억을 사진이라는 이름으로 간직할 수 있게 되었습니다.

그렇기 때문에, 때로는 색다른 느낌의 추억을 남기고 싶기도 하고, 과거의 추억을 곱씹으며 새로운 의미를 부여하고 싶어질 수도 있을 것입니다.

그러한 요구에 부합하기 위해 인물은 보존하고 사진의 배경을 왜곡하여 색다른 느낌을 줄 수 있는 이 프로그램을 제작하였습니다.

때로는 친구들과 색다른 추억을 남기기 위해, 때로는 과거에 찍은 사진을 편집하며 추억을 곱씹기 위해 이 프로그램을 활용할 수 있을 것입니다.
