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

## 개발 결과물 사용법 소개

리눅스 터미널을 통해 사용됩니다. 

./main.py 파일명 을 입력하면 해당 이미지를 편집하게 됩니다.

만일 파일명을 입력하지 않았거나, 하나 이상의 파일명을 입력하거나, 존재하지 않는 파일을 입력하는 등 잘못된 방식으로 사용 시 올바른 사용법을 출력합니다.

![image](https://user-images.githubusercontent.com/92137084/144740212-5fbafcc5-456a-4c6e-b4d3-ac08344798b4.png)

올바른 방식으로 사용한 경우, 사용법과 함께 사진의 영역 지정 창이 생성됩니다.

![image](https://user-images.githubusercontent.com/92137084/144740237-c5301ed9-39c5-470b-80e9-5d7c8a4d2e6c.png)

사용법에 따라 영역을 지정한 후 필터를 지정하면 필터가 적용된 파일이 저장됩니다.

![image](https://user-images.githubusercontent.com/92137084/144740371-dcda6840-f204-4755-902f-424bb274218c.png)

![image](https://user-images.githubusercontent.com/92137084/144740504-6866c5df-a83e-409b-af1d-607241965375.png)

![image](https://user-images.githubusercontent.com/92137084/144740546-2f39f6f2-a125-49a7-a676-256f6eb5dc84.png)



