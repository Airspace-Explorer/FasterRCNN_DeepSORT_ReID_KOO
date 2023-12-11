# Detection and Tracking of abnormal objects in the air in various weather environments


## 팀명: Airspace-Explorer 
- 지도 교수: 이대호
- 팀원:  
  소프트웨어융합학과 구현서 2018102091 (PM)  
  소프트웨어융합학과 김민환 2019102081  
  소프트웨어융합학과 이정원 2020110480  

  
## 1.개요
[조류 충돌]      
우리 말로 '조류 충돌'이라고 불리는 '버드 스트라이크(Bird Strike)'는 조류가 비행기에 부딪히거나 엔진 속에 빨려 들어가는 현상을 말한다. 주로 공항 부근, 그리고 이착륙 시 주로 발생하는데, 우리나라에서는 조류충돌 사고가 매년 100~200건 이상 발생 하고 있으며 지난해 미국 연방항공청에서는 1만 7천 건이 넘는 신고가 접수 되었다. 실제로 1.8kg의 새가 시속 960km로 비행하는 항공기와 부딪치면 64t 무게의 충격이 발생하며, 전 세계적 피해규모는 연간 약 1조원으로 추정된다. 최근 5년간 항공기-조류간 충돌은 주로 공항구역에서 발생하고 있으며, 이를 예방하기 위해 공항에서는 사격팀 운영, 천적류 사육을 통해 노력하고 있지만 효과가 미비한 상황이다. 그래서 본팀은 최소 수십명에서 많게는 수백명 이상의 사망자를 발생시키는 버드 스트라이크 방지를 위해 Faster-RCNN, YOLOF, SSD 기반의 조류를 실시간으로 Detection 하는 모델을 학습을 통해 구축하여 다양한 기상 환경에서 비행중인 상공물체(조류)를 실시간으로 탐지할 것이며, 또한 Detection 모델과 ReID 모델을 결합하여 DeepSORT 모델을 구축한 뒤 실시간으로 상공 물체(조류)를 추적 할 것이다.

[적성국 무인 항공기의 영공 침범]      
최근 북한의 무인 항공기의 영공 침입이 문제로 떠오르고 있다. 적의 고정익기와 회전익기, 중대형 무인기의 경우 탐지 및 식별이 쉽고, 금방 격추에 나설테지만 북한이 내세우는 소형 무인기들의 경우 이야기가 다르다. 철새나 풍선 같은 작은 동물이나 물체들 조차 미확인 항적으로 탐지되는 상황에서 상위 부대가 항적 식별에 나서느라 시간이 소요될 수 밖에 없는데, 본팀은 해당 문제를 해결하기 위해 무인 항공기 학습 이미지 데이터셋, 테스트 이미지 데이터셋을 수집하여 버드스트라이크 방지를 위한 모델에 전이학습(Transfer Learning)을 추가로 진행할 예정이다.

## 2.프로젝트 목표  
- 3개의 Object Detection Model(Faster-RCNN, YOLOF, SSD) Training & Evaluation & Visualization & Inference 및 성능 최적화 수행
- DeepSORT에 대한 Detection Model Training, Evaluation & ReID Model Training, Evaluation
- ReID Model을 사용한 DeepSORT Model과 ReID Model을 사용하지 않은 DeepSORT Model 간의 성능 비교

## 3.OpenMMLab Detection & Video Perception Toolbox and Benchmark  
MMdetection: https://github.com/open-mmlab/mmdetection  
MMtracking: https://github.com/open-mmlab/mmtracking  
## 4.Model Specification  
Faster-RCNN  
Backbone: ResNet50  
Neck: FPN  
RPN_head: RPNHead  
Classification Loss: CE(Cross Entropy) Loss  
Bounding Box Regression Loss: L1 Loss  

YOLOF  
Backbone: ResNet  
Neck: DilatedEncoder  
BBox_head: YOLOFHead  
Classification Loss: Focal Loss  
Bounding Box Regression Loss: GIoU Loss  

SSD  
Backbone: ResNet  
Neck: SSDNeck  
BBox_head: SSDHead  
Classification Loss: Localization Loss  
Bounding Box Regression Loss: IoU Loss  


  
## 5.공통 SPEC & Runtime Environment
[공통 Spec]  
Framework: MMDetection  
Learning_rate=0.02 / 8  
Workers_per_gpu: 4  
Batch_size: 16  
Epochs: 100  
Classes = ('Bird', 'Airplane', 'Helicopter', 'FighterPlane', 'Paragliding’ , 'Drone’)  
Visualization Tool: Tensorboard, Matplotlib  

[Runtime Environment]  
Sys.platform: linux  
Python: 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0]  
CUDA available: True  
GPU 0: NVIDIA GeForce RTX 3090  
CUDA_HOME: /usr/local/cuda  
NVCC: Cuda compilation tools, release 11.7, V11.7.99  
GCC: gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0  
PyTorch: 1.13.0+cu116  
PyTorch compiling details: PyTorch built with:(- GCC 9.3, - C++ Version: 201402,- Intel(R) 64        
architecture applications)  

  
## 6.Object Detection Datasets
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=476  
  
AI-Hub의 Small object detection을 위한 이미지 데이터셋을 이용하였다. 해당 데이터 셋에서는 이미지(2800x2100 해상도) 내에 일정 크기 이하의 소형 객체(200x200 픽셀 크기 이하)들만 존재하며 이미지에 대한 JSON 형태의 어노테이션 파일 또한 포함하고 있다.      
- Type: AircraftDataset (Classes: "Bird", "Airplane", "Helicopter", "FighterPlane", "Paragliding", "Drone")
- Train Datasets: 5760
- Validation Datasets: 1601
- Test Datasets: 640

  
## 7.Data Augmentation
추가로 다양한 환경에서의 객체 탐지율을 높이기 위해 학습 과정 중 아래와 같은 Data Augmentation 기법들을 적용하였다. 하지만 MMDetection은 파이프라인 내부에서 모델의 학습과 평가가 
이루어지기 때문에 Augmentation이 적용 된 이후의 정확한 Datasets의  Size를 식별하기 불가능하다는 Issue가 존재하였다.
- Brightness Distortion (이미지의 명도 변경)     
 → brightness_delta=32    
 → 최소값과 최대값이 각각 -32, 32인 균일 분포 함수의 Output을 통해 이미지의 명도를 변환 하였다.      
- Contrast Distortion (이미지의 대비 변경)   
  → contrast_range=(0.5, 1.5)   
  →최소값과 최대값이 각각 0.5, 1.5인 균일분포함수의 Output을 통해 이용해 이미지의 대비를 변환 하였다. Contras Distortion은 Brightness Distortion과는 다르게 Sum이 아닌 Multiplication 연산을 수행하여 Pixel Intensity의 대비를 증가시킨다  
- Saturation Distortion (이미지의 채도 변경)  
  → saturation_range=(0.5, 1.5)   
  → 최소값과 최대값이 각각 0.5, 1.5인 균일 분포 함수의 Output을 통해 이미지의 채도를 변환 하였다, H(Hue; 색조), S(Saturation; 채도), V(Value; 명도)에서 1의 인덱스를 갖는 S를 변경
- Hue Distortion (이미지의 색상 변경)  
  → hue_delta=18  
- Resize (이미지의 사이즈 변경)  
  → img_scale=(1333, 800)  
- RandomFlip (이미지 회전)  
  → flip_ratio=0.5  
- Normalize (Pixel Intensity Normalization)  


   
## 8.Object Detection Model 학습 수행 결과
### [Faster-RCNN]  
Faster-RCNN Learning Rate   
![lr](https://github.com/Airspace-Explorer/.github/assets/104192273/873b2eed-aace-4c9f-8fce-bb6413a8d5f9)  
Faster-RCNN Accuracy & Train_loss  
![faster_rcnn_train](https://github.com/Airspace-Explorer/.github/assets/104192273/3040eeab-fbe5-4061-bdf8-9f501ba6d8de)  
Faster_RCNN mAP  
![faster_map](https://github.com/Airspace-Explorer/.github/assets/104192273/898e30fd-ed3f-4d97-bfe3-048de2a0f300)  

### [YOLOF]
YOLOF Learning Rate  
![yolo lr](https://github.com/Airspace-Explorer/.github/assets/104192273/9c2f5579-3b5b-4e16-a076-31825d4be620)  
YOLOF Train_loss  
![yolo tr](https://github.com/Airspace-Explorer/.github/assets/104192273/823a2bd7-62cf-473a-af67-2c0aee0ca189)  
YOLOF mAP  
![yolo val](https://github.com/Airspace-Explorer/.github/assets/104192273/16939b6d-1027-4f0d-a04b-b45ea2858e41)

### [SSD]  
SSD Learning Rate  
![lr](https://github.com/Airspace-Explorer/.github/assets/104192273/873b2eed-aace-4c9f-8fce-bb6413a8d5f9)  
  
SSD Train_loss  
![DASD](https://github.com/Airspace-Explorer/.github/assets/104192273/55bf1b47-044f-4c34-bd13-0477b92a4d1d)  

SSD mAP  
![bfs](https://github.com/Airspace-Explorer/.github/assets/104192273/6637bf45-ac09-4a6c-9357-a4f59b9f7416)

  
## 9.모델 Inference 결과  
![asdxcz](https://github.com/Airspace-Explorer/.github/assets/104192273/1300493c-1966-4624-a7a2-fb22aa69c31e)  

## 10.Object Tracking Datasets  
Detector와 ReID Model 학습 수행을 위해 대중화된 MOT datasets의 보행자나 차량 class가 아닌 Bird class를 위한 Custom Datasets을 구축해야 했다.따라서 CVAT Tool을 이용하여 Track Rectangle로 각각의 객체를 지정한 다음 Frame마다 상자를 이동시켜 추적 좌표를 저장하고 객체가 화면에서 사라질시 Switch OFF시켜 Ground Truth 파일을 산출했다.gt.txt는 차례로 Frame number,Identity number,Bonding box left,Bounding box top,Bounding box width,Bounding box height,<Confidence score>,Class,Visibility순이다.Train Dataset은 Video를 Frame Per Second 단위로 분할 뒤 저장하였다.최종으로 MMtracking의 mot2coco.py,mot2reid.py 파일을 이용해 MOT형식의 COCO Format Annotation과 Bounding Box Image로 구성되어 있는 ReID Datasets으로 변환하여 Multi Object Tracking Datasets을 구축하였다.    
#### [CVAT tool을 활용한 Custom DataSet Labelling]  
![1241](https://github.com/Airspace-Explorer/.github/assets/104192273/59b7163d-749e-4f58-b803-7096178aefee)  
#### [산출된 Ground Truth 파일]  
![asdq](https://github.com/Airspace-Explorer/.github/assets/104192273/5c8f484a-8e8b-4c87-89fc-56e4fb71594d)  
#### [Video to Image]  
![vd2](https://github.com/Airspace-Explorer/.github/assets/104192273/a4335efb-0e8c-4877-9bd8-e54b9c202bb6)  
#### [최종 Multi Object Tracking Dataset Structure]  
![zaza](https://github.com/Airspace-Explorer/.github/assets/104192273/9c43470b-87b1-4bba-9918-653c85711dd5)  

  
## 11.DeepSORT를 이용한 Multi Object Tracking 결과  
MMTracking에서 제공하는 DeepSORT의 경우 Object Detection Model 과 ReID Model을 혼합하여 MOT을 수행할 수 있었다. Detection Model의 경우 본 팀이 구축한 Faster-RCNN, YOLOF, SSD의 Checkpoint 파일에 Tracking Video에 대한 전이 학습(Epochs: 10, Step: 10, Batch Size: 2, # of Training Datasets: 216)을 수행한 뒤 적용하였다.  ReID(Re-Identification) Model의 경우 사용되는 데이터셋의 객체 간의 구별되는 특징이 없는 경우, 객체의 식별이 어려워지고 성능이 제한될 수 있다. 즉 객체 간의 차이가 충분히 크지 않거나 유의미한 특징이 부족하면 다중 객체에 대한 정확한 식별과 추적이 어려워지고 Generalization Performance의 저하를 초래할 수 있다. 따라서 ReID Model을 효과적으로 학습시키기 위해서는 데이터셋이 객체 간의 유의미하고 구별되는 특징을 포함하고 있어야하는데 본 팀의 학습 데이터는사람과 같이 구별되는 특징을 가진 객체가 포함되지 않았기 때문에 ReID Model을 DeepSORT에 적용하였을 때 눈에 띄는 성능의 향상을 불러올 수 있을지 의문을 가지게 되었다. 그래서 ReID Model을 DeepSORT에 적용했을 때와 적용하지 않았을 때의 성능 비교 연구를 수행하였고 아래와 
같은 결과를 도출할 수 있었다.

### [ReID Model Training 결과] 
![image](https://github.com/Airspace-Explorer/.github/assets/43543906/00eb1655-a4e3-49be-9975-d496e1e37755)

### [Triplet Loss → 0.000 (figure 2)] 
Triplet Loss는 어떤 한 객체(Anchor)와 같은 객체(Positive), 다른 객체(Negative)이라는 파라미터를이용해 학습 시 미니 배치 안에서 Anchor, Positive, Negative들이 임베딩 된 값들의 유클리드 거리를 구해 아래와 같은 Loss 함수를 만든다.

  ![image](https://github.com/Airspace-Explorer/.github/assets/43543906/a9b14d42-8214-473b-902d-ee0aaff85f69)

대괄호 안의 첫번째 항이 의미하는 것은 Anchor와 Positive간의 Distance고, 두번째 항은 Anchor와   Negative와의 Distance이며 α는 마진(Hyper Parameter)을 의미한다. 따라서 L을 최소화한다는 것은 
Positive와의 거리는 가까워지도록 하고 Negative와의 거리는 멀어지도록 하는 것이다. 즉 Triplet 
Loss가 0.000이 나왔다는 것은 ReID 모델이 학습 중에 모든 Anchor, Positive, Negative 쌍에 대해 
거리를 올바르게 구별했다는 의미이며 Anchor와 Positive 간의 거리가 Negative 간의 거리보다 
작거나 같게 학습되었다는 것을 나타낸다. 즉 이는 모델이 이미 학습 데이터에서 제시된 유사성 및 
차이를 이해하고 있음을 시사할 수 있다.

### [Top-1 Accuracy: 99.2000 (figure 4)] 
Top-1 Accuracy란 Softmax Activation Function에서의 Output에서 제일 높은 수치를 가지는 값이 정답일 경우에 대한 지표를 계산한 것을 의미한다. 즉 Top-1 Accuracy가 99.2000와 같이 높은 수치가 나왔다는 것은 해당 ReID 모델이 대부분의 경우에 대해 가장 높은 확률을 가진 클래스를 정확하게 식별한다는 의미이며 이는 모델이 주어진 분류 작업을 잘 수행하고 있음을 나타낸다.

### [ReID Model을 사용하지 않은 경우 DeepSORT 성능 평가(MOTA: 85.3%, MOTP: 0.228)] 
  ![FER](https://github.com/Airspace-Explorer/.github/assets/104192273/6dfcb68a-6edb-416e-88b9-5cd538b56d39)
### [ReID Model을 사용한 경우 DeepSORT 성능 평가(MOTA: 93.0%, MOTP: 0.212)] 
  ![12345](https://github.com/Airspace-Explorer/.github/assets/104192273/224c379e-5003-445c-846c-78faec6fd15d)
  
본 팀의 예상과는 달리 DeepSORT를 이용한 MOT 수행시 ReID Model을 이용한 경우 MOTA(Multi Object Tracking Accuracy)성능이 7.7% 향상하며, MOTP(Multi Object Tracking Precision) 성능은 0.016% 향상한 것을 확인할 수 있었다. 또한 Recall과 Precision값이 각각 0.3%, 6.6% 증가하였다.

  
## 12.최종결과물 주요 특징 및 설명  
  
### [Object Detection]  
  
공통으로 ResNet계열의 BackBone 과 각기 다른 Neck,Head를 가지고 있다.학습 수행 결과에서 주요 차이점은 2-stage-detector Faster_RCNN에선 Feature Pyramid Network(FPN)을 사용하여 다양한 스케일로 feature map을 추출했고 Cross Entropy Loss로 class를 분류하여 L1 Loss를 통한 regression으로 높은 accuracy를 달성했다. 1-stage-detector인 YOLOF는 DilatedEncoder와 Focal Loss,GIoU Loss를 사용했고 SSD는 SSDNeck,SSDHead,Localization Loss,IoU Loss 를 사용하였다. 종합적인 결과: 모델 각각의 특징에 맞는 Neck,Head 적용과 Data augmentation을 적용하여 0.8이상의 높은 mAP를 보인다. 프로젝트를 통해 Small Size를 갖는 조류 및 비행물체를 높은 성능으로 탐지하고 실시간 추적하는 모델을 구축하였다. 따라서 해당 모델을 적용한다면, 조류나 비행물체 뿐만 아니라 다른 Small Object에 관한 높은 성능의 Detection 및 실시간 추적이 가능하다고 기대된다. 
  
### [Object Tracking]  
  
DeepSORT와 관련하여 이전에 발표된 논문들은 Re-identification 모델을 통해 사람과 같은 Object 간 고유하게 구별되는 특징을 갖는 데이터를 학습하고, 이로부터 Id-switching이나 Occlusion(폐색) 문제를 해결하였다. 하지만 본 프로젝트의 사용된 Training Datasets은 Small Size의 조류나 비행기, 드론과 같은 상공 비행 물체이기 때문에, 이전 논문들과 달리 하나의 Class내에서 Object들을 고유하게 분류할만한 특징이 없을 것이라 예상하였다. 그러나 Re-identification 모델 학습 유무에 따라 Object Tracking 성능이 달라지는 것을 확인하였고, 이로부터 Re-identification 모델이 다형성 및 활용성 부분에서 향상됨을 증명하였다.

  
## 13.Deep Sort 데모 영상 
![gif_deepSORT_result](https://github.com/Airspace-Explorer/.github/assets/43543906/22eb1f37-41a7-40a1-beb8-69a371fc6db8)

  
## 14.기대효과 및 활용 방안  
  
가. 기대효과  
- 항공 안전 향상→ 조류 및 무인 항공기의 실시간 탐지 및 추적을 통해 조종사들에게 충돌 사고를 예방하고 항공 안전성을 향상시킬 것으로 기대된다
- 비상 상황 대응 강화→ 물체 추적 기술을 활용하여 항공 담당자가 비상 상황에 신속하게 대응하고 효과적으로 관리할 수 있다.
- 자동화된 모니터링 시스템 도입→ 효율적인 감시 및 경보 시스템을 통해 공항 및 항공 당국이 실시간으로 항공 교통 상황을 모니터링하고 위험 지역을 식별할 수 있다.
- 다양한 산업 분야에의 응용→ 해상 감시 및 경계 보안, 자연보전 및 환경 모니터링 등 다양한 분야에 물체 탐지 및 추적 기술을 응용하여 향상된 안전 및 효율성을 제공할 수 있다.
- 연구 및 교육 활용→ 수집된 데이터셋과 모델은 학계와 산업계에서 객체 탐지 및 추적에 관한 연구와 교육에 활용될 것으로 기대된다.
  
나. 활용방안
- 항공 관련 기관 및 공항 운영→ 공항 및 항공 당국은 개발된 시스템을 도입하여 항공 안전성을 향상시키고, 실시간 모니터링을 통해 항공 교통을 효율적으로 운영할 수 있다.
- 비상 상황 대응 및 관리 기관→ 비상 상황 대응 기관은 실시간 추적 결과를 활용하여 사고 현장에 빠르게 대응하고, 상황을 효과적으로 관리할 수 있다.
- 해상 감시 및 경계 보안→ 해군 및 경비 당국은 물체 탐지 및 추적 기술을 해상 감시에 활용하여 침입이나 위험한 상황을 신속히 감지하고 대응할 수 있다.
- 환경 보전 및 생태학 연구→ 환경 당국 및 연구 기관은 물체 추적 기술을 이용하여 조류 및 동물의 이동을 연구하고 환경 보전 활동에 활용할 수 있다.
- 연구 및 교육 기관→ 대학 및 연구 기관은 수집된 데이터셋과 모델을 활용하여 객체 탐지 및 추적에 관한 연구를 수행하고, 교육 과정에 활용할 수 있다.

    
## 15.결론 및 제언  
프로젝트 결과로 얻은 모델과 데이터는 항공 및 국방 관련 산업에서의 안전 및 효율성 향상에 큰 기여를 할 것으로 기대된다. 미래에는 더 많은 데이터를 수집하고 모델을 튜닝하여 다양한 상황에서의 적용 가능성을 높일수 있을것이다.



  
  














  

