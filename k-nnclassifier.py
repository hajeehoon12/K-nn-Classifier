import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
from math import sqrt
import numpy as np
import warnings
#-------------------------------------------------
import sklearn.utils._typedefs # 쓰는곳도없는데 빼면오류남 .std axis 쪽인듯
from sklearn.model_selection import train_test_split # 데이터 나누기 위한 import
#------------------------------------------------- 극히일부사용 편리용
import time

warnings.filterwarnings('ignore')
print("satisfaction_data.csv의 파일명에 해당하는 데이터파일을 같은 디렉토리안에 넣으세요. ") 

#file_name=input()# 사용시 입력받는 값이 하나씩 밀려서 오류가남

col_names = ['Hours_per_week', 'Workclass','Martial-status', 'Fnlwgt','Age','Educational-num','satisfaction']
dataset = pd.read_csv("satisfaction_data.csv", encoding='UTF-8', header=None, names=col_names)
print("분류중입니다....")
start = time.time() # 시간측정 시작
X_1 = dataset.iloc[:,0:4].to_numpy() # DataFrame을 np.ndarray로 변환
X_2 = dataset.iloc[:,4:6].to_numpy() # 이중 분석에 필요없는 교육번호를 제외한 0,1,2,4,5의 column을가져옴 -> 수정)제외한걸다시넣음
X = np.hstack((X_1, X_2)) # 분리된 np.array부분을 합침

y = dataset.iloc[:, -1].to_numpy() # 만족도 column만 분리
#print(y)


def dimension_decrease(dataz):
    #dataz = (dataz-dataz.min())/(dataz.max() - dataz.min()) # 표준화 진행
    norm_dataz = dataz-dataz.mean(axis=0)
    norm_dataz = norm_dataz/dataz.std(axis=0) # 표준화진행 2 dataz
    
    cov_norm_dataz = np.cov(norm_dataz.T) # Convariance Matrix 구하기
    
    eigen_val, eigen_vec = np.linalg.eig(cov_norm_dataz)
    #print(eigen_val)
    #print(eigen_vec)

    z1 = eigen_vec[:,0][0] * norm_dataz[:,0] + eigen_vec[:,0][1] * norm_dataz[:,1] + eigen_vec[:,0][2] * norm_dataz[:,2]
    z2 = eigen_vec[:,1][0] * norm_dataz[:,0] + eigen_vec[:,1][1] * norm_dataz[:,1] + eigen_vec[:,1][2] * norm_dataz[:,2]
    z3 = eigen_vec[:,2][0] * norm_dataz[:,0] + eigen_vec[:,2][1] * norm_dataz[:,1] + eigen_vec[:,2][2] * norm_dataz[:,2]
    z4 = eigen_vec[:,3][0] * norm_dataz[:,0] + eigen_vec[:,3][1] * norm_dataz[:,1] + eigen_vec[:,3][2] * norm_dataz[:,2]
    z5 = eigen_vec[:,4][0] * norm_dataz[:,0] + eigen_vec[:,4][1] * norm_dataz[:,1] + eigen_vec[:,4][2] * norm_dataz[:,2]
    z6 = eigen_vec[:,5][0] * norm_dataz[:,0] + eigen_vec[:,5][1] * norm_dataz[:,1] + eigen_vec[:,5][2] * norm_dataz[:,2]
    dataz_pca_res = np.vstack([z1,z2,z3,z4,z5,z6]).T # X_train에 대한 PCA를 통한 차원 축소완료
    #print(dataz_pca_res[:,:2])
    return dataz_pca_res[:,:6] # 원하는 차원수만큼 축소해서 사용하면됨


def euclidean_distance(row1, row2): # 다차원 유클리디언 거리측정
	distance = 0.0
	for i in range(len(row1)):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

def k_nn_discrimination(X_train, y_train,k,row0,d_d): # k는 k값 row0은 테스트 데이터에서 판별하고자 하는 featue의 값

    #k = 29 # k값 최초설정 임의의값 나중에 변하게 만들꺼 홀수값

    distance_list = []
    #d_d = dimension_decrease(X_train) 원래안에있었지만 계산량이 많아져서 밖으로뺌
   # 여러번 사용하기에 변수화시킴 , 원하는차원으로 차원축소시킨 feature값
    #row0 = [0.2,0.2] # 가짜 feature 나중에 내가 구하고자 하는 값 넣을거
    
    for row in d_d:
        distance_list.append(euclidean_distance(row0, row)) # 테스트 데이터 row0 과 훈련데이터 row를 비교  
    #print(d_d.shape)        # feature값 차원확인
    #print(y_train.shape)    # train의 정답 차원확인

    
    # 측정거리를 순서대로 나열 k에 해당하는 거리를 찾기위해
    sorted_list = sorted(distance_list)
    index = []
    #print(len(sorted_list)) # 제대로 출력되는지 길이확인
    #print(sorted_list) # 출력값확인
    for i in range(k):
        index.append(distance_list.index(sorted_list[i])) # k개에 해당하는 가장가까운 값들을 훈련 데이터에서 찾아서 index list에 index값으로넣음

    count = 0
    for i in index:
        #print(i)
        if y_train[i] == 'satisfied': # 만약 그 안에있는 값이 satisfied이면
            count+=1 # 카운터를 1증가시킴
    if count*2 >= k : # 카운터 값이 k의 절반이상이면
        result = 0 # 0이면 satisfied
    else:
        result = 1 # 1이면 unsatisfied로 판단한다.
    return count, result # count = k개안에있는 satisfied의 수 result = 0이면 satisfied로 판별 1이면 unssatisfied로 판별
    #print(y_train[1])
    #print(count)




correct =0# 초기화
iterationz = 10
result_final = []


for count in range(iterationz):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) # 데이터 세트를 1:9로 나눔
    X_test_2d=dimension_decrease(X_test) # 2차원특징 벡터로 축소시킨 X_test_2d -> 수정 다차원으로 변경
    d_d = dimension_decrease(X_train) 
    result_k_nn_result_Test = []
    #result_k_nn_count = [] # 안에있는 count의 값 분석할때 쓸것
    for i in X_test_2d:
        T=k_nn_discrimination(X_train, y_train,29,i,d_d) # 판별시행
        #result_k_nn_count.append(T[0]) # 안에있는 count 값 분석할때 쓸것
        result_k_nn_result_Test.append(T[1])    #dicrimination에서 나온 결괏값추출


    result_sat = []
    for i in range(len(y_test)):            # 테스트 데이터에 대해 맞춘갯수 세기
        if y_test[i] == 'satisfied':        # 데이터추출
            z = 0
        
        if y_test[i] == 'unsatisfied':
            z = 1
        
        if z==result_k_nn_result_Test[i]:
            correct +=1
    for i in range(len(result_k_nn_result_Test)):   # 만족인 불만족인지 데이터 추출
        if result_k_nn_result_Test[i] == 0 :
            result_sat.append('satisfied')
        if result_k_nn_result_Test[i] == 1:
            result_sat.append("unsatisfied")
    result_final.extend(result_sat)
    if count == 0 :   ## 처음에는 결괏값이 비어있기에 차원합병이불가능하기에 차원초기화해줌
        X_fin = X_test
        y_fin = y_test
    else:           ## 처음다음에는 합병
        X_fin=np.vstack ([X_fin, X_test])
        y_fin=np.vstack ([y_fin, y_test])
accuracy = correct/((len(y_test)*iterationz))        # 정확도 측정
print("결과와 예측이 일치한 총 개수 = ", correct)
print("accuracy = ",accuracy)           # 정확도 출력

print("분석한 결과를 기존파일 맨오른쪽칸에 예측결과를 추가")

#print(len(X_train), len(X_test)) # 잘나뉘었는지 확인

#print(X_train[:3]) # 임의로 나뉜 x_train에서 3개에 data에 대해 검사
#print(y_train[:3]) # 임의로 나뉜 y_train에서 3개의 data에 대해 검사


#print(X_train) # 표준화된 X_train 확인

#print(dataset.shape) # (row개수, column개수)
#print(dataset.info()) # 데이터 타입, row 개수, column 개수, 컬럼 데이터 타입
#print(dataset.describe()) # 요약 통계 정보


#print(k_nn_discrimination(X_train, y_train,100,[0.2,0.2])) 제대로 실행되는지 test
result_data_csv=np.hstack([X_fin,y_fin.reshape((iterationz*len(y_test)),1),(np.array(result_final).reshape(len(result_final),1))])
dataframe= pd.DataFrame(result_data_csv,columns = ['Hours_p_w', 'Wclass','Martial-st', 'Fnlwgt','Age','E-num','satis','expect-sat'])
dataframe.to_csv("20164064.csv",header=False,index=False)    # data를 헤더포함시켜서 csv파일로불러내기 # header =true로 바꾸면 헤더가표현된다.
print("파일출력완료")
print("time(걸린시간) :", time.time() - start) # 실행시간측정
input("Input any key to End the program...")

