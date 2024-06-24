import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

# 두 수직선 A, B에 위치한 점들의 데이터
data = np.array([[1, 5], [1, 10], [1, 15], [1, 20], [2, 5], [2, 10], [2, 15], [2, 20], [2, 24], [2, 26]])

# 가중치를 사용하여 수직선 그룹에 따라 거리를 변경할 거리 함수를 정의
def modified_distance(x, y):
    x_diff = abs(x[0] - y[0])
    y_diff = abs(x[1] - y[1])
    
    # 가중치 10을 곱해 수직선(A, B)에 따라 거리에 큰 영향을 줍니다.
    return x_diff * 50 + y_diff

# 거리 행렬 생성
dist_matrix = pdist(data, metric=modified_distance)
dist_matrix = squareform(dist_matrix)

# 사전 계산된 거리 행렬을 사용하여 클러스터링 수행
clustering = AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="average")
labels = clustering.fit_predict(dist_matrix)

print("클러스터 레이블:", labels)

# 각 군집의 점들을 추출합니다.
cluster_A = data[labels == 0]
cluster_B = data[labels == 1]

print("군집 A의 점들:\n", cluster_A)
print("군집 B의 점들:\n", cluster_B)