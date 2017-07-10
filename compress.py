import numpy as np
import matplotlib.image as img
from sklearn.cluster import KMeans



def kMeansInitCentroids(X, K):
    centroids = np.zeros((K, X.shape[1]))
    randidx=np.random.permutation(X.shape[0])
    centroids = X[randidx[range(0,K)],:]
    return centroids

def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros((X.shape[0],1))
    D = np.zeros((X.shape[0], K))
    for i in range(0,K):
        A=np.subtract(X, centroids[i,:])
        D[:,i]=np.sum(np.power(A,2),axis=1)
    idx = np.argmin(D,axis=1)
    return  idx

def computeCentroids(X, idx, K):
    m,n=X.shape
    centroids = np.zeros((K,n))
    for i in range(0,K):
        X_clean=X[idx==i,:]
        if X_clean.size>0:
            centroids[i,:] = np.mean(X_clean,axis=0)
    return centroids

K=16
max_iters = 10

image = img.imread('image1.png')
img_size = image.shape
X = np.reshape(image,(img_size[0]*img_size[1],3))
X=X.astype(float)

centroids=kMeansInitCentroids(X,K)
idx=np.zeros((X.shape[0],1))
for i in range(0,max_iters):
    print(i+1,'/',max_iters)
    idx=findClosestCentroids(X,centroids)
    centroids=computeCentroids(X,idx,K)

print('Almost done...')
idx = findClosestCentroids(X,centroids)
X_recovered = centroids[idx,:]
X_recovered=np.reshape(X_recovered, (img_size[0],img_size[1],3))
img.imsave('image_comp.png', X_recovered)
print('Done')

print('Running K-Means from sklearn.cluster')
kmeans = KMeans(n_clusters=16)
kmeans.fit(X)
X_new = kmeans.cluster_centers_[kmeans.labels_,:]
X_new=np.reshape(X_new, (img_size[0],img_size[1],3))
img.imsave('image_comp1.png', X_new)
print('Done')
