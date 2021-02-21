import cv2
from sklearn.cluster import KMeans

img = cv2.imread('mypic.jpg')
print(img.shape)

x, y, z = img.shape
image_2d = img.reshape(x*y, z)
print(image_2d.shape)

n = 1
kmeans_cluster = KMeans(n_clusters=n)
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_
for x in range(len(cluster_labels)):
    image_2d[x] = cluster_centers[cluster_labels[x]]
    
imgFinall = image_2d.reshape(img.shape)
cv2.imshow('pic', imgFinall)
IMG_name = 'clustering'+str(n)+'.jpg'
cv2.imwrite(IMG_name, image_2d)
cv2.waitKey(0)
cv2.destroyAllWindows()


