#import libraries
import cv2
from sklearn.cluster import KMeans

# loading image
img = cv2.imread('mountain.jpg')
print(img.shape)

#change 3D image to 2D image 
x, y, z = img.shape
image_2d = img.reshape(x*y, z)
print(image_2d.shape)

# set n to the number of cluster that you want 
n = 1
kmeans_cluster = KMeans(n_clusters=n)
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_
# change the color of the cluster with a center value of image
for x in range(len(cluster_labels)):
    image_2d[x] = cluster_centers[cluster_labels[x]]

#make a new 3D image
imgFinall = image_2d.reshape(img.shape)
cv2.imshow('pic', imgFinall)
#name of the file for saving
IMG_name = 'clustering'+str(n)+'.jpg'
cv2.imwrite(IMG_name, image_2d)
cv2.waitKey(0)
cv2.destroyAllWindows()


