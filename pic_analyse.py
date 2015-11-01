from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# define some parameters
im = Image.open("test.jpg")
pix = im.load()
x_max = im.size[0]
y_max = im.size[1]
rgb_vals = np.zeros((x_max*y_max,5),dtype=np.int)
x_vals = np.arange(0,x_max)
y_vals = np.arange(0,y_max)
k =0

def create_vals(i,j,k):
	rgb_vals[k,:] = np.array([i,j,pix[i,j][0],pix[i,j][1],pix[i,j][2]])
	k= k+1


map(lambda i : map(lambda j : create_vals(i,j,k),y_vals),x_vals)
rgb_vals = np.array(rgb_vals)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')	
# k_means = KMeans(init='k-means++', n_clusters=9, n_init=100)

# k_means.fit(Z)
# z.append(k_means.cluster_centers_)


# for x in k_means.cluster_centers_:
# 	plt.scatter(x[0],x[1])

# plt.show()

# ax.scatter(np.arange(1,10),np.arange(1,10),np.arange(1,10))


# fignum = 1
# for name, est in estimators.items():
#     fig = plt.figure(fignum, figsize=(4, 3))
#     plt.clf()
#     ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

#     plt.cla()
#     est.fit(R)
#     labels = est.labels_

#     ax.scatter(R[:, 0], R[:, 1], R[:, 2], c=labels.astype(np.float))

#     ax.w_xaxis.set_ticklabels([])
#     ax.w_yaxis.set_ticklabels([])
#     ax.w_zaxis.set_ticklabels([])
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     fignum = fignum + 1

# # Plot the ground truth
# fig = plt.figure(fignum, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

# plt.cla()

# # for name, label in [('Setosa', 0),
# #                     ('Versicolour', 1),
# #                     ('Virginica', 2)]:
#     # ax.text3D(R[y == label, 3].mean(),
#     #           X[y == label, 0].mean() + 1.5,
#     #           X[y == label, 2].mean(), name,
#     #           horizontalalignment='center',
#     #           bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# # Reorder the labels to have colors matching the cluster results
# # y = np.choose(y, [1, 2, 0]).astype(np.float)
# ax.scatter(R[:, 0], R[:, 1], R[:, 2])

# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

# fig = plt.figure(figsize=(10, 6.4))

# ax = fig.add_subplot(221)
# ax.set_title('Red')
# plt.imshow(R)
# ax.set_aspect('equal')
# ax2 = fig.add_subplot(223)
# ax2.set_title('Green')
# plt.imshow(G)
# ax2.set_aspect('equal') 
# ax3 = fig.add_subplot(212)
# ax3.set_title('Blue')
# plt.imshow(B)
# ax3.set_aspect('equal') 


# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# plt.colorbar()
# plt.show()
