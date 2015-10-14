from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im = Image.open("test.jpg")
pix = im.load()
x_max = im.size[0]
y_max = im.size[1]
R = np.zeros(im.size)
# print "x_max = {0} y_max = {1}".format(x_max,y_max)
G = np.zeros(im.size)
B = np.zeros(im.size)


for i in range(1,x_max):
	for j in range(1,y_max):
		R[i,j] = pix[i,j][0]
		G[i,j] = pix[i,j][1]
		B[i,j] = pix[i,j][2]
		# A[i,j] = pix[i,j][3]

fig = plt.figure(figsize=(10, 6.4))

ax = fig.add_subplot(221)
ax.set_title('Red')
plt.imshow(R)
ax.set_aspect('equal')
ax2 = fig.add_subplot(223)
ax2.set_title('Green')
plt.imshow(G)
ax2.set_aspect('equal') 
ax3 = fig.add_subplot(212)
ax3.set_title('Blue')
plt.imshow(B)
ax3.set_aspect('equal') 


cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar()
plt.show()
