import numpy as np
import matplotlib.pyplot as plt
from method import *





back_img = cv2.imread('../demands/front.jpg')
back_img, H, W = auto_reshape(back_img, 640)
print(H,W)
plt.imshow(back_img)
plt.show()
# 中华人民共和国
# pt1, pt2 = [212,33], [557,74]
# 居民身份证
# pt1, pt2 = [189,106], [586,162]
# 签发机关
# pt1, pt2 = [141,292], [499,315]
# 有效期限
# pt1, pt2 = [141,345], [513,366]



x1, x2 = 42, 395
line = [41, 71, 102, 123, 151, 170, 203, 225, 238, 259, 334, 355]
# ROI
pt1, pt2 = [x1,line[0]], [x2,line[-1]]
cv2.rectangle(back_img, pt1, pt2, color=(0,255,0),thickness=1)
# line
for i in range(len(line)):
    pt1, pt2 = [x1,line[i]], [x2,line[i]]
    cv2.line(back_img, pt1, pt2, (0,0,255), 1)



cv2.imshow('21213', back_img)
cv2.waitKey(0)


# back_img = cv2.cvtColor(back_img)