import numpy as np
import random
import cv2
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.measure import label

# 最大连通域分割
def largestConnectComponent(bw_img):
    labeled_img, num = label(bw_img, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(0, num+1): 
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return ~lcc


# 图像reshape
def auto_reshape(img, lim):
    h, w = img.shape[:-1]
    if h > w:
        w =  int(lim * w / h) 
        h = lim
    else:
        h = int(lim * h / w)
        w = lim 
    img = cv2.resize(img, dsize=(w, h))
    return img, h, w

# 绘制线段
def draw_line(lines, img):
    for i in range(lines.shape[0]):
        rho, theta = lines[i,:]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        a,b,c = random.randint(0,255),random.randint(0,255),random.randint(0,255)
        cv2.line(img, (x1,y1), (x2,y2), (a,b,c), 1)
    

# 自定义模板匹配
def templet_matching(img, line, x1, x2):
    judge = 1
    cnt_all = 0
    for i in range(len(line) - 1):
        cnt = 0
        if(judge>0):
            for row in range(line[i], line[i + 1]):
                cnt += np.sum(img[row, :]>0)>0
            cnt_all += cnt / (line[-1] - line[0])
            judge *= -1
        else:
            for row in range(line[i], line[i + 1]):
                cnt += np.sum(img[row, :]>0)<3
            cnt_all += cnt / (line[-1] - line[0])
            judge *= -1            
    return cnt_all


# 身份证反面感兴趣区域匹配
def extract_ROI_back(img):
    x1, x2 = 170, 586
    line = [33, 74, 106, 162, 292, 315, 345, 366]

    img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(img_bin, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 15)
    img_bin = cv2.medianBlur(img_bin, 3)

    matching_rate = templet_matching(img_bin, line, x1, x2)
    return matching_rate

    # ROI
    pt1, pt2 = [x1,line[0]], [x2,line[-1]]
    cv2.rectangle(img_bin, pt1, pt2, color=(255,255,255),thickness=1)
    # line
    for i in range(len(line)):
        pt1, pt2 = [x1,line[i]], [x2,line[i]]
        cv2.line(img_bin, pt1, pt2, (255,255,255), 1)
    cv2.imshow('roi', img_bin)
    cv2.waitKey(0)




# 身份证正面感兴趣区域匹配
def extract_ROI_front(img):
    x1, x2 = 42, 395
    line = [41, 71, 102, 123, 151, 170, 203, 225, 238, 259, 334, 355]

    img_bin = img[:,:,0]
    # img_bin = cv2.Canny(img_bin, 100, 150, apertureSize=3)
    img_bin = cv2.adaptiveThreshold(img_bin, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 15)
    img_bin = cv2.medianBlur(img_bin, 3)

    matching_rate = templet_matching(img_bin, line, x1, x2)
    return matching_rate

    # ROI
    pt1, pt2 = [x1,line[0]], [x2,line[-1]]
    cv2.rectangle(img_bin, pt1, pt2, color=(255,255,255),thickness=1)
    # line
    for i in range(len(line)):
        pt1, pt2 = [x1,line[i]], [x2,line[i]]
        cv2.line(img_bin, pt1, pt2, (255,255,255), 1)
    cv2.imshow('roi', img_bin)
    cv2.waitKey(0)






# 计算所有直线的w, b
def line2grad(pts):
    k, b = [], []
    for i in range(pts.shape[0]):
        x1, x2, y1, y2 = pts[i, :]
        if x2 - x1 == 0:
            k.append((y2 - y1) / (1e-8))
        else:
            k.append((y2 - y1) / (x2 - x1))
        b.append(y1 - k[-1] * x1)
    return k, b



# 计算直线两两之间夹角
def calc_angle(k, i, j):
    ki, kj = k[i], k[j]
    theta = abs((kj - ki) / (kj * ki + 1 + 1e-10))
    return theta * 180 / np.pi


# 计算直线上的两点
def line2point(lines):
    # lines = lines.reshape(-1,2)
    pt = []
    for i in range(lines.shape[0]):
        rho,theta = lines[i,:]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = x0 + 1000*(-b)
        y1 = y0 + 1000*(a)
        x2 = x0 - 1000*(-b)
        y2 = y0 - 1000*(a)
        pt.append([x1, x2, y1, y2])
    return np.array(pt)


# 计算角点
def calc_point(pts, k, b, h, w, img):
    corner = []
    for i in range(pts.shape[0]):
        for j in range(i,pts.shape[0]):
            if calc_angle(k, i, j) > 15:

                x = (b[j] - b[i]) / (k[i] - k[j])
                y = int(k[j] * x + b[j])
                x = int(x)
                if(0<x<w and 0<y<h):
                    corner.append([x, y])
    return np.array(corner)


# 聚类
def edge_grads(corner):
    # scaler = StandardScaler()
    # grads = scaler.fit_transform(grads) #  归一化
    kmeans = KMeans(n_clusters=4, init='k-means++')
    kmeans.fit_predict(corner)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return centroids, labels


# 裁剪背景区域
def background_clip(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n = 1
    gray, Threshold =140, 0.7
    img = img[n:-n,n:-n] if n else img
    axis = []
    i = 0
    while(np.sum(img[i, :]>gray) / img.shape[1] < Threshold and i<20):
        i += 1
    axis.append(i+n)
    i = 0
    while(np.sum(img[-i, :]>gray) / img.shape[1] < Threshold and i<20):
        i += 1
    axis.append(i+n)
    i = 0
    while(np.sum(img[:, i]>gray) / img.shape[0] < Threshold and i<20):
        i += 1
    axis.append(i+n)
    i = 0
    while(np.sum(img[:, -i]>gray) / img.shape[0] < Threshold and i<20):
        i += 1
    axis.append(i+n)
    return axis

# 锐化
def sharp(img):
    ker = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    dst = cv2.filter2D(img, cv2.CV_8U, ker)
    return dst


# 利用国徽颜色特征判断是否是身份证反面
def judge_back_by_emblem(img):
    dst_r = img[:,:,2]
    dst_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst_diff = dst_r - dst_gray
    dst_diff[dst_diff>180] = 0
    dst_diff[dst_diff<30] = 0
    if np.mean(dst_diff[28:149,29:160]) > 1:
        return 1
    else:
        return 0



# 通过"中华人民共和国"字符数判定是否反面
def judge_back_by_PRC(dst):
    PRC = cv2.cvtColor(dst[25:82,200:570,:], cv2.COLOR_BGR2GRAY)
    PRC = cv2.medianBlur(PRC, 3)
    PRC = cv2.adaptiveThreshold(PRC, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 15)
    k = np.ones((3,3),np.uint8)
    PRC = cv2.dilate(PRC,k,iterations = 5)
    PRC = cv2.erode(PRC,k,iterations = 4)
    # print(count_blank_edge(PRC), count_blank(PRC))
    # cv2.imshow('wdwd', PRC)
    # cv2.waitKey(0)
    # 四周必须无遮挡
    if count_blank_edge(PRC):
        return 0
    # 字符数 = 8-1
    if count_blank(PRC) == 8:
        return 1
    else:
        return 0



# 通过"居民身份证"字符数判定是否反面
def judge_back_by_IDCard(dst):
    IDCard = dst[85:180,180:595,0]
    IDCard = cv2.GaussianBlur(IDCard, (5, 5), 0) 
    IDCard = cv2.Canny(IDCard, 100, 150, apertureSize=3)
    k = np.ones((1,1),np.uint8)
    IDCard = cv2.erode(IDCard,k,iterations = 2)
    k = np.ones((3,3),np.uint8)
    IDCard = cv2.dilate(IDCard,k,iterations = 7)

    # 四周必须无遮挡
    if count_blank_edge(IDCard):
        return 0
    # 字符数 = 6-1
    if count_blank(IDCard) == 6:
        return 1
    else:
        return 0



# 通过统计身份证ID字符数判定是否正面
def judge_front_by_ID(dst):
    ID = cv2.cvtColor(dst[326:364,210:570,:], cv2.COLOR_BGR2GRAY)
    ID = cv2.medianBlur(ID, 5)
    ID = cv2.adaptiveThreshold(ID, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 15)
    k = np.ones((3,3),np.uint8)
    ID = cv2.dilate(ID,k,iterations = 1)
    # 四周必须无遮挡
    if count_blank_edge(ID):
         return 0
    # 身份证ID数 = 19-1
    if count_blank(ID) == 19:
        return 1
    else:
        return 0




# 通过空白间距判定字符数量
def count_blank(img):
    space = 0
    last = True
    for i in range(img.shape[1]):
        judge_space = np.sum(img[:,i]) 
        if(judge_space == 0 and last):
            space += 1
            last = False
        if(judge_space != 0 and not last):
            last = True
    return space




# 判定四边是否有字符
def count_blank_edge(img):
    if np.sum(img[0, :] >0):return True
    if np.sum(img[-1, :]>0):return True
    if np.sum(img[:, 0] >0):return True
    if np.sum(img[:, -1]>0):return True
    return False




# 各种方法进行模板匹配
def judge_front_back(dst, is_video):
    # 判断正反面并返回结果
    # 视频流检测(判定条件少，速度更快)
    if is_video:
        if judge_back_by_emblem(dst):
            return 0
        elif extract_ROI_back(dst) > 0.85:
            return 0
        elif extract_ROI_front(dst) > 0.75:
            return 1
        elif judge_front_by_ID(dst):
            return 1
    # 图像检测(更精确，速度更慢)
    if not is_video:
        back_judge, front_judge = 0,0
        if extract_ROI_back(dst) > 0.85:
            back_judge +=0.5
        if judge_back_by_emblem(dst):
            back_judge +=0.3
        if judge_back_by_PRC(dst):
            back_judge += 0.1
        if judge_back_by_IDCard(dst):
               back_judge += 0.1
        if judge_front_by_ID(dst):
            front_judge +=0.4
        if extract_ROI_front(dst) > 0.75:
            front_judge +=0.6
        if(front_judge < back_judge):
            return 0
        if(back_judge < front_judge):
            return 1  
    return -1



# 先验的选择当前帧的类别
def calc_video_label(label, hull, container):
    # 如果没有检测到角点，说明可能翻面或者更换卡片了，重新计数
    if hull is None:
        container = [0,0]
        return -1, container
    # 如果检测到为未知，则不进行计数，返回出现次数最多的类别
    if(sum(container) != 0 and label == -1):
        return container.index(max(container)), container
    # 否则返回出现次数最多的类别
    else:
        container[label]+=1
        return container.index(max(container)), container



# 计算两点之间的距离
def calc_dist(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    return x * x + y * y


# 计算四个角点两两之间的距离
def calc_four_corner_dist(corner):
    dist = []
    for i in range(corner.shape[0]):
        for j in range(i+1, corner.shape[0]):
            dist.append(calc_dist(corner[i,:], corner[j,:]))
    return dist


# 计算当前帧角点与上一帧距离最近的角点的下标
def calc_last_current_corner_dist(cor, last_cor):
    index = [0,0,0,0]
    for i in range(cor.shape[0]):
        min_dist = -1
        min_idx = -1
        for j in range(last_cor.shape[0]):
            dist = calc_dist(cor[i,:], last_cor[j,:])
            if (min_dist == -1 or dist < min_dist):
                min_dist = dist
                min_idx = j
        index[i] = min_idx
    return index




# 透视变换
def perspective_transform(img, cor, target_cor, h, w):
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(cor, target_cor)
    # 透视变换
    dst = cv2.warpPerspective(img, M, (h, w))
    # 裁剪背景区域
    u, b, l, r =background_clip(dst)
    dst = dst[u:-b,l:-r,:]
    dst = cv2.resize(dst, dsize=(h, w)) 
    return dst



# 识别某帧的某张透视变换结果
def recognize_single_perspective_frame(img, hull, target_cor, direction, h, w, is_video, pred_container=None):
    # 原始角点坐标
    cor = np.float32(hull[direction,:]) # 逆时针有序
    # 透视变换
    dst = perspective_transform(img, cor, target_cor, h, w)
    # 识别
    if pred_container is None or max(pred_container) < 30:
        judge = judge_front_back(dst, is_video)    
    else:
        # 直接返回最大类别
        judge = pred_container.index(max(pred_container))
    return dst, judge, cor



# 矫正 + 识别 pipeline
def transform_recognize(img, hull, is_video, pred_container=None, last_corner=None):
    # 若未检出到身份证四个角点
    if( hull is None or hull.reshape(-1, 2).shape[0]!=4): 
        if is_video:
            return np.zeros((403, 640)), -1, -np.ones((4,2))
        else:
            return np.zeros((403, 640)), -1
            
    hull = hull.reshape(-1, 2)
    w, h, = 403, 640 # 变换区域大小
    # 变换后角点坐标(按照标准身份证模板的比例)
    target_cor = np.float32([[0,0], [0, w], [h, w], [h, 0]])


    # 视频流检测
    if is_video:
        # 启发式的选取角点区域
        if(last_corner is not None and last_corner[0,0]>=0):
            # 启发法计算正确的透视变换角点位置
            direction = calc_last_current_corner_dist(hull, last_corner)
            # 识别某帧的某张透视变换结果
            dst, judge, cor = recognize_single_perspective_frame(img, hull, target_cor, direction, h, w, 1, pred_container)
            if(judge !=-1): 
                return dst, judge, cor
        # 上一帧定位失败则重新定位
        else:
            # 选择遍历出正确的透视变换角点位置
            direction = {0:[3,2,1,0], 1:[2,1,0,3], 2:[1,0,3,2], 3:[0,3,2,1]}
            # 遍历四个方向寻找正确的角度
            for i in range(4):
                # 识别某帧的某张透视变换结果
                dst, judge, cor = recognize_single_perspective_frame(img, hull, target_cor, direction[i], h, w, 1)
                if(judge !=-1): 
                    return dst, judge, cor
        
        return dst, -1, -np.ones((4,2))
    # 图像检测
    else:
        # 选择遍历出正确的透视变换角点位置
        direction = {0:[3,2,1,0], 1:[2,1,0,3], 2:[1,0,3,2], 3:[0,3,2,1]}
        # 遍历四个方向寻找正确的角度
        for i in range(4):
            # 识别某帧的某张透视变换结果
            dst, judge, cor = recognize_single_perspective_frame(img, hull, target_cor, direction[i], h, w, 0)
            if(judge !=-1): 
                return dst, judge    
        return dst, -1

    
  



def show_result(img, target, label, hull, fps=None):
    # 类别信息
    cls = {0:"back side of ID card", 1:"front side of ID card",-1:"I dont Know"}
    cv2.putText(img, cls[label], (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    if fps is not None:
        cv2.putText(img, 'FPS:'+str(fps), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255,0, 255), 2)
    if(hull is None or hull.reshape(-1,2).shape[0]!=4): 
        cv2.imshow('Origin', img)
    else:
        # 绘制定位框
        cv2.polylines(img, [hull], isClosed=True, color=(0,0,255), thickness=2)
        for pt in hull.reshape(-1,2):
            cv2.circle(img, pt, 8, (0,255,0), 2)
            cv2.putText(img, '('+ str(pt[0]) + ',' + str(pt[1]) +')', pt-25, cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.imwrite('../exp/corner_points_location.jpg', img)
        # show
        cv2.imshow('Perspective transformation correction', target)
        cv2.imshow('Origin', img)