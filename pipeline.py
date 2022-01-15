from method import *
import glob
import time

# 前景背景分割pipeline
def Fore_Background_Segmentation(RGB_img):
    # 灰度化
    GRAY_img = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2GRAY)
    # 先高斯滤波，缓解噪声
    # GRAY_img = cv2.GaussianBlur(GRAY_img, (9, 9), 0)
    # 这里可以换成均值滤波
    GRAY_img = cv2.medianBlur(GRAY_img, 21) 
    # 再自适应阈值分割，区分前景，背景
    Bin_img = cv2.adaptiveThreshold(GRAY_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
    # 再分割最大连通区域，提取背景后，前景变成mask(去除前景中内容的影响, 能够保证边缘线一定在边缘上)
    Bin_img = largestConnectComponent(Bin_img).astype('uint8') * 255
    return Bin_img






# 身份证定位pipeline, 返回四个角点坐标
def IDcard_detection(RGB_img, Bin_img, H, W):
    # 提取前景和背景的边缘
    Bin_img = cv2.Canny(Bin_img, 100, 150, apertureSize=3)
    # 把边缘进行hough线检测, 提取边缘线参数
    lines = cv2.HoughLines(Bin_img, rho=1, theta=np.pi/180, threshold=60)
    if lines is None: return np.zeros((1,1,2))
    lines = lines.reshape(-1,2)
    # 计算得到边缘直线的各种参数
    pts = line2point(lines) # 计算直线上的两点
    k, b = line2grad(pts)
    # 去计算直线之间的交点
    corner = calc_point(pts, k, b, H, W, RGB_img)
    if corner.shape[0] == 0: return None             # 若没有检测到角点
    if corner.shape[0] < 4: return np.zeros((1,1,2)) # 若检测到角点数量 < 4
    # 交点聚类(4类)，得到的就是身份证的四个顶点
    cor_center,_ = edge_grads(corner)
    # 四个顶点就可以进行定位
    cor_center = cor_center.astype(int)
    if min(calc_four_corner_dist(cor_center))<200: return np.zeros((1,1,2)) # 若角点之间距离太小,则认为是一个点
    # 使角点顺时针有序
    hull = cv2.convexHull(cor_center.reshape((-1, 1, 2))) 
    return hull



# 基于视频流的pipeline
def video_stream(frame, pred_container, last_corner, fps):
    # 读取图像
    RGB_img = frame
    # 固定图像尺寸
    RGB_img, H, W = auto_reshape(RGB_img, 640)
    # 前景背景分割
    Bin_img = Fore_Background_Segmentation(RGB_img)
    # 身份证定位, 返回四个角点坐标
    hull = IDcard_detection(RGB_img, Bin_img, H, W)
    # 透视变换矫正, 并返回分类结果
    target, label, last_corner = transform_recognize(RGB_img, hull, 1, pred_container, last_corner)
    # target, label = transform_recognize(RGB_img, hull, 0, last_corner)
    # 根据先前的预测结果, 启发式的判别当前帧的所属类别
    label, pred_container = calc_video_label(label, hull, pred_container)
    # 可视化
    show_result(RGB_img, target, label, hull, fps)
    return pred_container, last_corner





# 基于图像的pipeline
def static_img(path):
    # 读取图像
    RGB_img = cv2.imread(path)
    # 固定图像尺寸
    RGB_img, H, W = auto_reshape(RGB_img, 640)
    # 前景背景分割
    Bin_img = Fore_Background_Segmentation(RGB_img)
    # 身份证定位, 返回四个角点坐标
    hull = IDcard_detection(RGB_img, Bin_img, H, W)
    # 透视变换矫正, 并返回分类结果
    target, label = transform_recognize(RGB_img, hull, 0)
    # 结果可视化
    show_result(RGB_img, target, label, hull)
    cv2.waitKey(0)    




if __name__ == "__main__":
    # root = '../img/*'
    # images = glob.glob(root)

    # for path in images:
    #    static_img(path)






    capture = cv2.VideoCapture('../video/ID1.mp4')
    # 启发法:
    # 存储每个类别的预测数量，用于启发式选择
    pred_container = [0,0] 
    # 存储上一帧角点的坐标，用于启发式选择
    last_corner = -np.ones((4,2))
    fps = 0
    cnt_fps = 0
    cnt = 0
    while True:
        time_start = time.time()  # 记录开始时间
        ret, frame = capture.read()
        # 视频流结束，退出
        if frame is None: break
        # 视频流检测
        pred_container, last_corner = video_stream(frame, pred_container, last_corner, fps)
        time_end = time.time()    # 记录结束时间
        fps = int(1 / (time_end - time_start))  # 计算的时间差为程序的执行时间，单位为秒/s
        cnt_fps += fps
        cnt +=1
        if cv2.waitKey(1)  == ord(' '):  #判断是哪一个键按下
            break
    cv2.destroyAllWindows()




