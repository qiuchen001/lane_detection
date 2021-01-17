import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh


def perspective_transform(img):
    """
    执行透视变换：将倾斜视角拍摄到的道路图像转换成鸟瞰图，即将摄像机的视角转换到和道路平行。
    """
    img_size = (img.shape[1], img.shape[0])
    # 手动提取用于执行透视变换的顶点
    src = np.float32(
        [[200, 720],
         [1100, 720],
         [595, 450],
         [685, 450]])
    dst = np.float32(
        [[300, 720],
         [980, 720],
         [300, 0],
         [980, 0]])
    # src源图像中待测矩形的四点坐标
    # dst目标图像中矩形的四点坐标
    # cv2.getPerspectiveTransform() 计算透视变换矩阵
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    # cv2.warpPerspective()进行透视变换
    # 参数：输入图像、输出图像、目标图像大小、cv2.INTER_LINEAR插值方法
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # 调试

    return warped, unwarped, m, m_inv


if __name__ == '__main__':
    img_file = 'test_images/test6.jpg'

    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']

    img = mpimg.imread(img_file)
    img = cv2.undistort(img, mtx, dist, None, mtx)

    img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)

    warped, unwarped, m, m_inv = perspective_transform(img)

    plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
    plt.show()

    plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
    plt.show()
