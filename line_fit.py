import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform


def line_fit(binary_warped):
    """
    查找并拟合车道线
    """
    # 假设已经创建了一个透视变换后的二值图像binary_warped
    # 取图像下半部分的直方图
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # 创建输出图像并可视化结果
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped)) * 255).astype('uint8')
    # 找出直方图左右两半的峰值
    # 这些将是左车道线与右车道线线的起点
    midpoint = np.int64(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[100:midpoint]) + 100
    rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

    # 选择滑动窗口的数量
    nwindows = 9
    # 设置滑动窗口的高度
    window_height = np.int64(binary_warped.shape[0] / nwindows)
    # 确定图像中所有非零像素点的x和y位置
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 为每个滑动窗口更新当前位置
    leftx_current = leftx_base
    rightx_current = rightx_base
    # 设置滑动窗口的宽度 + / - margin
    margin = 100
    # 设置滑动窗口重新居中时找到的最小像素数
    minpix = 50
    # 创建空列表以接收左车道线和右车道线的像素索引
    left_lane_inds = []
    right_lane_inds = []

    # 一个一个地穿过滑动窗口
    for window in range(nwindows):
        # 确定x和y（以及左右）的滑动窗口边界
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # 在可视化图像上绘制滑动窗口
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # 确定滑动窗口中x和y方向上的非零像素
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
        # 将这些索引附加到列表中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # 如果大于最小像素，则将下一个滑动窗口重新居中于其平均位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))

    # 连接索引数组
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 提取左、右行像素位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 为每个曲线拟合一个二次多项式，使用np.polyfit进行多项式的拟合。
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 返回相关变量的字典
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['out_img'] = out_img
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    return ret


def tune_fit(binary_warped, left_fit, right_fit):
    """
    给定一条先前拟合的线，快速尝试根据之前的线查找该线。
    """
    # 假设现在有一个新的透视变换后的二值图像
    # 来自下一帧视频（也称为透视变换后的二值图像）
    # 现在更容易找到车道线的像素点
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # 再次提取左行和右行像素的位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 如果找不到足够的相关点，则返回全部无（这意味着错误）
    min_inds = 10
    if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
        return None

    # 为每个曲线拟合一个二次多项式
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # 生成用于可视化的x值和y值
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # 返回相关变量的字典
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    return ret


def viz1(binary_warped, ret, save_file=None):
    """
    在透视变换后的二值图像上可视化每个滑动窗口的位置和预测的车道线
    save_file是一个字符串，表示保存图像的位置（如果没有，则只显示）
    """
    # 从返回的字典中选取变量
    left_fit = ret['left_fit']
    right_fit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    out_img = ret['out_img']
    left_lane_inds = ret['left_lane_inds']
    right_lane_inds = ret['right_lane_inds']

    # 生成用于可视化的x值和y值
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
    plt.gcf().clear()


def viz2(binary_warped, ret, save_file=None):
    """
    在透视变换后的二值图像上显示带边缘的预测车道线
    save_file是一个字符串，表示保存图像的位置（如果没有，则只显示）
    """
    # 从返回的字典中选取变量
    left_fit = ret['left_fit']
    right_fit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    left_lane_inds = ret['left_lane_inds']
    right_lane_inds = ret['right_lane_inds']

    # 创建要绘制的图像和显示选择滑动窗口的图像
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped)) * 255).astype('uint8')
    window_img = np.zeros_like(out_img)
    # 以左右线像素为单位的颜色
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # 生成用于可视化的x值和y值
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # 生成多边形以演示搜索窗口区域
    # 并将x和y点重新转换为cv2.fillPoly()的可用格式
    margin = 100  # 注释：与with *_fit()保持同步
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # 将车道绘制到透视变换后的空白图像上
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
    plt.gcf().clear()


def calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
    """
    计算曲率半径（米）
    """
    y_eval = 719  # 720P的视频/图像，所以最后（屏幕上最低的）y的索引是719。

    # 定义从像素空间到米的x和y转换
    ym_per_pix = 30 / 720  # y维度的每像素米数
    xm_per_pix = 3.7 / 720  # x维度的每像素米数

    # 提取左、右行像素的位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 将新的多项式拟合到世界空间中的x，y
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # 计算新的曲率半径
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # 现在我们的曲率半径是米

    return left_curverad, right_curverad


def calc_vehicle_offset(undist, left_fit, right_fit):
    """
    计算车辆与车道中心的偏移，单位为米
    """
    # 计算车辆中心偏移（以像素为单位）
    bottom_y = undist.shape[0] - 1
    bottom_x_left = left_fit[0] * (bottom_y ** 2) + left_fit[1] * bottom_y + left_fit[2]
    bottom_x_right = right_fit[0] * (bottom_y ** 2) + right_fit[1] * bottom_y + right_fit[2]
    vehicle_offset = undist.shape[1] / 2 - (bottom_x_left + bottom_x_right) / 2

    # 将像素偏移转换为米
    xm_per_pix = 3.7 / 700  # x维度每像素米数
    vehicle_offset *= xm_per_pix

    return vehicle_offset


def final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset):
    """
    最终车道线预测可视化并叠加在原始图像上
    """
    # 为绘图生成x值和y值
    ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # 创建图像以绘制线条
    # warp_zero = np.zeros_like(warped).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # 注释：硬编码图像尺寸

    # 将x和y点重新转换为cv2.fillPoly()的格式
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))  # 将两个数组按水平方向组合起来

    # 将车道绘制到透视变换后的空白图像上
    # cv2.fillPoly()函数用来填充任意形状的图形
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))

    # 利用逆透视矩阵（MIV）将透视变换后的图像返回到原始图像空间
    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    # 将结果与原始图像合并
    # cv2.addWeighted()函数将两张相同shape的图片按权重进行融合
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # 注释车道曲率值和车辆中心偏移
    avg_curve = (left_curve + right_curve) / 2
    label_str = 'Radius of curvature: %.1f m' % avg_curve
    # cv2.putText()函数向图像上添加文本内容
    result = cv2.putText(result, label_str, (30, 40), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)

    label_str = 'Vehicle offset from lane center: %.1f m' % vehicle_offset
    result = cv2.putText(result, label_str, (30, 70), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return result
