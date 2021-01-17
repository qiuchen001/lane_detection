import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# 定义一个类来接收每行检测的特征
class Line():
    def __init__(self, n):
        """
        n是移动平均值的窗口大小
        """
        self.n = n
        self.detected = False

        # 多项式系数: x = A*y^2 + B*y + C
        # A，B，C是一个最大长度为n的列表队列
        self.A = []
        self.B = []
        self.C = []
        # 平均值
        self.A_avg = 0.
        self.B_avg = 0.
        self.C_avg = 0.

    def get_fit(self):
        return (self.A_avg, self.B_avg, self.C_avg)

    def add_fit(self, fit_coeffs):
        """
        获取最新的线拟合系数并更新内部平滑系数
        fit_coeffs是二阶多项式系数的三元列表
        """
        # 系数队列已满？
        q_full = len(self.A) >= self.n

        # 附加线拟合系数
        self.A.append(fit_coeffs[0])
        self.B.append(fit_coeffs[1])
        self.C.append(fit_coeffs[2])

        # 如果已满，从索引0弹出
        if q_full:
            _ = self.A.pop(0)
            _ = self.B.pop(0)
            _ = self.C.pop(0)

        # 线系数的简单平均值
        self.A_avg = np.mean(self.A)
        self.B_avg = np.mean(self.B)
        self.C_avg = np.mean(self.C)

        return (self.A_avg, self.B_avg, self.C_avg)
