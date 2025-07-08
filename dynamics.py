import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(0)

class HUAV:
    def __init__(self):
        # 惯性矩阵
        self.m = 2.1
        self.Ixx, self.Iyy, self.Izz = 0.1, 0.1, 0.1
        self.M = np.array([[self.m, 0, 0, 0, 0, 0],
                           [0, self.m, 0, 0, 0, 0],
                           [0, 0, self.m, 0, 0, 0],
                           [0, 0, 0, self.Ixx, 0, 0],
                           [0, 0, 0, 0, self.Iyy, 0],
                           [0, 0, 0, 0, 0, self.Izz]])
        # 水阻尼系数
        self.CDw = np.array([20, 20, 20, 10, 10, 10])
        # self.CDw = np.array([80, 80, 80, 10, 10, 10])
        # 空气阻尼系数
        self.CDa = self.CDw / 800
        # 接触面积
        self.A = 0.05
        # self.A = 0.1
        # 水密度
        self.phow = 1000
        # 空气密度
        self.phoa = 1.25
        # 重力加速度
        self.g = 9.81
        # 对角桨距
        self.l = 0.1
        # 螺旋桨扭矩系数
        self.k = 0.1
        # 跨介质特征高度
        # self.H = 0.1
        self.H = 0.05
        # 螺旋桨推力系数
        self.k1, self.k2, self.k3, self.k4, self.k5 = 8.470e-07, 0.0655, -67.806, 3.476e-19, 2.644
        #体积
        self.V = 0.02
        #扰动噪声强度
        self.noise_intence_water = 10
        self.noise_intence_air = self.noise_intence_water / 8
        self.drift = 0.01  # 维纳过程的漂移
        self.diffusion = 0.1  # 维纳过程的扩散系数
        #海流影响强度
        self.current_intence = 1
        #干扰项开关
        self.switch_noise, self.switch_current = 0, 0
        #海浪开关
        self.switch_wave = 0
        #海浪波形
        self.wave_intence = 0.1
        self.wave_omega = 0.5 * np.pi
        # self.wave_intence = 0.4
        # self.wave_omega = 10 * np.pi
        self.wave_phi = 0
        #海平面高度
        self.water_level = 0


    def damping_matrix(self, eta, nu, water_level):
        h = -water_level + eta[2]
        Dw = -nu * self.CDw * self.A
        Da = -nu * self.CDa * self.A
        # 跨介质条件判断
        if h < -self.H / 2:
            D = Dw
        elif h > self.H / 2:
            D = Da
        else:
            alpha = h / self.H
            D = (0.5 - alpha) * Dw + (0.5 + alpha) * Da
        return D

    def return_matrix(self, eta, water_level):
        h = -water_level + eta[2]
        m = self.M[0,0]
        g = self.g
        # 跨介质条件判断
        if h < -self.H / 2:
            B = self.phow * self.V * g
        elif h > self.H / 2:
            B = self.phoa * self.V * g
        else:
            alpha = h / self.H
            B = (0.5 - alpha) * self.phow * self.V * g+ (0.5 + alpha) * self.phoa * self.V * g
        return np.array([0, 0, B - m * g, 0, 0, 0])

    def disturbance_matrix(self, eta, water_level, switch_noise, switch_current, current_direction, dt):
        noise = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #开关条件判断
        if switch_noise:
            h = -water_level + eta[2]
            # 跨介质条件判断
            if h < -self.H / 2:
                noise_intence = self.noise_intence_water
            elif h > self.H / 2:
                noise_intence = self.noise_intence_air
            else:
                alpha = h / self.H
                noise_intence = (0.5 - alpha) * self.noise_intence_water + (0.5 + alpha) * self.noise_intence_air
            #随机噪声
            noise += (noise_intence * np.random.rand(6)).flatten()
            noise += np.random.normal(self.drift, np.sqrt(self.diffusion * dt), size=6)
            #noise += np.cumsum(np.random.normal(self.drift, np.sqrt(self.diffusion * dt)))
        if switch_current:
            if eta[2] <= water_level:
                #归一化
                normalized_direction = current_direction / np.linalg.norm(current_direction)
                #指定方向的海流影响
                current += self.current_intence * normalized_direction
        return noise + current

    def main_force_matrix(self, eta, water_level, omega):
        h = water_level - eta[2]
        '''
        # 定义函数M(h, \omega)
        def M(h, omega):
            if h < -self.H / 2:
                return (self.k1 / (1 + np.exp(-self.k2 * (self.k3 + self.H / 2))) + self.k4) * omega ** self.k5
            elif h > self.H / 2:
                return (self.k1 / (1 + np.exp(-self.k2 * (self.k3 - self.H / 2))) + self.k4) * omega ** self.k5
            else:
                return (self.k1 / (1 + np.exp(-self.k2 * (self.k3 - h))) + self.k4) * omega ** self.k5
        '''
        # 计算单螺旋桨推力
        def C_D(h):
            if h < 0:
                return 1.0e-7
            elif h > self.H/2:
                return 1.1e-4
            else:
                return (2e-4 - 2e-7)/self.H * h + 1e-7
            
        def k(h):
            if (h > -self.H / 2)&(h < 0):
                return 2.3
            else:
                return 2
            
        def M(h, omega):
            return np.clip(C_D(h) * omega ** k(h), -self.m * 9.81* 20, self.m * 9.81* 20)
        
        def sign(num): return 1 if num > 0 else -1

        F1 = M(h, abs(omega[0])) * sign(omega[0])
        F2 = M(h, abs(omega[1])) * sign(omega[1])
        F3 = M(h, abs(omega[2])) * sign(omega[2])
        F4 = M(h, abs(omega[3])) * sign(omega[3])

        return np.array([0, 0, F1+F2+F3+F4, self.l*(F4-F2), self.l*(F3-F1), self.k*(-F1+F2-F3+F4)]), F1

    def dynamics(self, eta, nu, water_level, dt, omega):
        # 阻尼向量
        D = self.damping_matrix(eta, nu, water_level)
        # 恢复力向量
        R = self.return_matrix(eta, water_level)
        # 环境干扰向量
        current_direction = np.array([1, 0, 0, 0, 0, 0])
        E = self.disturbance_matrix(eta, water_level, self.switch_noise, self.switch_current, current_direction, dt)
        # 主动力向量
        F, F1 = self.main_force_matrix(eta, water_level, omega)
        # 合力（矩）计算
        F_total = D + R + E + F
        # （角）加速度计算
        nu_dot = np.linalg.inv(self.M) @ F_total

        # 积分
        nu += nu_dot * dt
        eta += nu * dt

        # 将eta中的后三个姿态角坐标限制在-pi到pi之间
        eta[3:] = np.mod(eta[3:] + np.pi, 2 * np.pi) - np.pi
        
        #系统噪声

        return nu_dot, nu, eta, D, F1

    def wave_generate(self, time):
        if self.switch_wave:
            wave_level = self.water_level + self.wave_intence * np.sin(self.wave_omega * time + self.wave_phi)
            return wave_level
        return self.water_level
    
    def l_and_k(self):
        return self.l, self.k

if __name__ == "__main__":
    huav = HUAV()

    time = 0.0
    eta = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    nu = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    nu_dot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    total_time = 1.1
    dt = 0.001
    timesteps = int(total_time / dt)

    # 用于记录
    time_history = np.zeros(timesteps)
    D_z_history = np.zeros(timesteps)
    z_position_history = np.zeros(timesteps)
    z_velocity_history = np.zeros(timesteps)
    z_acceleration_history = np.zeros(timesteps)
    F_history = np.zeros(timesteps)
    water_level_history = np.zeros(timesteps)

    # 运行模拟
    for i in range(timesteps):
        #海浪更新
        water_level = huav.wave_generate(time)
        #动力学更新
        if time >= 0.5:
            nu_dot, nu, eta, D, F1= huav.dynamics(eta, nu, water_level, dt, np.array([100, 100, -100, -100]))
        else:
            nu_dot, nu, eta, D, F1= huav.dynamics(eta, nu, water_level, dt, np.array([2000, 2000, 2000, 2000]))

        # 高斯扰动
        noise = np.random.normal(0, 0.2, eta.shape)
        # 维纳过程噪声
        wiener_process_noise = np.sqrt(dt) * np.random.normal(0, 0.005, eta.shape)
        # 更新带有噪声的状态
        #eta += noise + wiener_process_noise
        #nu += noise + wiener_process_noise
        #记录所需数据
        time_history[i] = time
        water_level_history[i] = water_level  # 记录水面高度
        D_z_history[i] = D[2]  # z方向阻尼
        z_position_history[i] = eta[3]  # z方向位置
        z_velocity_history[i] = nu[3]  # z方向速度
        z_acceleration_history[i] = nu_dot[3]  # z方向加速度
        F_history[i] = F1
        #时间步长加1
        time += dt

    # 创建一个图表
    plt.figure(figsize=(12, 10))

    # 绘制水面高度随时间变化的曲线
    plt.subplot(3, 2, 1)
    plt.plot(time_history, water_level_history, label='Water Level')
    plt.title('Water Level Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Water Level (m)')
    plt.grid(True)
    plt.legend()

    # 绘制z方向阻尼随时间变化的曲线
    plt.subplot(3, 2, 2)
    #plt.plot(time_history, D_z_history, label='Z Direction Damping', color='orange')
    plt.title('Z Direction Damping Over Time')
    plt.plot(time_history, F_history, label='F', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Damping (Ns/m)')
    plt.grid(True)
    plt.legend()

    # 绘制z方向位置随时间变化的曲线
    plt.subplot(3, 2, 3)
    plt.plot(time_history, z_position_history, label='Z Position', color='green')
    plt.title('Z Position Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.grid(True)
    plt.legend()

    # 绘制z方向速度随时间变化的曲线
    plt.subplot(3, 2, 4)
    plt.plot(time_history, z_velocity_history, label='Z Velocity', color='red')
    plt.title('Z Velocity Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()

    # 绘制z方向加速度随时间变化的曲线
    plt.subplot(3, 2, 5)
    plt.plot(time_history, z_acceleration_history, label='Z Acceleration', color='purple')
    plt.title('Z Acceleration Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.grid(True)
    plt.legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图表
    plt.show()