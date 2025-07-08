import dynamics
import trajectory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
import pandas as pd
random.seed(0)






'''这里的2个系数要改，还有下面的控制参数'''
def C_D(h):
    if -h <= 0:
        return 1.0e-7
    elif 0 < -h <= 0.05:
        return 4.0e-3 * (-h) + 1.0e-7
    else:
        return 2.0e-4

def k(h):
    if -0.05 <= -h <= 0.05:
        return 2.5
    else:
        return 2


class SlidingModeController:
    def __init__(self, lambda_param, k):
        self.lambda_param = lambda_param
        self.k = k
        self.sigma = np.zeros(6)
        self.sigma_history = []

    def controller(self, eta, eta_desired):
        error = eta_desired - eta
        self.sigma = error + self.lambda_param * self.sigma
        self.sigma_history.append(self.sigma.copy())
        control = self.k * np.sign(self.sigma)
        return control

    def thrust_distribution(self, control, height):
        F = (abs(control[2])) / 4 * 2.1 * 0.15
        F = np.clip(F, -5, 5)
        omega = (F / C_D(height)) ** (1 / k(height)) * np.array([1, 1, 1, 1])
        def sign(num): return 1 if num > 0 else -1
        return omega * sign(control[2])

class PID:
    def __init__(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D
        self.integral = np.zeros(6)
        self.previous_error = np.zeros(6)
        self.max_integral = 100.0

    def controller(self, eta, eta_desired):
        error = eta_desired - eta
        self.integral += error
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        derivative = np.round(error - self.previous_error, 10)
        control = self.P * error + self.I * self.integral + self.D * derivative
        self.previous_error = error
        return np.round(control, 10)

    def thrust_distribution(self, control, height):
        F = (abs(control[2])) / 4 * 2.1 * 32
        omega = (F / C_D(height)) ** (1 / k(height)) * np.array([1, 1, 1, 1])
        def sign(num): return 1 if num > 0 else -1
        return omega * sign(control[2])

if __name__ == "__main__":
    huav = dynamics.HUAV()

    time = 0.0
    '''初始位置需要与轨迹定义相同'''



    eta = np.array([0.0, 0.0, 0.5 + 0.05 * np.random.rand(), 0.0, 0.0, 0.0], dtype=np.float64)
    nu = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    total_time = 15
    dt = 0.005
    timesteps = int(total_time / dt)

    time_his = np.zeros(timesteps)
    z_position_history = np.zeros(timesteps)
    z_desired_history = np.zeros(timesteps)
    z_error_history = np.zeros(timesteps)

    smc_controller = SlidingModeController(lambda_param=0.25 , k=150 + 20 - 20)
    pid_water = PID(20, 0.5, 150)
    pid_air = PID(60, 0.5, 3000)
    pid_hybird = PID(80, 0.0, 150)

    alpha = 0.02  # EMA滤波器系数，值越小平滑效果越强

    # 噪声采样频率提升
    noise_dt = dt / 10  # e.g., 10 times the control sampling rate
    noise_steps = int(dt / noise_dt)  # 每次控制更新中的噪声更新次数

    # 初始化EMA滤波器
    eta_ema = eta.copy()
    nu_ema = nu.copy()

    for i in range(timesteps):
        water_level = huav.wave_generate(time)
        eta_desired = trajectory.eta_desired(time)
        hybrid_height=0.2
        if eta_desired[2] >= hybrid_height:
            control = pid_air.controller(eta, eta_desired)
            omega = pid_air.thrust_distribution(control, eta[2])
        elif eta_desired[2] < hybrid_height and eta_desired[2] > -hybrid_height:
            if abs(eta_desired[2] - eta[2]) > 0.1:
                control = smc_controller.controller(eta, eta_desired)
                omega = smc_controller.thrust_distribution(control, eta[2])
            else:
                control = pid_hybird.controller(eta, eta_desired)
                omega = pid_hybird.thrust_distribution(control, eta[2])
        else: 
            control = pid_water.controller(eta, eta_desired)
            omega = pid_water.thrust_distribution(control, eta[2])

        for _ in range(noise_steps):
            # 在更高采样率下加入Wiener过程噪声
            if eta[2] <= 0:
                noise = np.random.normal(0, 0.1, eta.shape)
                wiener_process_noise = np.sqrt(noise_dt) * np.random.normal(0, 0.1, eta.shape)
            else:
                noise = np.random.normal(0, 0.2, eta.shape)
                wiener_process_noise = np.sqrt(noise_dt) * np.random.normal(0, 0.05, eta.shape)

            # 使用更高频率更新噪声
            nu += noise * 0.5 + wiener_process_noise * 0.5

        # 每个控制步长更新一次系统动力学（使用原始的 dt）
        nu_dot, nu, eta, D, F1 = huav.dynamics(eta, nu, water_level, dt, omega)

        # 使用EMA滤波器平滑状态
        eta_ema = alpha * eta + (1 - alpha) * eta_ema
        nu_ema = alpha * nu + (1 - alpha) * nu_ema

        time_his[i] = time
        z_position_history[i] = eta_ema[2]  # 使用平滑后的z位置
        z_desired_history[i] = eta_desired[2]
        z_error_history[i] = eta_desired[2] - eta_ema[2]  # 使用平滑后的z位置误差

        time += dt
        time = np.round(time, 5)

        print("time=%.3f z=%.3f z_desired=%.3f" % (time, z_position_history[i], z_desired_history[i]))


    # 绘制图像
    plt.figure(figsize=(8, 4))
    plt.plot(time_his, z_desired_history, 'r--', label='Desired Trajectory', alpha=0.8)
    plt.plot(time_his, z_position_history, 'b-', label='HUAV (prototype test)', alpha=0.8)
    plt.plot(time_his, z_error_history, 'g--', label='Error', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.legend(loc='lower left', borderaxespad=0)
    plt.grid(True)
    plt.show()

    #保存数据
    data = {'time':time_his, 'z_desired':z_desired_history, 'z_position':z_position_history, 'z_error':z_error_history}
    df = pd.DataFrame(data)
    df.to_csv('z_position_record3.csv', index=False)