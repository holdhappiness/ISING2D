import random
from math import exp
import numpy as np
import matplotlib.pyplot as plt

class Ising:
    #创建新的二维伊辛模型系统，并设定系统大小、温度与相互作用强度
    #随机化各格点初始自旋
    def __init__(self, size, j, beta):
        self.size = size
        self.j = j
        self.beta = beta
        self.energy = 0
        self.mag_mom = 0
        self.sites = np.ones((size, size))

        #初始化格点自旋
        for i in range(size):
            for j in range(size):
                spin = random.choice([-1, 1])
                self.sites[i, j] = spin
                self.mag_mom += spin

        for i in range(size):
            for j in range(size):
                self.energy += self.sites[i,j] * self.sites[i, (j+1)%size] + self.sites[i,j] * self.sites[(i+1)%size,j]

        self.energy *= -self.j

    #获取系统总能量
    def get_energy(self):
        return self.energy

    #获取系统总磁化强度
    def get_magnetic_momentum(self):
        return self.mag_mom / self.size ** 2

    #翻转(m, n)处格点的自旋，并重新计算总能量与总磁化强度
    def flip(self, m, n):
        m = m % self.size
        n = n % self.size

        self.sites[m, n] *= -1
        self.mag_mom += 2 * self.sites[m, n]
        self.energy += -2 * self.j * (self.sites[m, n] *
                                      (self.sites[m, (n+1)%self.size] + self.sites[m, (n+self.size-1)%self.size]
                                      +self.sites[(m+1)%self.size, n] + self.sites[(m+self.size-1)%self.size, n]))

    #从(m, n)处起始，通过深度优先搜索算法进行一次 Wolff集团更新
    def single_wolff(self, m, n):
        m = m % self.size
        n = n % self.size
        spin_0 = self.sites[m, n]

        p = 1 - exp(-2 * self.beta * self.j)

        pool = []
        self.flip(m, n)
        pool.append((m, (n+1)%self.size))
        pool.append((m, (n+self.size-1) % self.size))
        pool.append(((m+1)%self.size, n))
        pool.append(((m+self.size-1) % self.size, n))

        while pool:
            i, j = pool.pop()
            if self.sites[i, j] * spin_0 == 1:
                rand = random.random()
                if rand < p:
                    self.flip(i, j)

                    pool.append((i, (j+1)%self.size))
                    pool.append((i, (j + self.size - 1) % self.size))
                    pool.append(((i + 1) % self.size, j))
                    pool.append(((i + self.size - 1) % self.size, j))

    #随机选取起始位点，连续进行 steps次 Wolff集团更新
    def series_wolff(self, steps):
        for step in range(steps):
            m = random.randint(0, self.size-1)
            n = random.randint(0, self.size-1)
            self.single_wolff(m, n)

#测量二维伊辛模型中 Binder Cumulant随温度的变化情况，绘制曲线图并保存至指定路径
def draw_binder_cumulant(size, j, k, beta_span, simulate_times, pathname):
    sizes = [2*size, size, size//2]
    beta_start, beta_end = beta_span
    betas = np.linspace(beta_start, beta_end, 100)
    t_range = np.array([1/(k*beta) for beta in betas])
    u_result = np.zeros((3, 100))

    #进行simulate_times次的Monte Carlo模拟
    for i, s in enumerate(sizes):
        for k, beta in enumerate(betas):
            m2_sig = 0
            m4_sig = 0

            for _ in range(simulate_times):
                ising = Ising(s, j, beta)
                ising.series_wolff(500)
                m = abs(ising.get_magnetic_momentum())

                m2_sig += m**2
                m4_sig += m**4

            m2 = m2_sig / simulate_times
            m4 = m4_sig / simulate_times
            u_result[i, k] = 1.5 * (1 - m4 / (3 * m2**2))

    #绘制曲线图
    fig, ax = plt.subplots()
    ax.set_title('Binder Cumulant')
    ax.set_xlabel('t')
    ax.set_ylabel('U')
    for i in range(3):
        plt.plot(t_range, u_result[i, :], label='size = {}'.format(sizes[i]))
    plt.legend()
    plt.savefig(pathname)
    plt.close()

#测量二维伊辛模型中无量纲量 ML^1/8与 tL的变化关系，绘制曲线图并保存至指定路径
def draw_universal_curve(j, beta_span, simulate_times, pathname):
    tc = 2.269185
    sizes = [10, 20, 30, 40, 50]
    beta_start, beta_end = beta_span
    betas = np.linspace(beta_start, beta_end, 100)

    x_result = np.zeros(500)
    y_result = np.zeros(500)

    #进行simulate_times次的Monte Carlo模拟
    for i, s in enumerate(sizes):
        for k, beta in enumerate(betas):
            m_sig = 0

            for _ in range(simulate_times):
                ising = Ising(s, j, beta)
                ising.series_wolff(500)

                m_sig += abs(ising.get_magnetic_momentum())

            x_result[i * 100 + k] = s * (1 / beta - tc) / tc
            y_result[i * 100 + k] = (m_sig/simulate_times) * s ** 0.125

    #绘制曲线图
    fig, ax = plt.subplots()
    ax.set_title('Universal Curve')
    ax.set_xlabel('tL')
    ax.set_ylabel('ML^1/8')
    plt.scatter(x_result, y_result)
    plt.savefig(pathname)
    plt.close()



if __name__ == '__main__':
    l = 50
    j, k = 1, 1
    beta_span = [0.34, 0.7]
    pathname_bc = r'.\images\test_bc.png'
    pathname_mc = r'.\images\test_mc.png'
    draw_binder_cumulant(l, j, k, beta_span, 100, pathname_bc)
    draw_universal_curve(j, beta_span, 100, pathname_mc)









