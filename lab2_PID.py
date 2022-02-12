import numpy as np
import matplotlib.pyplot as plt


R = 8.31446262
g = 9.80665000
mu = 0.029
one_atm = 101.385


class BlackBox:
    def __init__(self, target, p0=one_atm, t=288.2):  # ~ 15 C
        self._p0 = p0
        self._T0 = t
        self._T = t
        self.h = 0
        self.p = one_atm
        self.target = target
        self.int_d = 0

    def get_outer_pressure(self):
        return self._p0 * np.exp(-mu * g * self.h / (R * self._T))

    def change_target(self, new_target=one_atm):
        self.target = new_target
        self.int_d = 0

    def step(self, inp: int, dh: int):
        dp = (self.get_outer_pressure() - self.p) / self.p
        self.int_d += (self.target - self.p) / 5
        self.p += np.clip(inp - (self.int_d / 15), -10, 10)  # + dp
        self.h += dh - dp


class PID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.pe = 0
        self.i = 0

    def step(self, target, measured, dt):
        error = target - measured
        self.i += error * dt
        d = (error - self.pe) / dt
        self.pe = error
        output = self.kp * error + self.ki * self.i + self.kd * d
        return output


def experiment_1(pid):
    dh = 1
    size = 10000
    heights = np.zeros(size)
    pr = np.zeros(size)
    inputs = np.zeros(size)
    tar = np.zeros(size)
    bb = BlackBox(2 * one_atm)

    for i in range(1, size):
        if i == 2000:
            bb.change_target(3 * one_atm)
        if i == 5000:
            bb.change_target(2 * one_atm)
        inp = pid.step(bb.target, bb.p, dh)
        bb.step(inp, dh)
        heights[i] = bb.h
        pr[i] = bb.p
        inputs[i] = inp
        tar[i] = bb.target

    plt.figure()
    plt.title('Heights')
    plt.plot(heights)

    plt.figure()
    plt.title('Targets')
    plt.plot(tar)

    plt.figure()
    plt.title('Pressure')
    plt.plot(pr)

    plt.figure()
    plt.title('Inputs (pressure)')
    plt.plot(np.clip(inputs, -10, 10))
    plt.show()


if __name__ == '__main__':
    experiment_1(PID(1, .0155, 0))
    # experiment_1(PID(1, 0, 0))
