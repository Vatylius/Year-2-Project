# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:11:19 2017
@author: law16
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from time import sleep
from tqdm import trange

class Ball:
    balls = 0

    def __init__(self, mass=1.0, rad=1.0, pos=[0.0, 0.0], vel=[0.0, 0.0], clr='r',
                 k=0):  # k=0 denotes a ball, k=1 a container
        self.__mass = mass
        self.__rad = rad
        self.__pos = np.array(pos, 'float64')
        self.__vel = np.array(vel, 'float64')
        self.__k = k

    def pos(self):
        return self.__pos

    def vel(self):
        return self.__vel

    def k(self):
        return self.__k

    def move(self, dt):
        pos1 = self.__pos
        self.__pos = pos1 + self.__vel * dt

    def rad(self):
        return self.__rad

    def momentum(self):
        v = np.linalg.norm(self.__vel)
        m = self.__mass
        p = m * v
        return p

    def kin_energy(self):
        v = np.linalg.norm(self.__vel)
        m = self.__mass
        ke = m * v * v / 2
        return ke

    def time_to_collision(self, other):
        r = self.pos() - other.pos()
        v = other.vel() - self.vel()
        rad1 = self.__rad
        rad2 = other.__rad
        q = np.dot(r, v) / np.dot(v, v)  # to shorten length of calculations
        dt = -1.0
        # print('k_self: {}, k_other: {}'.format(self.k(), other.k()))
        if self.k() == 0 and other.k() == 0:
            # k = q * q - (np.dot(r, r) - (rad1 + rad2)**2) / np.dot(v, v)
            # k = q * q - (np.dot(r, r) - 4) / np.dot(v, v)
            r_dot = np.dot(r, r)
            v_dot = np.dot(v, v)
            k = q * q - (r_dot - (rad1 + rad2) ** 2) / v_dot
            if k >= 0:
                sq = np.sqrt(k)
                dt1 = -q + sq
                # print('dt: {}, q: {}, r_dot: {}, v_dot: {}'.format(dt, q, r_dot, v_dot))
                dt2 = -q - sq
                if dt1 >= dt2:
                    dt = -dt1
                else:
                    dt = -dt2
        elif self.k() == 1 or other.k() == 1:
            k = q * q - (np.dot(r, r) - (rad1 - rad2) ** 2) / np.dot(v, v)
            if k >= 0:
                sq = np.sqrt(k)
                dt1 = q + sq
                dt2 = q - sq
                if dt1 >= dt2:
                    dt = dt1
                else:
                    dt = dt2
        # if dt1 >= 0:
        #     dt = dt1
        # elif dt2 >= 0:
        #     dt = dt2
        return dt

    def collide(self, other):
        m1 = self.__mass
        m2 = other.__mass
        u1 = self.__vel
        u2 = other.__vel
        r = self.__pos - other.__pos
        R = np.linalg.norm(r)
        u1_para = np.dot(u1, r) / R
        u1_perp = u1 - u1_para * r / R
        u2_para = np.dot(u2, r) / R
        u2_perp = u2 - u2_para * r / R
        v1_perp = u1_perp
        v2_perp = u2_perp
        dp = 0  # Change in momentum
        if self.__k == 0 and other.__k == 0:
            v1_para = ((u1_para * (m1 - m2) + 2 * m2 * u2_para) / (m1 + m2) * r / R)
            v2_para = ((u2_para * (m2 - m1) + 2 * m1 * u1_para) / (m1 + m2) * r / R)
        elif self.__k == 1 or other.__k == 1:
            v1_para = -u1_para * r / R
            v2_para = -u2_para * r / R
            if self.__k == 1:
                dv = u2_para - np.linalg.norm(v2_para)
                dp = m2 * dv
            else:
                dv = np.linalg.norm(u1_para * r / R - v1_para)
                dp = m1 * dv
        v1 = v1_perp + v1_para
        v2 = v2_perp + v2_para
        self.__vel = v1
        other.__vel = v2
        return dp

    def coll(self, other):
        r = other.pos() - self.pos()
        v = other.vel() - self.vel()
        check = np.dot(r, v)
        if check < 0 and self.k() == 0 and other.k() == 0:
            return True
        elif check > 0 and (self.k() == 1 or other.k() == 1):
            return True
        else:
            return False

    def pos_gen(self):
        r = 10. - self.rad()
        loc = np.random.uniform(-r, r, 2)

        i = 0
        while np.linalg.norm(loc) > r:
            loc = np.random.uniform(-r, r, 2)
            i += 1
            if i > 100:
                break

        self.__pos = loc

    def pos_overlap_check(self, data):
        i = 0
        c = 0
        self.pos_gen()
        while i < len(data):
            d = np.linalg.norm(self.__pos - data[i].pos())
            if c == 1000:
                self.__pos[1] = False
                break
            elif d < self.rad() + data[i].rad():
                self.pos_gen()
                i = 0
            else:
                i += 1
            c += 1

    def distance_to_collision(self, other):
        r = self.pos() - other.pos()
        d = np.sqrt(np.dot(r, r))
        if d < self.rad() + other.rad() and self.k() == 0 and other.k() == 0:
            return True
        elif d > abs(self.rad() - other.rad()) and (self.k() == 1 or other.k() == 1):
            return True
        else:
            return False


class Balls:
    def __init__(self, n=1, r=1.0, u=1):
        i = 0
        self.__set = []
        while i < n:
            vx, vy = np.random.uniform(-3000., 3000., 2)
            v = [vx, vy]
            if u == 0:
                r = np.random.uniform(0.05, 1.5)
            self.__ball = Ball(vel=v, rad=r, mass=1)
            self.__ball.pos_gen()
            self.__ball.pos_overlap_check(self.__set)
            if not self.__ball.pos()[1]:
                i += 1
            else:
                self.__set.append(self.__ball)
                i += 1

    def data(self):
        return self.__set

    def p(self):
        p = 0
        for ball in self.__set:
            p += ball.momentum()
        return p

    def energy(self):
        ke = 0
        for ball in self.__set:
            ke += ball.kin_energy()
        return ke


class Orbits:
    momentum = 0

    def __init__(self):
        self.__container = Ball(rad=10, k=1)
        # self.__ball1 = Ball(pos=[-2, 0], vel=[100, 0], rad=1)
        # self.__ball2 = Ball(pos=[2, 0], vel=[-80, 0], rad=1, clr='b')
        self.__balls = Balls(100, r=0.05)
        self.__data = self.__balls.data()
        # self.__data = []
        self.__data.append(self.__container)
        # self.__data.append(self.__ball1)
        # self.__data.append(self.__ball2)
        self.__n = 0
        # self.__p = 0
        # self.__ke = 0
        self.__text0 = None
        self.__frame_interval = 1  # milliseconds
        self.__P = 0

    def frame_interval(self):
        return self.__frame_interval

    def next_frame(self, framenumber):
        c = 1
        for ball in self.__data:
            i = c
            while i < len(self.__data):
                dt = ball.time_to_collision(self.__data[i])
                if 0 < dt <= 0.0001 and ball.coll(self.__data[i]):
                    dp = ball.collide(self.__data[i])
                    Orbits.momentum += dp
                i += 1
            c += 1
        for b in self.__data:
            b.move(0.0001)
        if framenumber % 1000 == 0 and framenumber != 0:
            dt = (self.__frame_interval / 1000) * 1000
            r = self.__container.rad()
            self.__P = Orbits.momentum / (dt * np.pi * 2 * r)  # Pressure given in N/m
            P = self.__P
            self.__P = 0
            Orbits.momentum = 0
            return P


simulation = Orbits()
iterations = 5001
P = []

sleep(0.1)

for i in trange(iterations, ncols=50):
    dP = simulation.next_frame(i)
    if i % 1000 == 0 and i != 0:
        P.append('{:6.3f}'.format(dP))

sleep(0.1)

Pressure = 'Pressure: \n'
for i in P:
    Pressure += str(i) + '\n'

print(Pressure)
