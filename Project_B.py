# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:11:19 2017

@author: law16
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Ball:
    balls = 0

    def __init__(self, mass=1.0, rad=1.0, pos=[0.0, 0.0], vel=[0.0, 0.0], clr='r',
                 k=0):  # k=0 denotes a ball, k=1 a container
        self.__mass = mass
        self.__rad = rad
        self.__pos = np.array(pos, 'float64')
        self.__vel = np.array(vel, 'float64')
        self.__k = k
        if self.__k == 0:
            Ball.balls += 1
            self.__patch = plt.Circle(self.__pos, self.__rad, fc=clr)  # image of ball imported from lab script
        elif self.__k == 1:
            self.__patch = plt.Circle(self.__pos, self.__rad, fill=False)

    def pos(self):
        return self.__pos

    def vel(self):
        return self.__vel

    def k(self):
        return self.__k

    def move(self, dt):
        pos1 = self.__pos
        self.__pos = pos1 + self.__vel * dt
        self.__patch.center = self.__pos

    def get_patch(self):
        return self.__patch

    def rad(self, other):
        return [self.__rad, other.__rad]

    def time_to_collision(self, other):
        r = other.pos() - self.pos()
        v = other.vel() - self.vel()
        rad1 = self.__rad
        rad2 = other.__rad
        q = -np.dot(r, v) / np.dot(v, v)  # to shorten length of calculations
        dt = -1.0
        # print('k_self: {}, k_other: {}'.format(self.k(), other.k()))
        if self.k() == 0 and other.k() == 0:
            # k = q * q - (np.dot(r, r) - (rad1 + rad2)**2) / np.dot(v, v)
            # k = q * q - (np.dot(r, r) - 4) / np.dot(v, v)
            r_dot = np.dot(r, r)
            v_dot = np.dot(v, v)
            k = q*q - (r_dot - (rad1 + rad2)**2) / v_dot
            if k >= 0:
                sq = np.sqrt(k)
                dt = -q + sq
                # print('dt: {}, q: {}, r_dot: {}, v_dot: {}'.format(dt, q, r_dot, v_dot))
                # dt2 = -q - sq
                # if dt1 >= dt2:
                #     dt = dt1
                # else:
                #     dt = dt2
        elif self.k() == 1 or other.k() == 1:
            k = q * q - (np.dot(r, r) - (rad1 - rad2)**2) / np.dot(v, v)
            if k >= 0:
                sq = np.sqrt(k)
                dt1 = q + sq
                dt2 = q - sq
                if dt1 >= dt2:
                    dt = dt1
                else:
                    dt = dt2
                # print (dt)
        # if dt1 >= 0:
        #     dt = dt1
        # elif dt2 >= 0:
        #     dt = dt2
        return dt

    def collide(self, other):
        # print(id(self), id(other))
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
        if self.__k == 0 and other.__k == 0:
            v1_para = ((u1_para * (m1 - m2) + 2 * m2 * u2_para) / (m1 + m2) * r / R)
            v2_para = ((u2_para * (m2 - m1) + 2 * m1 * u1_para) / (m1 + m2) * r / R)
        elif self.__k == 1 or other.__k == 1:
            v1_para = -u1_para * r / R
            v2_para = -u2_para * r / R
        v1 = v1_perp + v1_para
        v2 = v2_perp + v2_para
        self.__vel = v1
        other.__vel = v2
        return [v1, v2]

    def coll(self, other):
        r = other.pos() - self.pos()
        v = other.vel() - self.vel()
        check = np.dot(r, v)
        # print('o_pos: {}, s_pos: {}, o_vel: {}, s_vel: {} r: {}, v: {}, check: {}'.format(other.pos(), self.pos(), other.vel(), self.vel(), r, v, check))
        if check < 0 and self.k() == 0 and other.k() == 0:
            return True
        elif check > 0 and (self.k() == 1 or other.k() == 1):
            return True
        else:
            return False

    def pos_gen(self):
        _generated_position = np.random.uniform(-9.0, 9.0, 2)

        i = 0
        while np.linalg.norm(_generated_position) > 9:
            _generated_position = np.random.uniform(-9.0, 9.0, 2)
            i += 1
            if i > 100:
                break

        self.__pos = _generated_position

    def pos_gen2(self, data):
        i = 0
        c = 0
        self.pos_gen()
        while i < len(data):
            d = np.linalg.norm(self.__pos - data[i].pos())
            if c == 1000:
                self.__pos[1] = False
                break
            elif d < 2.0:
                self.pos_gen()
                i = 0
            else:
                i += 1
            c += 1


class Balls:
    def __init__(self, n=1, r=1.0, u=1):
        i = 0
        self.__set = []
        while i < n:
            vx, vy = np.random.uniform(-10., 10., 2)
            v = [vx, vy]
            if u == 0:
                r = np.random.uniform(0.2, 1.5)
            self.__ball = Ball(vel=v, rad=r, mass=r*r)
            self.__ball.pos_gen()
            self.__ball.pos_gen2(self.__set)
            if not self.__ball.pos()[1]:
                i += 1
            else:
                self.__set.append(self.__ball)
                i += 1

    def data(self):
        return self.__set


class Orbits:
    def __init__(self):
        self.__container = Ball(rad=10, k=1)
        # self.__ball1 = Ball(pos=[-5, 0], vel=[0.2, 0], rad=1)
        # self.__ball2 = Ball(pos=[5, 0], vel=[-0.2, 0], rad=1, clr='b')
        balls = Balls(20, u=0)
        self.__balls = balls.data()
        self.__balls.append(self.__container)
        # self.__balls.append(self.__ball1)
        # self.__balls.append(self.__ball2)
        self.__text0 = None
        # print(self.__balls[0].pos(), self.__balls[0].vel())

    def init_figure(self):
        big_circ = plt.Circle((0, 0), 10, ec='b', fill=False, ls='solid')
        ax.add_artist(big_circ)
        self.__text0 = ax.text(-9.9, 9, "f={:4d}".format(0, fontsize=12))
        patches = [self.__text0]
        for b in self.__balls:
            pch = b.get_patch()
            ax.add_patch(pch)
            patches.append(pch)
        return patches

    def next_frame(self, framenumber):
        self.__text0.set_text("f={:4d}".format(framenumber))
        patches = [self.__text0]
        c = 1
        for ball in self.__balls:
            i = c
            while i < len(self.__balls):
                dt = ball.time_to_collision(self.__balls[i])
                if 0.0 <= dt <= 0.01 and ball.coll(self.__balls[i]):
                    # print(dt)
                    # if ball.out(self.__balls[i]) < 0 and self.__balls[i].k() == 0:
                    ball.collide(self.__balls[i])
                i += 1
            c += 1
        for b in self.__balls:
            b.move(0.01)
            patches.append(b.get_patch())
        return patches


if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
    ax.axes.set_aspect('equal')

    movie = Orbits()

    anim = animation.FuncAnimation(fig,
                                   movie.next_frame,
                                   init_func=movie.init_figure,
                                   # frames = 1000,
                                   interval=10,
                                   blit=True)

    plt.show()
