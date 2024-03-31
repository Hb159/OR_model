# -*- coding: utf-8 -*-
# @author: HB
# @email: haobin@buaa.edu.cn
# @date: 2023/12/08
import math

from model.model_demo import ModelDemo
import matplotlib.pyplot as plt


class TestModel(ModelDemo):
    def __init__(self, solver, model_params, solver_params, dataclass=None):
        super().__init__(solver, model_params, solver_params, dataclass)
        self.t_actual = 20
        t_delta = 40
        self.T = list(range(t_delta))
        self.t_delta = [self.t_actual / len(self.T)] * (t_delta - 1)
        self.init = [10, 10, 0, 0]  # 0 x 1 y 2 x` 3 y`
        self.end = [18.5, 16, 0, 0]  # 0 x 1 y 2 x` 3 y`
        self.v_max = 3
        self.f_max = 6
        self.circle_center = [(12, 12), (16, 12), (18, 12),
                              (12, 14), (20, 14),
                              (14, 16), (16, 16), (20, 16),
                              (18, 18),
                              ]
        self.circle_radius = [2, 1, 1,
                              1, 1,
                              1, 2, 1,
                              1,
                              ]
        self.circle_buffer = [0.2, 0.2, 0.2,
                              0.2, 0.2,
                              0.2, 0.2, 0.2,
                              0.2,
                              ]
        self.linear_number_state = 8
        self.linear_number_obstacle = 8
        self._set_model()

    def _set_model(self):
        self._set_variables()
        self._set_objectives()
        self._set_constraints()

    def _set_variables(self):
        print('_set_variables')
        self.x = self.tupledict({})
        self.y = self.tupledict({})
        self.x_prime = self.tupledict({})
        self.y_prime = self.tupledict({})
        self.x_prime2 = self.tupledict({})
        self.y_prime2 = self.tupledict({})
        self.f_x = self.tupledict({})
        self.f_y = self.tupledict({})
        self.b = self.tupledict({})
        self.z_x = self.tupledict({})
        self.z_y = self.tupledict({})
        # self.x_prime = self.tupledict({})
        # self.x_prime = self.tupledict({})
        # self.x_prime = self.tupledict({})
        for t in self.T:
            self.x[t] = self.model.addVar(vtype='C', lb=-self.const.INFINITY, name=f'x_{t}')
            self.y[t] = self.model.addVar(vtype='C', lb=-self.const.INFINITY, name=f'y_{t}')
            self.x_prime[t] = self.model.addVar(vtype='C', lb=-self.const.INFINITY, name=f'x_prime_{t}')
            self.y_prime[t] = self.model.addVar(vtype='C', lb=-self.const.INFINITY, name=f'y_prime_{t}')
            self.x_prime2[t] = self.model.addVar(vtype='C', lb=-self.const.INFINITY, name=f'x_prime2_{t}')
            self.y_prime2[t] = self.model.addVar(vtype='C', lb=-self.const.INFINITY, name=f'y_prime2_{t}')
            self.f_x[t] = self.model.addVar(vtype='C', lb=-self.const.INFINITY, name=f'f_x_{t}')
            self.f_y[t] = self.model.addVar(vtype='C', lb=-self.const.INFINITY, name=f'f_y_{t}')

            self.b[t] = self.model.addVar(vtype='B', name=f'b_{t}')
            self.z_x[t] = self.model.addVar(vtype='C', name=f'z_x_{t}')
            self.z_y[t] = self.model.addVar(vtype='C', name=f'z_y_{t}')

    def _set_objectives(self):
        print('_set_objectives')
        theta = 1e-3
        self.model.setObjective(self.quicksum(t * self.b[t] + theta * (self.z_x[t] + self.z_y[t]) for t in self.T),
                                self.const.MINIMIZE)

    def _set_constraints(self):
        print('_set_constraints')
        # 动力学
        for t in self.T[:-1]:
            self.model.addConstr(
                self.x[t + 1] == self.x[t] + self.x_prime[t] * self.t_delta[t] + 0.5 * self.x_prime2[
                    t] * self.t_delta[t] * self.t_delta[t],
                name=f'dynamic1_{t}')  # x_{i+1} = x_{i} + vx_{i}t + 0.5 * ax_{i}t^2
            self.model.addConstr(
                self.y[t + 1] == self.y[t] + self.y_prime[t] * self.t_delta[t] + 0.5 * self.y_prime2[
                    t] * self.t_delta[t] * self.t_delta[t],
                name=f'dynamic2_{t}')  # y_{i+1} = y_{i} + vy_{i}t + 0.5 * ay_{i}t^2
            self.model.addConstr(self.x_prime[t + 1] == self.x_prime[t] + self.x_prime2[t] * self.t_delta[t],
                                 name=f'dynamic3_{t}')  # vx_{i+1} = vx_{i} + ax_{i}t
            self.model.addConstr(self.y_prime[t + 1] == self.y_prime[t] + self.y_prime2[t] * self.t_delta[t],
                                 name=f'dynamic4_{t}')  # vy_{i+1} = vy_{i} + ay_{i}t
            self.model.addConstr(self.x_prime2[t + 1] == self.f_x[t], name=f'dynamic5_{t}')  # ax_{i+1} = fx_{t}/m
            self.model.addConstr(self.y_prime2[t + 1] == self.f_y[t], name=f'dynamic6_{t}')  # ay_{i+1} = fy_{t}/m

        # init
        self.model.addConstr(self.x[self.T[0]] == self.init[0], name=f'init_x')
        self.model.addConstr(self.y[self.T[0]] == self.init[1], name=f'init_y')
        self.model.addConstr(self.x_prime[self.T[0]] == self.init[2], name=f'init_x1')
        self.model.addConstr(self.y_prime[self.T[0]] == self.init[3], name=f'init_y1')
        self.model.addConstr(self.x_prime2[0] == self.f_x[0], name=f'init_x2')  # ax_{0} = fx_{0}/m
        self.model.addConstr(self.y_prime2[0] == self.f_y[0], name=f'init_x2')  # ay_{0} = fy_{0}/m
        # self.model.addConstr(self.f_x[self.T[0]] == self.init[6], name=f'init_f_x')
        # self.model.addConstr(self.f_y[self.T[0]] == self.init[7], name=f'init_f_y')

        # end
        M = 1e3
        for t in self.T:
            self.model.addConstr(self.x[t] - self.end[0] <= M * (1 - self.b[t]), name=f'end_x_less{t}')
            self.model.addConstr(self.x[t] - self.end[0] >= -M * (1 - self.b[t]), name=f'end_x_great{t}')
            self.model.addConstr(self.y[t] - self.end[1] <= M * (1 - self.b[t]), name=f'end_y_less{t}')
            self.model.addConstr(self.y[t] - self.end[1] >= -M * (1 - self.b[t]), name=f'end_y_great{t}')

            self.model.addConstr(self.x_prime[t] - self.end[2] <= M * (1 - self.b[t]), name=f'end_x_prime_less{t}')
            self.model.addConstr(self.x_prime[t] - self.end[2] >= -M * (1 - self.b[t]), name=f'end_x_prime_great{t}')
            self.model.addConstr(self.y_prime[t] - self.end[3] <= M * (1 - self.b[t]), name=f'end_y_prime_less{t}')
            self.model.addConstr(self.y_prime[t] - self.end[3] >= -M * (1 - self.b[t]), name=f'end_y_prime_great{t}')
        self.model.addConstr(self.b.sum('*') == 1, name=f'sum_b')

        # linear obj abs
        for t in self.T:
            self.model.addConstr(self.z_x[t] >= self.f_x[t], name=f'linear_x_positive{t}')
            self.model.addConstr(self.z_x[t] >= -self.f_x[t], name=f'linear_x_negative{t}')
            self.model.addConstr(self.z_y[t] >= self.f_y[t], name=f'linear_y_positive{t}')
            self.model.addConstr(self.z_y[t] >= -self.f_y[t], name=f'linear_y_negative{t}')

        # linear state limit
        v_max = self.v_max
        f_max = self.f_max
        number = self.linear_number_state
        for t in self.T:
            for n in range(1, number + 1):
                self.model.addConstr(self.x_prime[t] * math.sin(2 * math.pi * n / number) + self.y_prime[t] * math.cos(
                    2 * math.pi * n / number) <= v_max, name=f'v_limit_{t}_{n}')
                self.model.addConstr(self.f_x[t] * math.sin(2 * math.pi * n / number) + self.f_y[t] * math.cos(
                    2 * math.pi * n / number) <= f_max, name=f'f_limit_{t}_{n}')

        # linear circle mass
        number = self.linear_number_obstacle
        self.b_c = self.tupledict({})
        for i in range(len(self.circle_center)):
            for t in self.T:
                for n in range(1, number + 1):
                    self.b_c[i, t, n] = self.model.addVar(vtype='B', name=f'b_c_{i}_{t}_{n}')

                    self.model.addConstr(
                        (self.x[t] - self.circle_center[i][0]) * math.sin(2 * math.pi * n / number)
                        + (self.y[t] - self.circle_center[i][1]) * math.cos(2 * math.pi * n / number)
                        >= self.circle_radius[i] + self.circle_buffer[i] - M * self.b_c[i, t, n],
                        name=f'circle_{t}_{i}_{n}')
                self.model.addConstr(self.b_c.sum(i, t, '*') <= number - 1,
                                     name=f'circle_constraint_number_{i}_{number}')

        # seq state cons
        for i in range(len(self.circle_center)):
            for t in self.T[:-1]:
                for n in range(1, number + 1):
                    self.model.addConstr(
                        (self.x[t] - self.circle_center[i][0]) * math.sin(2 * math.pi * n / number)
                        + (self.y[t] - self.circle_center[i][1]) * math.cos(2 * math.pi * n / number)
                        >= self.circle_radius[i] + self.circle_buffer[i] - M * self.b_c[i, t+1, n],
                        name=f'circle_seq_{t}_{i}_{n}')

    def _output_solution(self):
        print('_output_solution')
        self.res = {
            'x': [],
            'y': [],
            'x_prime': [],
            'y_prime': [],
            'x_prime2': [],
            'y_prime2': [],
            'f_x': [],
            'f_y': [],
            'z_x': [],
            'z_y': [],
            'b': []
        }
        for t in self.T:
            self.res['x'].append(self.x[t].x)
            self.res['y'].append(self.y[t].x)
            self.res['x_prime'].append(self.x_prime[t].x)
            self.res['y_prime'].append(self.y_prime[t].x)
            self.res['x_prime2'].append(self.x_prime2[t].x)
            self.res['y_prime2'].append(self.y_prime2[t].x)
            self.res['f_x'].append(self.f_x[t].x)
            self.res['f_y'].append(self.f_y[t].x)
            self.res['z_x'].append(self.z_x[t].x)
            self.res['z_y'].append(self.z_y[t].x)
            self.res['b'].append(self.b[t].x)

        self.t_star = 0
        for t in self.T:
            if self.b[t].x > 0.5:
                self.t_star = t
                break


if __name__ == '__main__':
    solver = 'copt'
    solver_params = {
        "MipLogLevel": 1,
        'TimeLimit': 100,
    }
    model_params = {
        'model_name': 'model_demo',
    }
    demo = TestModel(solver=solver, solver_params=solver_params, model_params=model_params)
    demo.solve_model()

    # 画图
    end = demo.t_star + 1
    fig, ax = plt.subplots(4, 3)

    def draw_scatter_and_line(ax_inner, x_data, y_data):
        ax_inner.scatter(x_data, y_data, color='r', marker='x', s=10)
        ax_inner.plot(x_data, y_data, color='k')
        return ax_inner

    draw_scatter_and_line(ax[0, 0], demo.T[:end], demo.res['x'][:end])
    ax[0, 0].set_title('x')
    draw_scatter_and_line(ax[0, 1], demo.T[:end], demo.res['y'][:end])
    ax[0, 1].set_title('y')
    draw_scatter_and_line(ax[0, 2], demo.res['x'][:end], demo.res['y'][:end])
    ax[0, 2].set_title('trajectory')
    for center, r in zip(demo.circle_center, demo.circle_radius):
        circle = plt.Circle((center[0], center[1]), r, color='y', fill=False)
        ax[0, 2].add_artist(circle)

    draw_scatter_and_line(ax[1, 0], demo.T[:end], demo.res['x_prime'][:end])
    ax[1, 0].set_title('v_x')
    draw_scatter_and_line(ax[1, 1], demo.T[:end], demo.res['y_prime'][:end])
    ax[1, 1].set_title('v_y')
    speed_all = [math.sqrt(i**2+j**2) for i, j in zip(demo.res['x_prime'][:end], demo.res['y_prime'][:end])]
    draw_scatter_and_line(ax[1, 2], demo.T[:end], speed_all)
    ax[1, 2].set_title('speed')

    draw_scatter_and_line(ax[2, 0], demo.T[:end], demo.res['x_prime2'][:end])
    ax[2, 0].set_title('a_x')
    draw_scatter_and_line(ax[2, 1], demo.T[:end], demo.res['y_prime2'][:end])
    ax[2, 1].set_title('a_y')
    acc_all = [math.sqrt(i ** 2 + j ** 2) for i, j in zip(demo.res['x_prime2'][:end], demo.res['y_prime2'][:end])]
    draw_scatter_and_line(ax[2, 2], demo.T[:end], acc_all)
    ax[2, 2].set_title('acc')

    draw_scatter_and_line(ax[3, 0], demo.T[:end], demo.res['f_x'][:end])
    ax[3, 0].set_title('f_x')
    draw_scatter_and_line(ax[3, 1], demo.T[:end], demo.res['f_y'][:end])
    ax[3, 1].set_title('f_y')
    force_all = [math.sqrt(i ** 2 + j ** 2) for i, j in zip(demo.res['f_x'][:end], demo.res['f_y'][:end])]
    draw_scatter_and_line(ax[3, 2], demo.T[:end], force_all)
    ax[3, 2].set_title('force')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1)
    draw_scatter_and_line(ax, demo.res['x'][:end], demo.res['y'][:end])
    ax.set_title('trajectory')
    for center, r, buffer in zip(demo.circle_center, demo.circle_radius, demo.circle_buffer):
        circle = plt.Circle((center[0], center[1]), r, color='y', fill=False)
        circle_buffer = plt.Circle((center[0], center[1]), r + buffer, color='r', fill=False)
        ax.add_artist(circle)
        ax.add_artist(circle_buffer)

    plt.show()

