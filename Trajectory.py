import numpy as np
class CosTraj():
    def __init__(self):
        self.A = np.pi
        self.w = np.array([1.0,1.0])
        self.b = np.array([0.,0.])
        self.dof = 2
    def forward(self, t):
        q = np.cos(self.w * t + self.b)*self.A
        qd = -self.w * np.sin(self.w * t + self.b)*self.A
        qdd = -self.w * self.w * np.cos(self.w * t + self.b) *self.A
        return q, qd, qdd