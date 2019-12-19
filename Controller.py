
class PD_Controller():
    def __init__(self):
        self.kp = [100., 100.]
        self.kd = [10, 10]
        self.dof = 2
    def forward(self, s, qDes):
        """
        mapping state to action
        :param state s = [q1, q2, qd1, qd2]
               qDes = [qDes1, qDes2]
        :return: a = [a1, a2]
        """
        a1 = (qDes[0] - s[0]) * self.kp[0] - s[2] * self.kd[0]
        a2 = (qDes[1] - s[1]) * self.kp[1] - s[3] * self.kd[1]
        return [a1, a2]