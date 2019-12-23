from numpy import pi
class PD_Controller():
    def __init__(self):
        self.kp = [100., 100.]
        self.kd = [10, 10]
        self.dof = 2
    def forward(self, s, sDes):
        """
        mapping state to action
        :param state s = [q1, q2, qd1, qd2]
               sDes = [q1Des, q2Des, qd1Des, qd2Des]
        :return: a = [a1, a2]
        """
        qe1 = sDes[0] - s[0]
        if qe1>pi:
            qe1 -= 2*pi
        elif qe1<-pi:
            qe1 += 2 * pi
        qe2 = sDes[1] - s[1]
        if qe2>pi:
            qe2 -= 2*pi
        elif qe2<-pi:
            qe2 += 2 * pi

        a1 = qe1 * self.kp[0] + (sDes[2]- s[2]) * self.kd[0]
        a2 = qe2 * self.kp[1] + (sDes[3]- s[3]) * self.kd[1]
        return [a1, a2]