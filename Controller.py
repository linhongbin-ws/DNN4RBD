from numpy import pi
import torch
import numpy as np

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

class Dynamic_Controller():
    def __init__(self, model, input_scaler, output_scaler):
        self.model = model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
    def forward(self, s):
        """
        mapping state to action
        :param state s = [q1, q2, qd1, qd2, qdd1, qdd2]
        :return: a = [a1, a2]
        """
        input_mat = np.array([s])
        feature_norm = torch.from_numpy(self.input_scaler.transform(input_mat)).to('cpu').float()
        target_norm_hat = self.model(feature_norm)
        target_hat_mat = self.output_scaler.inverse_transform(target_norm_hat.detach().numpy())
        return [target_hat_mat[0,i] for i in range(target_hat_mat.shape[1])]

class PD_Dynamic_Controller():
    def __init__(self, basic_controller, dynamic_controller):
        self.basic_controller = basic_controller
        self.dynamic_controller = dynamic_controller
    def forward(self, s, sDes):
        """
        mapping state to action
        :param state s = [q1, q2, qd1, qd2, qdd1, qdd2]
               sDes = [q1Des, q2Des, qd1Des, qd2Des, qdd1Des, qdd2Des]
        :return: a = [a1, a2]
        """
        [tau1, tau2] = self.basic_controller.forward(s, sDes)
        [tau1_res, tau2_res] = self.dynamic_controller.forward(sDes)
        return [tau1+tau1_res, tau2+tau2_res]
