from os.path import join
import string

import pinocchio
from pinocchio.robot_wrapper import RobotWrapper

import crocoddyl

import eagle_mpc
from eagle_mpc.utils.path import EAGLE_MPC_YAML_DIR

import numpy as np


def check_error(text: str, val: float, threshold: float = 1e-10):
    if val > threshold:
        print(text, 'NOT Passed')
        return False

    print(text, 'Passed')
    return True


class ModelTests:

    def __init__(self):
        robot = RobotWrapper.BuildFromURDF(
            join(EAGLE_MPC_YAML_DIR, 'iris/description/iris.urdf'),
            package_dirs=[join(EAGLE_MPC_YAML_DIR, '../..')],
            root_joint=pinocchio.JointModelFreeFlyer())
        self.r_model = robot.model

        self.mc_params = eagle_mpc.MultiCopterBaseParams()
        self.mc_params.autoSetup(
            join(EAGLE_MPC_YAML_DIR, 'iris', 'platform', 'iris.yaml'))

        self.state = crocoddyl.StateMultibodyActuated(self.r_model,
                                                      self.mc_params.n_rotors)
        self.act_model = crocoddyl.ActuationModelMultiCopterBaseFos(
            self.state, self.mc_params.tau_f, self.mc_params.cf)
        self.act_data = self.act_model.createData()

    def run_tests(self):
        self.actuation_model_test()
        self.state_model_test()

    def actuation_model_test(self):
        u = 600 * np.random.rand(self.mc_params.n_rotors)
        x = self.state.rand()
        x[:3] = np.array([0, 0, 0])

        self.act_model.calc(self.act_data, x, u)
        self.act_model.calcDiff(self.act_data, x, u)

        thrusts = x[-self.mc_params.n_rotors:]**2 * self.mc_params.cf
        tau = self.mc_params.tau_f @ thrusts

        error = np.linalg.norm(tau - self.act_data.tau)

        if not check_error('Actuation: ', error):
            return

        dtau_df = self.mc_params.tau_f
        df_dw = np.diag(2 * self.mc_params.cf * x[-self.mc_params.n_rotors:])

        dtau_dw = np.zeros([6, self.state.ndx])
        dtau_dw[-6:, -self.mc_params.n_rotors:] = dtau_df @ df_dw
        diff_error = np.linalg.norm(self.act_data.dtau_dx - dtau_dw)

        if not check_error('Actuation Diff: ', diff_error):
            return False

        return True

    def state_model_test(self):
        print()

    def dam_ffd_act(self):
        print()


def main():
    tests = ModelTests()
    tests.run_tests()


if __name__ == '__main__':
    main()
