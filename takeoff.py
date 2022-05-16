from os.path import join

import pinocchio
from pinocchio.robot_wrapper import RobotWrapper

import crocoddyl

import eagle_mpc
from eagle_mpc.utils.path import EAGLE_MPC_YAML_DIR

import numpy as np

def main():
    dt = 1e-2;
    motor_ct = 1e-3;

    robot = RobotWrapper.BuildFromURDF(join(EAGLE_MPC_YAML_DIR, 'iris/description/iris.urdf'),
        package_dirs=[join(EAGLE_MPC_YAML_DIR, '../..')],
        root_joint=pinocchio.JointModelFreeFlyer())
    r_model = robot.model

    mc_params = eagle_mpc.MultiCopterBaseParams()
    mc_params.autoSetup(join(EAGLE_MPC_YAML_DIR,'iris', 'platform', 'iris.yaml'))

    state = crocoddyl.StateMultibodyActuated(r_model, mc_params.n_rotors)
    actuation = crocoddyl.ActuationModelMultiCopterBaseFos(state, mc_params.tau_f, mc_params.cf)

    run_costs = crocoddyl.CostModelSum(state, actuation.nu)
    ter_costs = crocoddyl.CostModelSum(state, actuation.nu)

    #  Regularization state
    res_reg_state = crocoddyl.ResidualModelState(state, state.zero(), actuation.nu)

    w_reg_state = np.ones(state.ndx)
    w_reg_state[6:6+mc_params.n_rotors * 2] = np.zeros(mc_params.n_rotors * 2)
    act_reg_state = crocoddyl.ActivationModelWeightedQuad(w_reg_state)

    cost_reg_state = crocoddyl.CostModelResidual(state, act_reg_state, res_reg_state)

    run_costs.addCost("reg_state", cost_reg_state, 1e-3)

    #   Regularization control
    res_reg_control = crocoddyl.ResidualModelControl(state, actuation.nu)

    cost_reg_control = crocoddyl.CostModelResidual(state, res_reg_control)

    run_costs.addCost("reg_control", cost_reg_control, 1e-3)

    # Regularization state
    target_state = state.zero()
    target_state[2] = 1.5;
    res_target_state = crocoddyl.ResidualModelState(state, target_state, actuation.nu)

    w_target_state = np.zeros(state.ndx)
    w_target_state[:6] = np.ones(6);
    act_target_state = crocoddyl.ActivationModelWeightedQuad(w_target_state)

    cost_target_state = crocoddyl.CostModelResidual(state, act_target_state, res_target_state)

    ter_costs.addCost("target_state", cost_target_state, 100);

    # Action models
    run_dam = crocoddyl.DifferentialActionModelFreeFwdDynamicsActuated(state, actuation, run_costs, motor_ct)
    run_iam = crocoddyl.IntegratedActionModelEuler(run_dam, dt)

    ter_dam = crocoddyl.DifferentialActionModelFreeFwdDynamicsActuated(state, actuation, ter_costs, motor_ct)
    ter_iam = crocoddyl.IntegratedActionModelEuler(ter_dam, dt)

    #     Problem
    num_nodes = 200
    run_iams = num_nodes * [run_iam];
    problem = crocoddyl.ShootingProblem(state.zero(), run_iams, ter_iam)

    solver = crocoddyl.SolverFDDP(problem)
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    solver.solve()
    print()

if __name__ == '__main__':
    main()
