#include <iostream>
#include <vector>

#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/multibody/model.hpp"

#include "eagle_mpc/path.hpp"
#include "eagle_mpc/multicopter-base-params.hpp"

#include "crocoddyl/multibody/states/multibody_actuated.hpp"
#include "crocoddyl/multibody/actuations/multicopter-base-fos.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn-actuated.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"

int main()
{
  // Problem parameters
  double dt = 1e-2;
  double motor_ct = 1e-3;

  pinocchio::Model r_model;
  pinocchio::urdf::buildModel(EAGLE_MPC_YAML_DIR "/iris/description/iris.urdf",
                              pinocchio::JointModelFreeFlyer(), r_model);

  eagle_mpc::MultiCopterBaseParams mc_params;
  mc_params.autoSetup(EAGLE_MPC_YAML_DIR "/iris/platform/iris.yaml");

  boost::shared_ptr<crocoddyl::StateMultibodyActuated> state = boost::make_shared<crocoddyl::StateMultibodyActuated>(boost::make_shared<pinocchio::Model>(r_model), mc_params.n_rotors_);
  boost::shared_ptr<crocoddyl::ActuationModelMultiCopterBaseFos> actuation = boost::make_shared<crocoddyl::ActuationModelMultiCopterBaseFos>(state, mc_params.tau_f_, mc_params.cf_);

  boost::shared_ptr<crocoddyl::CostModelSum> run_costs = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
  boost::shared_ptr<crocoddyl::CostModelSum> ter_costs = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

  // Regularization state
  boost::shared_ptr<crocoddyl::ResidualModelState> res_reg_state = boost::make_shared<crocoddyl::ResidualModelState>(state, state->zero(), actuation->get_nu());

  Eigen::VectorXd w_reg_state = Eigen::VectorXd::Ones(state->get_ndx());
  w_reg_state.segment(6, mc_params.n_rotors_ * 2) = Eigen::VectorXd::Zero(mc_params.n_rotors_ * 2);
  boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> act_reg_state = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(w_reg_state);

  boost::shared_ptr<crocoddyl::CostModelResidual> cost_reg_state = boost::make_shared<crocoddyl::CostModelResidual>(state, act_reg_state, res_reg_state);

  run_costs->addCost("reg_state", cost_reg_state, 1e-3);

  // Regularization control
  boost::shared_ptr<crocoddyl::ResidualModelControl> res_reg_control = boost::make_shared<crocoddyl::ResidualModelControl>(state, actuation->get_nu());

  boost::shared_ptr<crocoddyl::CostModelResidual> cost_reg_control = boost::make_shared<crocoddyl::CostModelResidual>(state, res_reg_control);

  run_costs->addCost("reg_control", cost_reg_control, 1e-3);

  // Regularization state
  Eigen::VectorXd target_state = state->zero();
  target_state(2) = 1.5; // takeoff at 1.5
  boost::shared_ptr<crocoddyl::ResidualModelState> res_target_state = boost::make_shared<crocoddyl::ResidualModelState>(state, target_state, actuation->get_nu());

  Eigen::VectorXd w_target_state = Eigen::VectorXd::Zero(state->get_ndx());
  w_target_state.head(6) = Eigen::VectorXd::Ones(6);
  boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> act_target_state = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(w_target_state);

  boost::shared_ptr<crocoddyl::CostModelResidual> cost_target_state = boost::make_shared<crocoddyl::CostModelResidual>(state, act_target_state, res_target_state);

  ter_costs->addCost("target_state", cost_target_state, 100);

  // Action models
  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamicsActuated> run_dam = boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamicsActuated>(state, actuation, run_costs, motor_ct);
  boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> run_iam = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(run_dam, dt);

  boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamicsActuated> ter_dam = boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamicsActuated>(state, actuation, ter_costs, motor_ct);
  boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> ter_iam = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(ter_dam, dt);

  // // Problem
  std::size_t num_nodes = 200;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> run_iams(num_nodes, run_iam);
  boost::shared_ptr<crocoddyl::ShootingProblem> problem = boost::make_shared<crocoddyl::ShootingProblem>(state->zero(), run_iams, ter_iam);

  boost::shared_ptr<crocoddyl::SolverBoxFDDP> solver = boost::make_shared<crocoddyl::SolverBoxFDDP>(problem);
  std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> solver_callbacks;
  solver_callbacks.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
  solver->setCallbacks(solver_callbacks);
  solver->solve(crocoddyl::DEFAULT_VECTOR, crocoddyl::DEFAULT_VECTOR);

  return 0;
}