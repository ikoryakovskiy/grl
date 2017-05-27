/** \file rbdl_leo_model.cpp
 * \brief RBDL file for C++ description of Leo model.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2016-09-13
 *
 * \copyright \verbatim
 * Copyright (c) 2016, Ivan Koryakovskiy
 * All rights reserved.
 *
 * This file is part of GRL, the Generic Reinforcement Learning library.
 *
 * GRL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * \endverbatim
 */

#include <grl/environments/leo/rbdl_leo_model.h>
#include <grl/environments/leo/rbdl_leo_task.h>
#include <DynamixelSpecs.h>
#include <iomanip>

using namespace grl;

REGISTER_CONFIGURABLE(LeoSquattingSandboxModel)
REGISTER_CONFIGURABLE(LeoWalkingSandboxModel)

void LeoSandboxModel::request(ConfigurationRequest *config)
{
  dm_.request(config);
  config->pop_back();
  config->push_back(CRP("dynamics", "dynamics/rbdl", "Equations of motion", dm_.dynamics_));

  config->push_back(CRP("target_dof", "int.target_dof", "Number of degrees of freedom of the target model", target_dof_, CRP::Configuration, 0, INT_MAX));
  config->push_back(CRP("animation", "Save current state or full animation", animation_, CRP::Configuration, {"nope", "full", "immediate"}));
  config->push_back(CRP("target_env", "environment", "Interaction environment", target_env_, true));

  condition_ = VectorConstructor(0.01, 0.01);
  config->push_back(CRP("condition", "vector.condition", "Box-like conditions for switching direction of squat", condition_));
}

void LeoSandboxModel::configure(Configuration &config)
{
  dm_.configure(config);
  dynamics_ = (RBDLDynamics*) dm_.dynamics_;

  target_env_ = (Environment*)config["target_env"].ptr(); // Select a real enviromnent if needed
  target_dof_ = config["target_dof"];
  animation_ = config["animation"].str();
  condition_ = config["condition"].v();
}

void LeoSandboxModel::export_meshup_animation(const Vector &state, const Vector &action) const
{
  if (animation_ == "nope")
    return;

  std::ios_base::openmode om = (animation_ == "full" && action.size() != 0)?(std::ios_base::app):(std::ios_base::trunc);

  std::ofstream data_stream;
  data_stream.open("sd_leo.csv", om);
  if (!data_stream || 2*target_dof_ > state.size())
  {
    std::cerr << "Error opening file sd_leo.csv" << std::endl;
    abort();
  }
  data_stream << state[rlsTime] << ", ";
  for (int i = 0; i < 2*target_dof_-1; i++)
    data_stream << state[i] << ", ";
  data_stream << state[2*target_dof_-1] << std::endl;
  data_stream.close();

  data_stream.open ("u_leo.csv", om);
  if (!data_stream || target_dof_ > action.size())
  {
    std::cerr << "Error opening file u_leo.csv" << std::endl;
    abort();
  }
  data_stream << state[rlsTime] << ", ";
  for (int i = 0; i < target_dof_-1; i++)
    data_stream << action[i] << ", ";
  data_stream << action[target_dof_-1] << std::endl;
  data_stream.close();
}

///////////////////////////////////////////////

void LeoSquattingSandboxModel::request(ConfigurationRequest *config)
{
  LeoSandboxModel::request(config);

  config->push_back(CRP("lower_height", "double.lower_height", "Lower bound of root height to switch direction", lower_height_, CRP::Configuration, 0.0, DBL_MAX));
  config->push_back(CRP("upper_height", "double.upper_height", "Upper bound of root height to switch direction", upper_height_, CRP::Configuration, 0.0, DBL_MAX));
  config->push_back(CRP("mode", "Control mode (torque/voltage )", mode_, CRP::Configuration, {"tc", "vc"}));
}

void LeoSquattingSandboxModel::configure(Configuration &config)
{
  LeoSandboxModel::configure(config);

  lower_height_ = config["lower_height"];
  upper_height_ = config["upper_height"];
  mode_ = config["mode"].str();
}

void LeoSquattingSandboxModel::start(const Vector &hint, Vector *state)
{
  // Fill parts of a state such as Center of Mass, Angular Momentum
  dynamics_->updateKinematics(*state);
  dynamics_->finalize(*state, rbdl_addition_);

  // Compose a complete state <state, time, height, com, ..., squats>
  // Immediately try to stand up
  state_.resize(stsStateDim);
  state_ << *state, VectorConstructor(upper_height_), rbdl_addition_,
      VectorConstructor(0); // zero squats
  *state = state_;

  export_meshup_animation(state_, ConstantVector(target_dof_, 0));

  TRACE("Initial state: " << state_);
}

double LeoSquattingSandboxModel::step(const Vector &action, Vector *next)
{
  target_state_.resize(2*target_dof_+1);
  target_state_next_.resize(2*target_dof_+1);
  target_action_.resize(target_dof_);
  next->resize(state_.size());

  // reduce state
  target_state_ << state_.block(0, 0, 1, 2*target_dof_+1);

  // auto-actuate arm
  if (action.size() == target_dof_-1)
  {
    double arma;
    if (mode_ == "vc")
    {
      arma = XM430_VS_RX28_COEFF*(14.0/3.3) * 5.0*(-0.26 - state_[rlsLeftArmAngle]);
      arma = fmin(LEO_MAX_DXL_VOLTAGE, fmax(arma, -LEO_MAX_DXL_VOLTAGE)); // ensure voltage within limits
    }
    else
    {
      arma = 0.05*(-0.26 - state_[rlsLeftArmAngle]);
      arma = fmin(DXL_MAX_TORQUE, fmax(arma, -DXL_MAX_TORQUE)); // ensure torque within limits
    }

    //std::cout << "Arm: " << -0.26 - state_[rlsArmAngle] << " -> " << arma << std::endl;

    target_action_ << action, arma;
  }
  else
    target_action_ << action;

//  action_step_ << ConstantVector(target_dof_, 0);

//  std::cout << state_step_ << std::endl;
//  std::cout << "  > Action: " << action_step_ << std::endl;

  // call dynamics of the reduced state
  double tau;
  if (target_env_)
  {
    Observation obs;
    tau = target_env_->step(target_action_, &obs, NULL, NULL);
    target_state_next_ <<  obs.v, VectorConstructor(target_state_[rlsTime] + tau);
    dynamics_->updateKinematics(target_state_next_); // update kinematics if rbdl integration is not used
  }
  else
    tau = dm_.step(target_state_, target_action_, &target_state_next_);

  dynamics_->finalize(target_state_next_, rbdl_addition_);

  // Compose the next state
  (*next) << target_state_next_, VectorConstructor(state_[rlsRefRootZ]),
      rbdl_addition_, VectorConstructor(state_[stsSquats]);

  // Switch setpoint if needed
  if (fabs((*next)[rlsComVelocityZ] - 0.0) < condition_[1])
  {
    if (fabs((*next)[rlsRootZ] - lower_height_) < condition_[0])
    {
      (*next)[rlsRefRootZ] = upper_height_;
      //std::cout << "Lower setpoint is reached at " << (*next)[rlsRootZ] << std::endl;
    }
    else if (fabs((*next)[rlsRootZ] - upper_height_) < condition_[0])
    {
      (*next)[rlsRefRootZ] = lower_height_;
      //std::cout << "Upper setpoint is reached at " << (*next)[rlsRootZ] << std::endl;
    }
  }

  // Increase number of half-squats if setpoint changed
  if ((*next)[rlsRefRootZ] != state_[rlsRefRootZ])
    (*next)[stsSquats] = state_[stsSquats] + 1;

  //std::cout << "  > Height: " << std::fixed << std::setprecision(3) << std::right
  //          << std::setw(10) << (*next)[rlsRootZ] << std::setw(10) << (*next)[rlsComVelocityZ]
  //          << std::setw(10) << (*next)[rlsRefRootZ] << std::endl;
  //std::cout << "  > Next state: " << std::fixed << std::setprecision(3) << std::right << std::setw(10) << *next << std::endl;

  export_meshup_animation(*next, target_action_);

  state_ = *next;
  return tau;
}

////////////////////////////////////////////

void LeoWalkingSandboxModel::request(ConfigurationRequest *config)
{
  LeoSandboxModel::request(config);
  config->push_back(CRP("mode", "Control mode (torque/voltage )", mode_, CRP::Configuration, {"tc", "vc"}));
}

void LeoWalkingSandboxModel::configure(Configuration &config)
{
  LeoSandboxModel::configure(config);
  mode_ = config["mode"].str();
}

void LeoWalkingSandboxModel::start(const Vector &hint, Vector *state)
{
  // Fill parts of a state such as Center of Mass, Angular Momentum
  dynamics_->updateKinematics(*state);
  dynamics_->finalize(*state, rbdl_addition_);

  // Compose a complete state <state, time, height, com, ..., squats>
  state_.resize(rlsStateDim);
  state_ << *state, rbdl_addition_;
  *state = state_;

  active_left_heel_contact_ = 0;
  active_right_heel_contact_ = 0;
  active_left_tip_contact_ = 0;
  active_right_tip_contact_ = 0;
  active_num_contacts_ = 0;
  acting_left_heel_contact_ = 0;
  acting_right_heel_contact_ = 0;
  acting_left_tip_contact_ = 0;
  acting_right_tip_contact_ = 0;
  acting_num_contacts_ = 0;
  active_constraint_set_ = "";
  acting_constraint_set_ = "";

  export_meshup_animation(state_, ConstantVector(target_dof_, 0));

  //TRACE("Initial state: " << state_);
}

double LeoWalkingSandboxModel::step(const Vector &action, Vector *next)
{
  Vector rbdl_addition_mid, next_state_mid, qd_plus;

  qd_plus.resize(target_dof_);
  next_state_mid.resize(rlsStateDim);
  target_state_.resize(2*target_dof_+1);
  target_state_next_.resize(2*target_dof_+1);
  target_action_.resize(target_dof_);
  next->resize(state_.size());

  // reduce state
  target_state_ << state_.head(2*target_dof_+1);
  target_action_ << action;

  double tau = 0;

  for (int ii=0; ii < 1; ++ii)
  {

    if (target_env_)
    {
      Observation obs;
      tau = target_env_->step(target_action_, &obs, NULL, NULL);
      target_state_next_ <<  obs.v, VectorConstructor(target_state_[rlsTime] + tau);
      dynamics_->updateKinematics(target_state_next_); // update kinematics if rbdl integration is not used
    }
    else
    {
      tau += dm_.step(target_state_, target_action_, &target_state_next_);

      // Add additional states
      dynamics_->finalize(target_state_next_, rbdl_addition_mid);
      next_state_mid << target_state_next_, rbdl_addition_mid;

      // Check for collision points and update active constraint set
      getCollisionPoints(next_state_mid);
      getConstraintSet(active_constraint_set_, active_num_contacts_, active_left_tip_contact_, acting_right_tip_contact_, active_left_heel_contact_, active_right_heel_contact_);
      dynamics_->updateActiveConstraintSet(active_constraint_set_);
      // Update velocities if found in violation of constraints
      dynamics_->calcCollisionImpactRhs(target_state_next_, qd_plus);
      // Update state based on new velocities
      for (int ij=0; ij < target_dof_; ++ij)
      {
        target_state_next_[target_dof_+ij] = qd_plus[ij];
      }

      checkContactForces();
      acting_left_heel_contact_ = (int)(active_left_heel_contact_ || acting_left_heel_contact_);
      acting_right_heel_contact_ = (int)(active_right_heel_contact_ || acting_right_heel_contact_);
      acting_left_tip_contact_ = (int)(active_left_tip_contact_ || acting_left_tip_contact_);
      acting_right_tip_contact_ = (int)(active_right_tip_contact_ || acting_right_tip_contact_);
      acting_num_contacts_ = acting_left_tip_contact_ + acting_right_tip_contact_ + acting_left_heel_contact_ + acting_right_heel_contact_;
      getConstraintSet(acting_constraint_set_, acting_num_contacts_, acting_left_tip_contact_, acting_right_tip_contact_, acting_left_heel_contact_, acting_right_heel_contact_);
      dynamics_->updateActingConstraintSet(acting_constraint_set_);

      grl_assert(next_state_mid.size() == rlsStateDim);

      // Transfer old values to new
      target_state_ = target_state_next_;
      active_left_heel_contact_ = 0;
      active_right_heel_contact_ = 0;
      active_left_tip_contact_ = 0;
      active_right_tip_contact_ = 0;
      active_num_contacts_ = 0;
    }
  }

  dynamics_->finalize(target_state_next_, rbdl_addition_);

  // Compose the next state
  (*next) << target_state_next_, rbdl_addition_;
  export_meshup_animation(*next, target_action_);

  state_ = *next;
  return tau;
}


void LeoWalkingSandboxModel::getCollisionPoints(const Vector &state)
{
  grl_assert(state.size() == rlsStateDim);

  if ((state[rlsLeftTipZ] < root_to_feet_height_) && (state[rlsLeftTipVelZ] < 0) && (!acting_left_tip_contact_))
  {
    active_left_tip_contact_ = 1;
    active_num_contacts_ += 1;
  }
  if ((state[rlsRightTipZ] < root_to_feet_height_) && (state[rlsRightTipVelZ] < 0) && (!acting_right_tip_contact_))
  {
    active_right_tip_contact_ = 1;
    active_num_contacts_ += 1;
  }
  if ((state[rlsLeftHeelZ] < root_to_feet_height_) && (state[rlsLeftHeelVelZ] < 0) && (!acting_left_heel_contact_))
  {
    active_left_heel_contact_ = 1;
    active_num_contacts_ += 1;
  }
  if ((state[rlsRightHeelZ] < root_to_feet_height_) && (state[rlsRightHeelVelZ] < 0) && (!acting_right_heel_contact_))
  {
    active_right_heel_contact_ = 1;
    active_num_contacts_ += 1;
  }
}

void LeoWalkingSandboxModel::getConstraintSet(std::string &constraint_name, const int contacts, const int left_tip_contact, const int right_tip_contact, const int left_heel_contact, const int right_heel_contact)
{
  constraint_name = "";

  if (contacts == 0)
      return;
  else if (contacts == 1)
  {
    if (left_tip_contact == 1)
      constraint_name = "single_support_tip_left";
    else if (right_tip_contact == 1)
      constraint_name = "single_support_tip_right";
    else if (left_heel_contact == 1)
      constraint_name = "single_support_heel_left";
    else
      constraint_name = "single_support_heel_right";
  }
  else if (contacts == 2)
  {
    if ((left_tip_contact == 1) && (left_heel_contact == 1))
      constraint_name = "single_support_flat_left";
    if ((right_tip_contact == 1) && (right_heel_contact == 1))
       constraint_name = "single_support_flat_right";
    if ((right_tip_contact == 1) && (left_heel_contact == 1))
       constraint_name = "double_support_hl_tr";
    if ((right_tip_contact == 1) && (left_tip_contact == 1))
       constraint_name = "double_support_tip";
    if ((right_heel_contact == 1) && (left_heel_contact == 1))
       constraint_name = "double_support_heel";
    if ((right_heel_contact == 1) && (left_tip_contact == 1))
       constraint_name = "double_support_hr_tl";
  }
  else if (contacts == 3)
  {
    if (!left_tip_contact)
      constraint_name = "double_support_fr_hl";
    if (!right_tip_contact)
      constraint_name = "double_support_fl_hr";
    if (!left_heel_contact)
      constraint_name = "double_support_fr_tl";
    if (!right_heel_contact)
      constraint_name = "double_support_fl_tr";
  }
  else if (contacts == 4)
    constraint_name = "double_support";
}

void LeoWalkingSandboxModel::checkContactForces()
{
  Vector3_t force;
  double precision = 0;

  if (!(acting_constraint_set_.empty()))
  {
    if (acting_right_tip_contact_)
    {
      dynamics_->getPointForce("tip_right", force);
      if (force[0] <= precision && force[2] <= precision)
      {
        acting_right_tip_contact_ = 0;
      }
    }
    if (acting_left_tip_contact_)
    {
      dynamics_->getPointForce("tip_left", force);
      if (force[0] <= precision && force[2] <= precision)
      {
        acting_left_tip_contact_ = 0;
      }
    }
    if (acting_right_heel_contact_)
    {
      dynamics_->getPointForce("heel_right", force);
      if (force[0] <= precision && force[2] <= precision)
      {
        acting_right_heel_contact_ = 0;
      }
    }
    if (acting_left_heel_contact_)
    {
      dynamics_->getPointForce("heel_left", force);
      if (force[0] <= precision && force[2] <= precision)
      {
        acting_left_heel_contact_ = 0;
      }
    }
  }

}
