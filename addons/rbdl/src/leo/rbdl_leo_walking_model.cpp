/** \file rbdl_leo_walking_model.cpp
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

#include <grl/environments/leo/rbdl_leo_walking_model.h>
#include <grl/environments/leo/rbdl_leo_walking_task.h>
#include <DynamixelSpecs.h>
#include <iomanip>

using namespace grl;

REGISTER_CONFIGURABLE(LeoWalkingSandboxModel)

void LeoWalkingSandboxModel::request(ConfigurationRequest *config)
{
  LeoSandboxModel::request(config);
  config->push_back(CRP("mode", "Control mode (torque/voltage )", mode_, CRP::Configuration, {"tc", "vc"}));
  config->push_back(CRP("sub_ext_state","signal/vector","external state", sub_ext_state_, true));
  config->push_back(CRP("knee_mode", "Select the mode knee constrain is handled", knee_mode_, CRP::Configuration, {"fail_and_restart", "punish_and_continue", "continue"}));
}

void LeoWalkingSandboxModel::configure(Configuration &config)
{
  LeoSandboxModel::configure(config);
  mode_ = config["mode"].str();
  sub_ext_state_ = (VectorSignal*)config["sub_ext_state"].ptr();
  knee_mode_ = config["knee_mode"].str();
}

void LeoWalkingSandboxModel::start(const Vector &hint, Vector *state)
{
  // Fill parts of a state such as Center of Mass, Angular Momentum
  dynamics_->updateKinematics(*state);
  dynamics_->finalize(*state, rbdl_addition_);

  // Compose a complete state <state, time, height, com, ...,>
  state_.resize(rlwStateDim);
  state_ << *state, rbdl_addition_, 0;
  state_[rlwPrevComX] = state_[rlwComX];
  *state = state_;

  active_left_heel_contact_ = 0;
  active_right_heel_contact_ = 0;
  active_left_tip_contact_ = 0;
  active_right_tip_contact_ = 0;
  acting_left_heel_contact_ = 0;
  acting_right_heel_contact_ = 0;
  acting_left_tip_contact_ = 0;
  acting_right_tip_contact_ = 0;
  active_constraint_set_ = "";
  acting_constraint_set_ = "";

  export_meshup_animation(state_[rlwTime], state_, ConstantVector(target_dof_, 0));

  TRACE("Initial state: " << state_);
}

double LeoWalkingSandboxModel::step(const Vector &action, Vector *next)
{
  Vector rbdl_addition_mid, next_state_mid, qd_plus, sub_state_drl;
  bool check;

  qd_plus.resize(target_dof_);
  next_state_mid.resize(rlwStateDim); // avoid copying previous torso X position
  target_state_.resize(2*target_dof_+1);
  target_state_next_.resize(2*target_dof_+1);
  target_action_.resize(target_dof_);
  next->resize(state_.size());

  if (sub_ext_state_) // If you receive the state from DRL
  {
    sub_state_drl = sub_ext_state_->get();
    target_state_ << sub_state_drl, state_[rlwTime];
    dynamics_->updateKinematics(target_state_);
    dynamics_->finalize(target_state_,rbdl_addition_mid);
    state_ << target_state_, rbdl_addition_mid;

    getActingConstraintPoints(state_);
    acting_num_contacts_ = getNumActingConstraintPoints();
    getConstraintSet(acting_constraint_set_, acting_num_contacts_, acting_left_tip_contact_, acting_right_tip_contact_, acting_left_heel_contact_, acting_right_heel_contact_);
    dynamics_->updateActingConstraintSet(acting_constraint_set_);
  }
  else
  {
    // reduce state
    target_state_ << state_.head(2*target_dof_+1);
  }

  target_action_ << action;

  double tau = 0;

  if (target_env_)
  {
    Observation obs;
    tau = target_env_->step(target_action_, &obs, NULL, NULL);
    target_state_next_ <<  obs.v, VectorConstructor(target_state_[rlwTime] + tau);
    dynamics_->updateKinematics(target_state_next_); // update kinematics if rbdl integration is not used
  }
  else
  {
    for (int ii=0; ii < 100; ++ii)
    {
      tau += dm_.step(target_state_, target_action_, &target_state_next_);

      // Knee constraints
      if (knee_mode_ == "continue")
      {
        if (target_state_next_[rlwLeftKneeAngle] > 0 && target_state_next_[rlwLeftKneeAngleRate] > 0)
        {
          target_state_next_[rlwLeftKneeAngle] = target_state_[rlwLeftKneeAngle];
          target_state_next_[rlwLeftKneeAngleRate] = 0;
        }
        if (target_state_next_[rlwRightKneeAngle] > 0 && target_state_next_[rlwRightKneeAngleRate] > 0)
        {
          target_state_next_[rlwRightKneeAngle] = target_state_[rlwRightKneeAngle];
          target_state_next_[rlwRightKneeAngleRate] = 0;
        }
      }

      // Add additional states
      dynamics_->finalize(target_state_next_, rbdl_addition_mid);
      next_state_mid << target_state_next_, rbdl_addition_mid, 0;

      // Check for collision points and update active constraint set
      check = getCollisionPoints(next_state_mid);
      // Update velocities if found in violation of constraints
      if (check)
      {
        acting_left_heel_contact_ = (int)(active_left_heel_contact_ || acting_left_heel_contact_);
        acting_right_heel_contact_ = (int)(active_right_heel_contact_ || acting_right_heel_contact_);
        acting_left_tip_contact_ = (int)(active_left_tip_contact_ || acting_left_tip_contact_);
        acting_right_tip_contact_ = (int)(active_right_tip_contact_ || acting_right_tip_contact_);
        active_num_contacts_ = getNumActingConstraintPoints();

        getConstraintSet(active_constraint_set_, active_num_contacts_, acting_left_tip_contact_, acting_right_tip_contact_, acting_left_heel_contact_, acting_right_heel_contact_);
        dynamics_->updateActiveConstraintSet(active_constraint_set_);
        dynamics_->calcCollisionImpactRhs(target_state_next_, qd_plus);
        // Update state based on new velocities
        for (int ij=0; ij < target_dof_; ++ij)
        {
          target_state_next_[target_dof_+ij] = qd_plus[ij];
        }
      }

      checkContactForces();
      acting_num_contacts_ = getNumActingConstraintPoints();
      getConstraintSet(acting_constraint_set_, acting_num_contacts_, acting_left_tip_contact_, acting_right_tip_contact_, acting_left_heel_contact_, acting_right_heel_contact_);
      dynamics_->updateActingConstraintSet(acting_constraint_set_);

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
  (*next) << target_state_next_, rbdl_addition_, state_[rlwComX];
  export_meshup_animation((*next)[rlwTime], *next, target_action_);

  state_ = *next;
  return tau;
}

bool LeoWalkingSandboxModel::getCollisionPoints(const Vector &state)
{
  grl_assert(state.size() == rlwStateDim);

  bool check = false;

  if ((state[rlwLeftTipZ] < root_to_feet_height_) && (state[rlwLeftTipVelZ] < 0) && (!acting_left_tip_contact_))
  {
    active_left_tip_contact_ = 1;
    active_num_contacts_ += 1;
    check = true;
  }
  if ((state[rlwRightTipZ] < root_to_feet_height_) && (state[rlwRightTipVelZ] < 0) && (!acting_right_tip_contact_))
  {
    active_right_tip_contact_ = 1;
    active_num_contacts_ += 1;
    check = true;
  }
  if ((state[rlwLeftHeelZ] < root_to_feet_height_) && (state[rlwLeftHeelVelZ] < 0) && (!acting_left_heel_contact_))
  {
    active_left_heel_contact_ = 1;
    active_num_contacts_ += 1;
    check = true;
  }
  if ((state[rlwRightHeelZ] < root_to_feet_height_) && (state[rlwRightHeelVelZ] < 0) && (!acting_right_heel_contact_))
  {
    active_right_heel_contact_ = 1;
    active_num_contacts_ += 1;
    check = true;
  }
  return check;
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
    if (acting_right_tip_contact_ && (!active_right_tip_contact_))
    {
      dynamics_->getPointForce("tip_right", force);
      if (force[0] <= precision && force[2] <= precision)
      {
        acting_right_tip_contact_ = 0;
      }
    }
    if (acting_left_tip_contact_ && (!active_left_tip_contact_))
    {
      dynamics_->getPointForce("tip_left", force);
      if (force[0] <= precision && force[2] <= precision)
      {
        acting_left_tip_contact_ = 0;
      }
    }
    if (acting_right_heel_contact_ && (!active_right_heel_contact_))
    {
      dynamics_->getPointForce("heel_right", force);
      if (force[0] <= precision && force[2] <= precision)
      {
        acting_right_heel_contact_ = 0;
      }
    }
    if (acting_left_heel_contact_ && (!active_left_heel_contact_))
    {
      dynamics_->getPointForce("heel_left", force);
      if (force[0] <= precision && force[2] <= precision)
      {
        acting_left_heel_contact_ = 0;
      }
    }
  }
}

void LeoWalkingSandboxModel::getActingConstraintPoints(const Vector &state)
{
  grl_assert(state.size() == rlwStateDim);

  if ((state[rlwLeftTipZ] < root_to_feet_height_) && (state[rlwLeftTipVelZ] < 0))
    acting_left_tip_contact_ = 1;
  else
    acting_left_tip_contact_ = 0;

  if ((state[rlwRightTipZ] < root_to_feet_height_) && (state[rlwRightTipVelZ] < 0))
    acting_right_tip_contact_ = 1;
  else
    acting_right_tip_contact_ = 0;

  if ((state[rlwLeftHeelZ] < root_to_feet_height_) && (state[rlwLeftHeelVelZ] < 0))
    acting_left_heel_contact_ = 1;
  else
    acting_left_heel_contact_ = 0;

  if ((state[rlwRightHeelZ] < root_to_feet_height_) && (state[rlwRightHeelVelZ] < 0))
    acting_right_heel_contact_ = 1;
  else
    acting_right_heel_contact_ = 0;
}

int LeoWalkingSandboxModel::getNumActingConstraintPoints()
{
  return acting_left_tip_contact_ + acting_right_tip_contact_ + acting_left_heel_contact_ + acting_right_heel_contact_;
}
