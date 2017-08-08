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

void LeoSandboxModel::request(ConfigurationRequest *config)
{
  dm_.request(config);
  config->pop_back();
  config->push_back(CRP("target_dof", "int.target_dof", "Number of degrees of freedom of the target model", target_dof_, CRP::Configuration, 0, INT_MAX));
  config->push_back(CRP("target_env", "environment", "Interaction environment", target_env_, true));
  config->push_back(CRP("dynamics", "dynamics/rbdl", "Equations of motion", dm_.dynamics_));
  config->push_back(CRP("true_model", "model", "True dynamical model", true_model_, true));

  config->push_back(CRP("animation", "Save current state or full animation", animation_, CRP::Configuration, {"nope", "full", "immediate"}));
}

void LeoSandboxModel::configure(Configuration &config)
{
  target_dof_ = config["target_dof"];
  target_env_ = (Environment*)config["target_env"].ptr(); // Select a real enviromnent if needed

  dm_.configure(config);
  dynamics_ = (RBDLDynamics*) dm_.dynamics_;

  true_model_ = (Model*)config["true_model"].ptr();

  animation_ = config["animation"].str();
  if (animation_ == "full")
  {
    // remove file because will be append otherwice
    std::remove("sd_leo.csv");
    std::remove("u_leo.csv");
  }
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
  precision_ = VectorConstructor(0.01, 0.01);
  config->push_back(CRP("precision", "vector.precision", "Precision of setpoints box conditions for switching the direction of motion", precision_));
  config->push_back(CRP("mode", "Control mode (torque/voltage )", mode_, CRP::Configuration, {"tc", "vc"}));
  config->push_back(CRP("sim_filtered", "Simulate filtering of velocity when rbdl is used", sim_filtered_, CRP::Configuration, 0, 1));
  config->push_back(CRP("sub_true_action", "signal/vector", "Subscriber to an external original action which is belived to be true", sub_true_action_, true));

  config->push_back(CRP("sub_sma_state", "signal/vector", "Subscriber of the type of the agent currently used by state machine", sub_sma_state_, true));
  config->push_back(CRP("timer_switch", "State-based direction switch (0) and time-based direction switch (1)", timer_switch_, CRP::Configuration, 0, 1));

  config->push_back(CRP("idle_time", "double", "Time after which geo is forced to go to a lower setpoint", idle_time_, CRP::Configuration, 0.0, DBL_MAX));
}

void LeoSquattingSandboxModel::configure(Configuration &config)
{
  LeoSandboxModel::configure(config);

  if (target_dof_ != 4)
    throw Exception("sandbox_model/leo_squatting: target dof is not correct for this task");

  target_state_.resize(2*target_dof_+1);
  target_state_next_.resize(2*target_dof_+1);
  true_state_next_.resize(2*target_dof_+1);
  target_action_.resize(target_dof_);

  if (target_env_ && sim_filtered_)
    throw Exception("sandbox_model/leo_squatting: filtering of real Leo velocities is done on-board");

  lower_height_ = config["lower_height"];
  upper_height_ = config["upper_height"];
  precision_ = config["precision"].v();
  mode_ = config["mode"].str();
  sim_filtered_ = config["sim_filtered"];
  sub_true_action_ = (VectorSignal*)config["sub_true_action"].ptr();
  timer_switch_ = config["timer_switch"];

  sub_sma_state_ = (VectorSignal*)config["sub_sma_state"].ptr();

  if ((true_model_ && !sub_true_action_) || (!true_model_ && sub_true_action_))
    throw Exception("sandbox_model/leo_squatting: if true model is used, true action should be defined as well");

  if (sim_filtered_)
  {
    // Init dynamixel speed filters
    for (int i=0; i<target_dof_; i++)
      speedFilter_[i].init(1.0/dm_.tau_, 5.0);
  }
}

void LeoSquattingSandboxModel::reconfigure(const Configuration &config)
{
  if (config.has("action") && config["action"].str() == "statclr")
  {
    if (config.has("sma_state"))
      sma_state_ = static_cast<SMAState>(config["sma_state"].i());
  }
}

void LeoSquattingSandboxModel::start(const Vector &hint, Vector *state)
{
  Vector rbdl_state = state->head(2*target_dof_+1);
  // Fill parts of a state such as Center of Mass, Angular Momentum
  dynamics_->updateKinematics(rbdl_state);
  dynamics_->finalize(rbdl_state, rbdl_addition_);

  double temperature = (*state)[rlsTemperature];
  double setpoint = (*state)[rlsRefRootZ];

  // Compose a complete state <state, time, height, com, ..., squats>
  // Immediately try to stand up
  state->resize(stsStateDim);
  *state << rbdl_state, VectorConstructor(temperature, setpoint), rbdl_addition_,
      VectorConstructor(0),           // zero mef error
      VectorConstructor(sma_state_),  // none type of state machine
      VectorConstructor(0);           // zero squats

  if (sim_filtered_)
  {
    state_raw_vel_ = state->segment(target_dof_, target_dof_);
    for (int i=0; i<target_dof_; i++)
      speedFilter_[i].clear();
  }

  export_meshup_animation(*state, ConstantVector(target_dof_, 0));

  TRACE("Initial state: " << *state);
}

double LeoSquattingSandboxModel::step(const Vector &action, Vector *next)
{
  Vector state = *next;

  // strip state
  if (sim_filtered_)
    target_state_ << state.head(target_dof_), state_raw_vel_, state[rlsTime]; // provide a raw (correct) vector of velocities
  else
    target_state_ << state.head(2*target_dof_+1);

  // auto-actuate arm
  if (action.size() == target_dof_-1)
  {
    double arma;
    if (mode_ == "vc")
    {
      arma = XM430_VS_RX28_COEFF*(14.0/3.3) * 5.0*(-0.26 - state[rlsArmAngle]);
      arma = fmin(LEO_MAX_DXL_VOLTAGE, fmax(arma, -LEO_MAX_DXL_VOLTAGE)); // ensure voltage within limits
    }
    else
    {
      arma = 0.05*(-0.26 - state[rlsArmAngle]);
      arma = fmin(DXL_MAX_TORQUE, fmax(arma, -DXL_MAX_TORQUE)); // ensure torque within limits
    }
    target_action_ << action, arma;
  }
  else
    target_action_ << action;

//  action_step_ << ConstantVector(target_dof_, 0);

//  std::cout << state_step_ << std::endl;
//  std::cout << "  > Action: " << action_step_ << std::endl;

  // call dynamics of the reduced state
  double temperature = state[rlsTemperature];
  double tau;
  if (target_env_)
  {
    Observation obs;
    tau = target_env_->step(target_action_, &obs, NULL, NULL);
    target_state_next_ <<  obs.v.head(2*target_dof_), VectorConstructor(target_state_[rlsTime] + tau);
    temperature = obs.v.tail(1)[0]; // last element of observation, which is temperature
    dynamics_->updateKinematics(target_state_next_); // update internal state of RBDL
  }
  else
  {
    tau = dm_.step(target_state_, target_action_, &target_state_next_);

    if (sim_filtered_)
    {
      // remember raw velocity
      state_raw_vel_ = target_state_next_.segment(target_dof_, target_dof_);

      // Calculate filtered velocity, same method as inside of Leo
      Vector dx = target_state_next_.head(target_dof_) - target_state_.head(target_dof_);
      double timeDiff = dm_.tau_;
      for (int i = 0; i < target_dof_; i++)
      {
        const double velLimit = 8.0;
        if (dx[i]/timeDiff > velLimit)
          dx[i] = velLimit*timeDiff;
        if (dx[i]/timeDiff < -velLimit)
          dx[i] = -velLimit*timeDiff;
        dx[i] = speedFilter_[i].filter(dx[i]/timeDiff);
      }
      target_state_next_ << target_state_next_.head(target_dof_), dx, target_state_next_[rlsTime];
    }
  }

  dynamics_->finalize(target_state_next_, rbdl_addition_);

  // Compose the next state
  (*next) << target_state_next_, VectorConstructor(temperature, state[rlsRefRootZ]),
      rbdl_addition_, VectorConstructor(state[rlsMEF], state[rlsSMAState], state[stsSquats]);

  int changing_direction_ok = 1;
  if (sub_sma_state_)
  {
    Vector tmp = sub_sma_state_->get();
    if (tmp.size() == 1)
    {
      (*next)[rlsSMAState] = tmp[0];
      changing_direction_ok = ((*next)[rlsSMAState] == SMA_MAIN) || ((*next)[rlsSMAState] == SMA_TEST);
    }
  }

  if (changing_direction_ok)
  {
    main_time_ += tau;
    //INFO("elapsed: " << main_time_);
    if (idle_time_ && main_time_ > idle_time_)
      (*next)[rlsRefRootZ] = lower_height_;
    else
    {
      if (timer_switch_)
      {
        // time-based setpoint switch
        double switch_every = 1.5; // [s]
        double time_loc = std::fmod(main_time_, 2*switch_every);
        (*next)[rlsRefRootZ] = (time_loc < switch_every) ? upper_height_ : lower_height_;
      }
      else
      {
        // State-based setpoint switch
        if (fabs((*next)[rlsComVelocityZ] - 0.0) < precision_[1])
        {
          if (fabs((*next)[rlsRootZ] - lower_height_) < precision_[0])
          {
            (*next)[rlsRefRootZ] = upper_height_;
            //std::cout << "Lower setpoint is reached at " << (*next)[rlsRootZ] << std::endl;
          }
          else if (fabs((*next)[rlsRootZ] - upper_height_) < precision_[0])
          {
            (*next)[rlsRefRootZ] = lower_height_;
            //std::cout << "Upper setpoint is reached at " << (*next)[rlsRootZ] << std::endl;
          }
        }
      }
    }
  }
  else
    main_time_ = 0.;

  // Increase number of half-squats if setpoint changed
  if ((*next)[rlsRefRootZ] != state[rlsRefRootZ])
    (*next)[stsSquats] = state[stsSquats] + 1;

  // Simulate true dynamics for Model Error Feedback algorithm
  if (true_model_ && sub_true_action_)
  {
    Vector true_action = sub_true_action_->get();
    if (true_action.size())
    {
      for (int i = 0; i < true_action.size(); i++)
        target_action_[i] = true_action[i]; // if arm is auto actuated, it is not touched here
      true_model_->step(target_state_, target_action_, &true_state_next_);
      int dof = true_action.size(); // if arm is auto actuated, do not account for it
      Vector x = true_state_next_.head(dof) - target_state_next_.head(dof);
      (*next)[rlsMEF] = - sqrt(x.cwiseProduct(x).sum());
      TRACE("MEF: " << (*next)[rlsMEF]);
    }
    else
      (*next)[rlsMEF] = 0;
  }
  else
    (*next)[rlsMEF] = 0;
/*
  std::cout << "  > " << (int)(*next)[rlsSMAState]
            << "  Height: " << std::fixed << std::setprecision(3) << std::right
            << std::setw(10) << (*next)[rlsRootZ] << std::setw(10) << (*next)[rlsComVelocityZ]
            << std::setw(10) << (*next)[rlsRefRootZ]
            << std::setw(10) << (*next)[rlsMEF] << std::endl;
*/
  export_meshup_animation(*next, target_action_);

  return tau;
}
