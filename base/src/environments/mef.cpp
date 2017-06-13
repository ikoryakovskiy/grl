/** \file mef.cpp
 * \brief Environment modifier that substitutes reward with Model-error reward.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-06-11
 *
 * \copyright \verbatim
 * Copyright (c) 2016, Wouter Caarls
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

#include <iomanip>

#include <grl/environment.h>

using namespace grl;

REGISTER_CONFIGURABLE(MEFEnvironment)
REGISTER_CONFIGURABLE(BenchmarkingEnvironment)

void MEFEnvironment::request(ConfigurationRequest *config)
{
  config->push_back(CRP("environment", "environment", "Environment to inject noise into", environment_));
  config->push_back(CRP("task", "task", "Task to perform in the environment (should match model)", task_));
  config->push_back(CRP("state", "signal/vector", "Current state of the real model", state_obj_));

  config->push_back(CRP("nominal_model", "model", "True dynamical model", nominal_model_));
  config->push_back(CRP("sub_nominal_action", "signal/vector", "Subscriber to an external action from a nominal controller", sub_nominal_action_));
}

void MEFEnvironment::configure(Configuration &config)
{
  environment_ = (Environment*)config["environment"].ptr();
  task_ = (Task*)config["task"].ptr();
  state_obj_ = (VectorSignal*)config["state"].ptr();

  nominal_model_ = (Model*)config["nominal_model"].ptr();
  sub_nominal_action_ = (VectorSignal*)config["sub_nominal_action"].ptr();
}

void MEFEnvironment::reconfigure(const Configuration &config)
{
}

MEFEnvironment &MEFEnvironment::copy(const Configurable &obj)
{
  const MEFEnvironment& se = dynamic_cast<const MEFEnvironment&>(obj);

  prev_state_ = se.prev_state_;
  total_reward_ = se.total_reward_;
  
  return *this;
}
    
void MEFEnvironment::start(int test, Observation *obs)
{
  environment_->start(test, obs);
  
  prev_state_ = state_obj_->get();
  total_reward_ = 0;
  mef_total_reward_ = 0;
  test_ = test;
}

double MEFEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  // Real step on a system
  double tau = environment_->step(action, obs, reward, terminal);
  total_reward_ += *reward;
  
  // Imaginary step on a model
  Vector nominal_action = sub_nominal_action_->get();
  Vector nominal_next;
  //for (int i = 0; i < nominal_action.size(); i++)
  //  nominal_action[i] = action[i]; // e.g. in case of Leo, if arm is auto actuated, it is not touched here
  nominal_model_->step(prev_state_, nominal_action, &nominal_next);

  Observation nominal_obs;
  nominal_obs.v.resize( obs->v.size() );
  int _;
  task_->observe(nominal_next, &nominal_obs, &_);

  // Difference in state
  int dof = obs->v.size();// / 2;
  Vector x = nominal_obs.v.head(dof) - obs->v.head(dof);
  *reward = - sqrt(x.cwiseProduct(x).sum());
  mef_total_reward_ += *reward;

  // Record real step
  prev_state_ = state_obj_->get();
  return tau;
}

void MEFEnvironment::report(std::ostream &os) const
{
  os << std::setw(15) << total_reward_  << std::setw(15) << mef_total_reward_;

  environment_->report(os);
}

double BenchmarkingEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  // Real step on a system
  double tau = environment_->step(action, obs, reward, terminal);
  total_reward_ += *reward;

  if (test_)
  {
    // Imaginary step on a model
    Vector nominal_action = sub_nominal_action_->get();
    Vector nominal_next;
    //for (int i = 0; i < nominal_action.size(); i++)
    //  nominal_action[i] = action[i]; // e.g. in case of Leo, if arm is auto actuated, it is not touched here
    nominal_model_->step(prev_state_, nominal_action, &nominal_next);

    Observation nominal_obs;
    nominal_obs.v.resize( obs->v.size() );
    int _;
    task_->observe(nominal_next, &nominal_obs, &_);

    // Difference in state
    int dof = obs->v.size() / 2;
    Vector x = nominal_obs.v.head(dof) - obs->v.head(dof);
    mef_total_reward_ += - sqrt(x.cwiseProduct(x).sum());

    // Record real step
    prev_state_ = state_obj_->get();
  }

  return tau;
}