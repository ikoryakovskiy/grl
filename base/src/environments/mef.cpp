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
REGISTER_CONFIGURABLE(MEFBenchmarkingEnvironment)

void MEFEnvironment::request(ConfigurationRequest *config)
{
  config->push_back(CRP("sub_nominal_action", "signal/vector", "Subscriber to an external action from a nominal controller", sub_nominal_action_));

  config->push_back(CRP("environment", "environment", "Real environment", environment_));
  config->push_back(CRP("state", "signal/vector", "Current state of the environment", state_obj_));
  config->push_back(CRP("task", "task", "Task to perform in the environment (should match model)", task_));
  config->push_back(CRP("model", "model", "Approximated model of the system", model_));
}

void MEFEnvironment::configure(Configuration &config)
{
  sub_nominal_action_ = (VectorSignal*)config["sub_nominal_action"].ptr();

  environment_ = (Environment*)config["environment"].ptr();
  state_obj_ = (VectorSignal*)config["state"].ptr();
  task_ = (Task*)config["task"].ptr();
  model_ = (Model*)config["model"].ptr();
}

void MEFEnvironment::reconfigure(const Configuration &config)
{
}

MEFEnvironment &MEFEnvironment::copy(const Configurable &obj)
{
  const MEFEnvironment& se = dynamic_cast<const MEFEnvironment&>(obj);

  total_reward_ = se.total_reward_;
  mef_total_reward_ = se.mef_total_reward_;

  return *this;
}
    
void MEFEnvironment::start(int test, Observation *obs)
{
  environment_->start(test, obs);

  total_reward_ = 0;
  mef_total_reward_ = 0;
}

double MEFEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  // Get current system state
  Vector state = state_obj_->get();

  // Step on the real system
  double tau = environment_->step(action, obs, reward, terminal);
  total_reward_ += *reward;
  
  // Imaginary step on the ideal model
  Vector nominal_action = sub_nominal_action_->get();
  grl_assert(nominal_action.size() == action.size());
  Vector nominal_next;
  model_->step(state, nominal_action, &nominal_next);

  int _;
  Observation nominal_obs;
  task_->observe(nominal_next, &nominal_obs, &_);

  TRACE(*obs);
  TRACE(nominal_obs);

  // Difference in state
  int dof = obs->v.size() / 2;
  Vector x = nominal_obs.v.head(dof) - obs->v.head(dof);
  *reward = - sqrt(x.cwiseProduct(x).sum()); // reward overwritten with mef
  mef_total_reward_ += *reward;

  return tau;
}

void MEFEnvironment::report(std::ostream &os) const
{
  os << std::setw(15) << total_reward_  << std::setw(15) << mef_total_reward_;

  environment_->report(os);
}

double MEFBenchmarkingEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  // Get current system state
  Vector state = state_obj_->get();

  // Real step on a system
  double tau = environment_->step(action, obs, reward, terminal);
  total_reward_ += *reward;

  // Imaginary step on a model
  Vector nominal_action = sub_nominal_action_->get();
  Vector nominal_next;
  model_->step(state, nominal_action, &nominal_next);

  int _;
  Observation nominal_obs;
  task_->observe(nominal_next, &nominal_obs, &_);

  // Difference in state
  int dof = obs->v.size() / 2;
  Vector x = nominal_obs.v.head(dof) - obs->v.head(dof);
  mef_total_reward_ += - sqrt(x.cwiseProduct(x).sum());

  return tau;
}


