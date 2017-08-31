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

REGISTER_CONFIGURABLE(MPMLEnvironment)
REGISTER_CONFIGURABLE(MPMLBenchmarkingEnvironment)

void MPMLBenchmarkingEnvironment::request(ConfigurationRequest *config)
{
  config->push_back(CRP("sub_nominal_action", "signal/vector", "Subscriber to an external action from a nominal controller", sub_nominal_action_, true));
  config->push_back(CRP("weights", "Per-dimention weight of MEF", weights_, CRP::Online));
  config->push_back(CRP("environment", "environment", "Real environment", environment_));
  config->push_back(CRP("task", "task", "Task to perform in the environment (should match model)", task_, true));
  config->push_back(CRP("model", "model", "Approximated model of the system", model_));
  config->push_back(CRP("exporter", "exporter", "Optional exporter for transition log (supports time, state, observation, action, reward, terminal)", exporter_, true));
}

void MPMLBenchmarkingEnvironment::configure(Configuration &config)
{
  sub_nominal_action_ = (VectorSignal*)config["sub_nominal_action"].ptr();
  weights_ = config["weights"].v();
  environment_ = (Environment*)config["environment"].ptr();
  task_ = (Task*)config["task"].ptr();
  model_ = (Model*)config["model"].ptr();
  exporter_ = (Exporter*)config["exporter"].ptr();

  if (exporter_)
  {
    // Register headers
    exporter_->init({"time", "state0", "observation", "action", "nominal_observation", "nominal_action", "mismatch", "terminal"});
  }
}

void MPMLBenchmarkingEnvironment::reconfigure(const Configuration &config)
{
}

MPMLBenchmarkingEnvironment &MPMLBenchmarkingEnvironment::copy(const Configurable &obj)
{
  const MPMLBenchmarkingEnvironment& se = dynamic_cast<const MPMLBenchmarkingEnvironment&>(obj);

  total_reward_ = se.total_reward_;
  mismatch_total_ = se.mismatch_total_;

  return *this;
}
    
void MPMLBenchmarkingEnvironment::start(int test, Observation *obs)
{
  environment_->start(test, obs);

  total_reward_ = 0;
  mismatch_total_ = 0;

  test_ = test;

  if (exporter_)
    exporter_->open((test_?"test":"learn"), (test_?time_test_:time_learn_) != 0.0);
}

/// Imaginary step on the ideal model + real step in the real environment
double MPMLBenchmarkingEnvironment::mpml_step(const Action &action, Observation *obs, double *reward, int *terminal, double *mismatch)
{
  // 1. Obtain state from previous observation
  Vector nominal_state = *obs;
  if (task_)
    task_->invert(*obs, &nominal_state);

  // 2. Step on the real system
  double tau = environment_->step(action, obs, reward, terminal);
  total_reward_ += *reward;

  // 3. Obtain nominal action without any correction
  Vector nominal_action;
  if (sub_nominal_action_)
    nominal_action = sub_nominal_action_->get();
  else
    nominal_action = action;
  grl_assert(nominal_action.size() == action.size());
  Vector nominal_next;
  model_->step(nominal_state, nominal_action, &nominal_next);

  // 4. Project next nominal state to a new observation
  int _;
  Observation nominal_obs = nominal_next;
  if (task_)
    task_->observe(nominal_next, &nominal_obs, &_);

  TRACE(*obs);
  TRACE(nominal_obs);

  // 5. Calculate MPML
  if (weights_.size() == 0)
  {
    int dof = obs->size() / 2;
    weights_.resize(2*dof);
    weights_ << ConstantVector(dof, 1.0), ConstantVector(dof, 0.0);
  }
  Vector x = (nominal_obs.v - obs->v) * weights_;
  *mismatch = - sqrt(x.cwiseProduct(x).sum());
  mismatch_total_ += *mismatch;

  // 6. Export transitions
  double &time = test_?time_test_:time_learn_;
  if (exporter_)
    exporter_->write({VectorConstructor(time), nominal_state, *obs, action, nominal_obs, nominal_action,
                      VectorConstructor(*reward), VectorConstructor((double)*terminal)});
  time += tau;
  return tau;
}


double MPMLBenchmarkingEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  double mismatch;
  double tau = mpml_step(action, obs, reward, terminal, &mismatch);
  return tau;
}

void MPMLBenchmarkingEnvironment::report(std::ostream &os) const
{
  os << std::setw(15) << total_reward_  << std::setw(15) << mismatch_total_;

  environment_->report(os);
}

double MPMLEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  double mismatch;
  double tau = mpml_step(action, obs, reward, terminal, &mismatch);
  *reward = mismatch; // overwrite reward with mismatch
  return tau;
}


