/** \file sandbox.cpp
 * \brief Non-Markov environment and dynamical model source file.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2016-01-22
 *
 * \copyright \verbatim
 * Copyright (c) 2015, Wouter Caarls
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

#include <grl/environment.h>
#include <iomanip>

using namespace grl;

REGISTER_CONFIGURABLE(SandboxEnvironment)

void SandboxEnvironment::request(ConfigurationRequest *config)
{
  config->push_back(CRP("model", "sandbox_model", "Environment model", sandbox_));
  config->push_back(CRP("task", "task", "Task to perform in the environment (should match model)", task_));
  config->push_back(CRP("exporter", "exporter", "Optional exporter for transition log (supports time, state, observation, action, reward, terminal)", exporter_, true));
  config->push_back(CRP("state", "signal/vector", "Current state of the model", CRP::Provided));
}

void SandboxEnvironment::configure(Configuration &config)
{
  sandbox_ = (Sandbox*)config["model"].ptr();
  task_ = (Task*)config["task"].ptr();
  exporter_ = (Exporter*)config["exporter"].ptr();

  if (exporter_)
  {
    // Register headers
    exporter_->init({"time", "state", "observation", "action", "reward", "terminal"});
  }

  state_obj_ = new VectorSignal();

  config.set("state", state_obj_);
}

void SandboxEnvironment::reconfigure(const Configuration &config)
{
  sandbox_->reconfigure(config);
  task_->reconfigure(config);

  if (config.has("action") && config["action"].str() == "reset")
    time_learn_ = time_test_ = 0.;
  if (config.has("action") && config["action"].str() == "statclr")
  {
    sandbox_->start(ConstantVector(1, test_), &state_);
    state_obj_->set(state_);
  }
}

SandboxEnvironment &SandboxEnvironment::copy(const Configurable &obj)
{
  const SandboxEnvironment& se = dynamic_cast<const SandboxEnvironment&>(obj);

  obs_ = se.obs_;
  test_ = se.test_;
  
  return *this;
}


void SandboxEnvironment::start(int test, Observation *obs)
{
  int terminal;

  task_->start(test, &state_);
  sandbox_->start(ConstantVector(1, test), &state_);
  task_->observe(state_, obs, &terminal);

  obs_ = *obs;
  state_obj_->set(state_);

  test_ = test;

  if (exporter_)
    exporter_->open((test_?"test":"learn"), (test_?time_test_:time_learn_) != 0.0);

  time_start_ = test_?time_test_:time_learn_;
}

double SandboxEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  Vector next = state_, actuation;
  double tau = 0;
  bool done = false;

  do
  {
    done = task_->actuate(next, action, &actuation);
    tau += sandbox_->step(actuation, &next);
  } while (!done);
  
  task_->observe(next, obs, terminal);
  task_->evaluate(state_, actuation, next, reward);

  double &time = test_?time_test_:time_learn_;

  if (exporter_)
    // was action !!
    exporter_->write({VectorConstructor(time), state_, *obs, actuation, VectorConstructor(*reward), VectorConstructor((double)*terminal)});

  time += tau;

  state_ = next;
  obs_ = *obs;
  state_obj_->set(state_);

  return tau;
}

void SandboxEnvironment::report(std::ostream &os) const
{
  const int pw = 15;
  std::stringstream progressString;
  progressString << std::fixed << std::setprecision(3) << std::right;
  progressString << std::setw(pw) << (test_?time_test_:time_learn_) - time_start_;
  os << progressString.str();

  sandbox_->report(os);
  task_->report(os, state_);
}

