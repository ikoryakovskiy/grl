/** \file sequential.cpp
 * \brief Sequential master agent source file.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2015-09-10
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

#include <grl/agents/sequential.h>
#include <grl/vector.h>

using namespace grl;

REGISTER_CONFIGURABLE(SequentialMasterAgent)
REGISTER_CONFIGURABLE(SequentialAdditiveMasterAgent)

void SequentialMasterAgent::request(ConfigurationRequest *config)
{
  config->push_back(CRP("predictor", "predictor", "Optional (model) predictor", predictor_, true));
  config->push_back(CRP("agent1", "agent", "First subagent, providing the suggested action", agent_[0]));
  config->push_back(CRP("agent2", "agent", "Second subagent, providing the final action", agent_[1]));
  config->push_back(CRP("pub_action1", "signal/vector", "Publisher of the action coming from the first subagent", action_[0], true));
  config->push_back(CRP("pub_action2", "signal/vector", "Publisher of the action coming from the second subagent", action_[1], true));
  config->push_back(CRP("exporter", "exporter", "Optional exporter for transition log (supports time, state, observation, action, reward, terminal)", exporter_, true));
}

void SequentialMasterAgent::configure(Configuration &config)
{
  predictor_ = (Predictor*)config["predictor"].ptr();
  agent_[0] = (SubAgent*)config["agent1"].ptr();
  agent_[1] = (SubAgent*)config["agent2"].ptr();
  action_[0] = (VectorSignal*)config["pub_action1"].ptr();
  action_[1] = (VectorSignal*)config["pub_action2"].ptr();

  exporter_ = (Exporter*) config["exporter"].ptr();

  if (exporter_)
  {
    exporter_->init({"time", "action0", "action1"});
    exporter_->open("", 0);
  }
}

void SequentialMasterAgent::reconfigure(const Configuration &config)
{
}

void SequentialMasterAgent::report(std::ostream &os)
{
  agent_[0]->report(os);
  agent_[1]->report(os);
}

void SequentialMasterAgent::start(const Observation &obs, Action *action)
{
  time_ = 0;

  agent_[0]->start(obs, action);
  if (action_[0])
    action_[0]->set(action->v);
  if (exporter_)
    exporter_->append({grl::VectorConstructor(time_), *action});

  agent_[1]->start(obs, action);
  if (action_[1])
    action_[1]->set(action->v);
  if (exporter_)
    exporter_->append({*action});

  prev_obs_ = obs;
  prev_action_ = *action;
}

void SequentialMasterAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  time_ += tau;

  agent_[0]->step(tau, obs, reward, action);
  if (action_[0])
    action_[0]->set(action->v);
  if (exporter_)
    exporter_->append({grl::VectorConstructor(time_), *action});

  agent_[1]->step(tau, obs, reward, action);
  if (action_[1])
    action_[1]->set(action->v);
  if (exporter_)
    exporter_->append({*action});

  if (predictor_)
  {
    Transition t(prev_obs_, prev_action_, reward, obs, *action);
    predictor_->update(t);
  }

  prev_obs_ = obs;
  prev_action_ = *action;
}

void SequentialMasterAgent::end(double tau, const Observation &obs, double reward)
{
  agent_[0]->end(tau, obs, reward);
  agent_[1]->end(tau, obs, reward);

  if (predictor_)
  {
    Transition t(prev_obs_, prev_action_, reward, obs);
    predictor_->update(t);
  }
}
/////////////////////////////////////

void SequentialAdditiveMasterAgent::request(ConfigurationRequest *config)
{
  SequentialMasterAgent::request(config);
  config->push_back(CRP("output_min", "vector.action_min", "Lower limit on outputs", min_, CRP::System));
  config->push_back(CRP("output_max", "vector.action_max", "Upper limit on outputs", max_, CRP::System));
}

void SequentialAdditiveMasterAgent::configure(Configuration &config)
{
  SequentialMasterAgent::configure(config);
  min_ = config["output_min"].v();
  max_ = config["output_max"].v();
}

void SequentialAdditiveMasterAgent::reconfigure(const Configuration &config)
{
}

void SequentialAdditiveMasterAgent::start(const Observation &obs, Action *action)
{
  time_ = 0;

  // First action
  agent_[0]->start(obs, action);
  if (action_[0])
    action_[0]->set(action->v);
  if (exporter_)
    exporter_->append({grl::VectorConstructor(time_), *action});

  // Second action
  Action action1;
  agent_[1]->start(obs, &action1);
  if (action_[1])
    action_[1]->set(action1.v);
  if (exporter_)
    exporter_->append({action1});

  // Add those
  action->v += action1.v;
  for (size_t ii=0; ii < (*action).size(); ++ii)
    (*action)[ii] = fmin(fmax((*action)[ii], min_[ii]), max_[ii]);
    
  // If any action type is undefined, the result is undefined
  // Otherwise, if any action type is exploratory, the result is exploratory.
  // Otherwise (both actions are greedy), the result is greedy.
  action->type = std::min(action->type, action1.type);

  prev_obs_ = obs;
  prev_action_ = *action;
}

void SequentialAdditiveMasterAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  time_ += tau;

  // First action
  timer t;
  agent_[0]->step(tau, obs, reward, action);
  if (action_[0])
    action_[0]->set(action->v);
  TRACE("Elapsed time (1): " << t.elapsed() << " s" << std::endl);
  if (exporter_)
    exporter_->append({grl::VectorConstructor(time_), *action});

  // Second action
  Action action1;
  t.restart();
  agent_[1]->step(tau, obs, reward, &action1);
  if (action_[1])
    action_[1]->set(action1.v);
  TRACE("Elapsed time (2) " << t.elapsed() << " s" << std::endl);
  if (exporter_)
    exporter_->append({action1});

  // Add those
  action->v += action1.v;
  for (size_t ii=0; ii < (*action).size(); ++ii)
    (*action)[ii] = fmin(fmax((*action)[ii], min_[ii]), max_[ii]);
  action->type = std::min(action->type, action1.type);

  if (predictor_)
  {
    Transition t(prev_obs_, prev_action_, reward, obs, *action);
    predictor_->update(t);
  }

  prev_obs_ = obs;
  prev_action_ = *action;
}

void SequentialAdditiveMasterAgent::end(double tau, const Observation &obs, double reward)
{
  agent_[0]->end(tau, obs, reward);
  agent_[1]->end(tau, obs, reward);

  if (predictor_)
  {
    Transition t(prev_obs_, prev_action_, reward, obs);
    predictor_->update(t);
  }
}

