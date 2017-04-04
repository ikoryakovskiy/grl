/** \file selective.cpp
 * \brief Selective master agent source file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-04-04
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

#include <grl/agents/selective.h>

using namespace grl;

REGISTER_CONFIGURABLE(SelectiveMasterAgent)

// *** SelectiveMasterAgent ***

void SelectiveMasterAgent::request(ConfigurationRequest *config)
{
  config->push_back(CRP("agent1", "agent/sub", "First subagent", agent_[0]));
  config->push_back(CRP("agent2", "agent/sub", "Second subagent", agent_[1]));
}

void SelectiveMasterAgent::configure(Configuration &config)
{
  agent_[0] = (SubAgent*)config["agent1"].ptr();
  agent_[1] = (SubAgent*)config["agent2"].ptr();
}

void SelectiveMasterAgent::reconfigure(const Configuration &config)
{

}

void SelectiveMasterAgent::start(const Observation &obs, Action *action)
{
  int idx = selectSubAgent(0, obs, action);

  current_agent_ = agent_[idx];
  current_agent_->start(obs, action);

  time_ = 0;
}

void SelectiveMasterAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  time_ += tau;
  int idx = selectSubAgent(time_, obs, action);
  executeSubAgent(agent_[idx], tau, obs, reward, action);
}

void SelectiveMasterAgent::end(double tau, const Observation &obs, double reward)
{
  current_agent_->end(tau, obs, reward);
}

size_t SelectiveMasterAgent::selectSubAgent(double time, const Observation &obs, Action *action)
{
  // Find most confident agent
  double maxconf = agent_[0]->confidence(obs);
  size_t maxconfa = 0;

  for (size_t ii=1; ii < agent_.size(); ++ii)
  {
    double confidence = agent_[ii]->confidence(obs);
    if (confidence > maxconf)
    {
      maxconf = confidence;
      maxconfa = ii;
    }
  }
  return maxconfa;
}

void SelectiveMasterAgent::executeSubAgent(SubAgent *agent, double tau, const Observation &obs, double reward, Action *action)
{
  if (current_agent_ != agent)
  {
    current_agent_->end(tau, obs, reward);          // finish previous agent
    current_agent_ = agent;                         // switch to the new agent
    current_agent_->start(obs, action);             // start it to obtain action
    INFO("Changing subAgents");
  }
  else
    current_agent_->step(tau, obs, reward, action); // or simply continue
}
