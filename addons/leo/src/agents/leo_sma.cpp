/** \file leo_sma.cpp
 * \brief State-machine agent source file for Leo
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2016-01-01
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

#include <grl/agents/leo_sma.h>
#include <leo.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoStateMachineAgent)

void LeoStateMachineAgent::request(ConfigurationRequest *config)
{
  LeoBaseAgent::request(config);
  config->push_back(CRP("agent_prepare", "agent", "Prepare agent", agent_prepare_, false));
  config->push_back(CRP("agent_standup", "agent", "Safe standup agent", agent_standup_, false));
  config->push_back(CRP("agent_starter", "agent", "Starting agent", agent_starter_, true));
  config->push_back(CRP("agent_main", "agent", "Main agent", agent_main_, false));

  config->push_back(CRP("upright_trigger", "trigger", "Trigger which finishes stand-up phase and triggers preparation agent", upright_trigger_, false));
  config->push_back(CRP("fc_trigger", "trigger", "Trigger which checks for foot contact to ensure that robot is prepared to walk", foot_contact_trigger_, false));
  config->push_back(CRP("starter_trigger", "trigger", "Trigger which initiates a preprogrammed walking at the beginning", starter_trigger_, true));
}

void LeoStateMachineAgent::configure(Configuration &config)
{
  LeoBaseAgent::configure(config);

  agent_prepare_ = (Agent*)config["agent_prepare"].ptr();
  agent_standup_ = (Agent*)config["agent_standup"].ptr();
  agent_starter_ = (Agent*)config["agent_starter"].ptr();
  agent_main_ = (Agent*)config["agent_main"].ptr();

  upright_trigger_ = (Trigger*)config["upright_trigger"].ptr();
  foot_contact_trigger_ = (Trigger*)config["fc_trigger"].ptr();
  starter_trigger_ = (Trigger*)config["starter_trigger"].ptr();
}

void LeoStateMachineAgent::reconfigure(const Configuration &config)
{
}

void LeoStateMachineAgent::start(const Observation &obs, Action *action)
{
  time_ = 0.;
  int touchDown, groundContact, stanceLegLeft;
  unpack_ic(&touchDown, &groundContact, &stanceLegLeft);
  if (failed(obs, stanceLegLeft))
    agent_ = agent_standup_; // standing up from lying position
  else
    agent_ = agent_prepare_; // prepare from hanging position

  agent_->start(obs, action);
}

void LeoStateMachineAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  time_ += tau;

  act(tau, obs, reward, action);

  // ensure limits
  for (int i = 0; i < ljNumDynamixels; i++)
    (*action)[i] = fmin(action_max_[i], fmax((*action)[i], action_min_[i]));
}

void LeoStateMachineAgent::end(double tau, const Observation &obs, double reward)
{
  std::cout << "End should not be called here!" << std::endl;
}

void LeoStateMachineAgent::act(double tau, const Observation &obs, double reward, Action *action)
{
  // obtain contact information for symmetrical switching
  // note that groundContact is not reliable when Leo is moving,
  // but is reliable when Leo is standing still
  int touchDown, groundContact, stanceLegLeft;
  unpack_ic(&touchDown, &groundContact, &stanceLegLeft);

  // if Leo fell down and we are not trying to stand up already, then try!
  if ((agent_ != agent_standup_) && (failed(obs, stanceLegLeft)))
    return set_agent(agent_standup_, tau, obs, reward, action, "Leo fall down, need to stand up!");

  if (agent_ == agent_prepare_)
  {
    // if Leo is in the upright position wait for the contact before we start the starter
    // or the main agent
    Vector gc = VectorConstructor(groundContact);
    if (foot_contact_trigger_->check(time_, gc))
    {
      if (agent_starter_ && starter_trigger_ && !starter_trigger_->check(time_, Vector()))
        return set_agent(agent_starter_, tau, obs, reward, action, "Starter!");
      else
        return set_agent(agent_main_, tau, obs, reward, action, "Main directly!");
    }
  }

  if (agent_ == agent_starter_)
  {
    // run starter agent for some time, it helps the main agent to start
    if (starter_trigger_->check(time_, Vector()))
      return set_agent(agent_main_, tau, obs, reward, action, "Main!");
  }

  if (agent_ == agent_standup_)
  {
    // try to stand up (body should be in the upright position)
    if (upright_trigger_->check(time_, obs))
      return set_agent(agent_prepare_, tau, obs, reward, action, "Prepare!");
  }

  agent_->step(tau, obs, reward, action);
}

void LeoStateMachineAgent::set_agent(Agent *agent, double tau, const Observation &obs, double reward, Action *action, const char* msg)
{
  agent_->end(tau, obs, reward);  // finish previous agent
  agent_ = agent;                 // switch to the new agent
  agent_->start(obs, action);     // start it to obtain action
  INFO(msg);
}

