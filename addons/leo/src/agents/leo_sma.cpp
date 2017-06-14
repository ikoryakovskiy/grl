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
  config->push_back(CRP("agent_prepare", "agent", "Prepare agent", agent_prepare_.a, false));
  config->push_back(CRP("agent_standup", "agent", "Safe standup agent", agent_standup_.a, false));
  config->push_back(CRP("agent_starter", "agent", "Starting agent", agent_starter_.a, true));
  config->push_back(CRP("agent_main", "agent", "Main agent", agent_main_.a, false));
  config->push_back(CRP("main_timeout", "Timeout for the main agent to work, switched to agent_standup afterwards", (double)agent_main_timeout_, CRP::Configuration, 0.0, DBL_MAX));

  config->push_back(CRP("upright_trigger", "trigger", "Trigger which finishes stand-up phase and triggers preparation agent", upright_trigger_, false));
  config->push_back(CRP("feet_on_trigger", "trigger", "Trigger which checks for foot contact to ensure that robot is prepared to walk", feet_on_trigger_, true));
  config->push_back(CRP("feet_off_trigger", "trigger", "Trigger which checks for foot contact to detect lifts of the robot", feet_off_trigger_, true));
  config->push_back(CRP("starter_trigger", "trigger", "Trigger which initiates a preprogrammed walking at the beginning", starter_trigger_, true));

  config->push_back(CRP("pub_sma_state", "signal/vector", "Publisher of the type of the agent currently used by state machine", pub_sma_state_, true));

}

void LeoStateMachineAgent::configure(Configuration &config)
{
  LeoBaseAgent::configure(config);

  agent_prepare_.a = (Agent*)config["agent_prepare"].ptr();
  agent_prepare_.s = SMA_PREPARE;
  agent_standup_.a = (Agent*)config["agent_standup"].ptr();
  agent_standup_.s = SMA_STANDUP;
  agent_starter_.a = (Agent*)config["agent_starter"].ptr();
  agent_starter_.s = SMA_STARTER;
  agent_main_.a = (Agent*)config["agent_main"].ptr();
  agent_main_.s = SMA_MAIN;

  agent_main_timeout_ = config["main_timeout"];

  upright_trigger_ = (Trigger*)config["upright_trigger"].ptr();
  feet_on_trigger_ = (Trigger*)config["feet_on_trigger"].ptr();
  feet_off_trigger_ = (Trigger*)config["feet_off_trigger"].ptr();
  starter_trigger_ = (Trigger*)config["starter_trigger"].ptr();

  pub_sma_state_ = (VectorSignal*)config["pub_sma_state"].ptr();
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

  agent_.a->start(obs, action);

  for (int i = 0; i < action_max_.size(); i++)
    (*action)[i] = fmin(action_max_[i], fmax((*action)[i], action_min_[i]));

  if (pub_sma_state_)
    pub_sma_state_->set(VectorConstructor(agent_.s));
}

void LeoStateMachineAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  time_ += tau;

  act(tau, obs, reward, action);

  for (int i = 0; i < action_max_.size(); i++)
    (*action)[i] = fmin(action_max_[i], fmax((*action)[i], action_min_[i]));

  if (pub_sma_state_)
    pub_sma_state_->set(VectorConstructor(agent_.s));
}

void LeoStateMachineAgent::end(double tau, const Observation &obs, double reward)
{
  std::cout << "End should not be called here!" << std::endl;
}

void LeoStateMachineAgent::act(double tau, const Observation &obs, double reward, Action *action)
{
  // obtain contact information for symmetrical switching
  // note that groundContact is not reliable when Leo is moving,
  // but is reliable when Leo is standing still.
  // Therefore we use a trigger to make sure the contact is lost over some amount of time.
  // Also, note that if "sub_ic_signal_" is not used, then robot is assumed to be on the ground (i.e. ready for operation)
  int touchDown = 0, groundContact = 1, stanceLegLeft = 1;
  unpack_ic(&touchDown, &groundContact, &stanceLegLeft);
  Vector gc = VectorConstructor(groundContact);

  // if Leo looses ground contact
  if (feet_off_trigger_ && feet_off_trigger_->check(time_, gc))
  {
    if (failed(obs, stanceLegLeft))
      // lost contact due to fall => try to standup
      set_agent(agent_standup_, tau, obs, reward, action, "Lost ground contact, need to stand up!");
    else
    {
      // lost contact due to lift => move prepare to continue walking
      if (agent_ != agent_standup_) // wait for standing up if needed
        set_agent(agent_prepare_, tau, obs, reward, action, "Lost ground contact, already upright!");
    }
  }

  // if Leo fell down and we are not trying to stand up already, then try!
  if (failed(obs, stanceLegLeft) || (agent_main_timeout_ && agent_ == agent_main_ && time_ - agent_main_time_ > agent_main_timeout_))
    return set_agent(agent_standup_, tau, obs, reward, action, "Main terminated (fall or timeout). Idle controller is applied.");

  if (agent_ == agent_prepare_)
  {
    // if Leo is in the upright position wait for the contact before we start the starter
    // or the main agent
    if (feet_on_trigger_->check(time_, gc))
    {
      if (agent_starter_.a && starter_trigger_ && !starter_trigger_->check(time_, Vector()))
        return set_agent(agent_starter_, tau, obs, reward, action, "Starter!");
      else
        set_agent_main(tau, obs, reward, action, "Main directly!");
    }
  }

  if (agent_ == agent_starter_)
  {
    // run starter agent for some time, it helps the main agent to start
    if (starter_trigger_->check(time_, Vector()))
      set_agent_main(tau, obs, reward, action, "Main!");
  }

  if (agent_ == agent_standup_)
  {
    // try to stand up (body should be in the upright position)
    if (upright_trigger_->check(time_, obs))
      return set_agent(agent_prepare_, tau, obs, reward, action, "Prepare!");
  }

  agent_.a->step(tau, obs, reward, action);
}

void LeoStateMachineAgent::set_agent(SMAgent &agent, double tau, const Observation &obs, double reward, Action *action, const char* msg)
{
  if (agent_ != agent)
  {
    agent_.a->end(tau, obs, reward);  // finish previous agent
    agent_ = agent;                   // switch to the new agent
    agent_.a->start(obs, action);     // start it to obtain action
    INFO(msg);
  }
}

void LeoStateMachineAgent::set_agent_main(double tau, const Observation &obs, double reward, Action *action, const char* msg)
{
  struct timespec start_time, now;
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  set_agent(agent_main_, tau, obs, reward, action, msg);
  clock_gettime(CLOCK_MONOTONIC, &now);

  // add start-up time (critical for NMPC because it is long, 1-2s)
  agent_main_time_ = time_ + now.tv_sec - start_time.tv_sec + (now.tv_nsec - start_time.tv_nsec) / 1000000000.0;;
  return;
}
