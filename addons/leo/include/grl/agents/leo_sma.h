/** \file leo_sma.h
 * \brief State-machine agent header file for Leo
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2015-02-04
 *
 * \copyright \verbatim
 * Copyright (c) 2015, Ivan Koryakovskiy
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

#ifndef GRL_LEO_STATE_MACHINE_AGENT_H_
#define GRL_LEO_STATE_MACHINE_AGENT_H_

#include <grl/agents/leo_base.h>
#include <grl/state_machine.h>

namespace grl
{

/// State machine agent.
class LeoStateMachineAgent : public LeoBaseAgent
{
  public:
    TYPEINFO("agent/leo/sma", "State-machine agent for Leo")

  protected:
    Agent *agent_prepare_, *agent_standup_, *agent_starter_, *agent_main_;
    Agent *agent_;
    Trigger *upright_trigger_, *feet_on_trigger_, *feet_off_trigger_, *starter_trigger_;
    double time_, agent_main_time_, agent_main_timeout_;
    
  public:
    LeoStateMachineAgent() :
      agent_prepare_(NULL),
      agent_standup_(NULL),
      agent_starter_(NULL),
      agent_main_(NULL),
      agent_(NULL),
      upright_trigger_(NULL),
      feet_on_trigger_(NULL),
      feet_off_trigger_(NULL),
      starter_trigger_(NULL),
      /*sub_ic_signal_(NULL),*/
      time_(0.),
      agent_main_time_(0.),
      agent_main_timeout_(0.)
    { }
  
    // From Configurable    
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Agent
    virtual void start(const Observation &obs, Action *action);
    virtual void step(double tau, const Observation &obs, double reward, Action *action);
    virtual void end(double tau, const Observation &obs, double reward);

  protected:
    virtual void set_agent(Agent *agent, double tau, const Observation &obs, double reward, Action *action, const char* msg);
    virtual void set_agent_main(double tau, const Observation &obs, double reward, Action *action, const char* msg);

    virtual void act(double tau, const Observation &obs, double reward, Action *action);
};

}

#endif /* GRL_LEO_STATE_MACHINE_AGENT_H_ */
