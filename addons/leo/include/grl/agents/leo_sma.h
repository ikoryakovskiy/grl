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

#include <grl/environment.h>
#include <grl/agents/leo_base.h>
#include <grl/state_machine.h>

namespace grl
{

enum SMAState {SMA_NONE, SMA_PREPARE, SMA_STANDUP, SMA_STARTER, SMA_MAIN, SMA_TEST};

/// State machine agent.
class LeoStateMachineAgent : public LeoBaseAgent
{
  struct SMAgent
  {
    SMAgent(Agent *_agent, SMAState _type = SMA_NONE) : a(_agent), s(_type) {}
    bool operator==(const SMAgent &other) const { return (this->a == other.a); }
    bool operator!=(const SMAgent &other) const { return (this->a != other.a); }
    Agent *a;
    SMAState s;
  };

  public:
    TYPEINFO("agent/leo/sma", "State-machine agent for Leo")

  protected:
    Environment *environment_;
    SMAgent agent_prepare_, agent_standup_, agent_starter_, agent_main_, agent_test_;
    SMAgent agent_;
    Trigger *upright_trigger_, *feet_on_trigger_, *feet_off_trigger_, *starter_trigger_;
    double time_, main_time_, timeout_;
    int steps_, ss_, tt_;
    double main_total_reward_;
    int test_interval_;
    std::string output_;

    VectorSignal *pub_sma_state_;

    pthread_t *thread_;

    std::ofstream ofs_;
    
  public:
    LeoStateMachineAgent() :
      environment_(NULL),
      agent_prepare_(NULL),
      agent_standup_(NULL),
      agent_starter_(NULL),
      agent_main_(NULL),
      agent_test_(NULL),
      agent_(NULL),
      upright_trigger_(NULL),
      feet_on_trigger_(NULL),
      feet_off_trigger_(NULL),
      starter_trigger_(NULL),
      time_(0.),
      main_time_(0.),
      timeout_(0),
      steps_(0),
      ss_(0),
      tt_(0),
      main_total_reward_(0),
      test_interval_(-1),
      output_(""),
      pub_sma_state_(NULL),
      thread_(NULL)
    { }
  
    ~LeoStateMachineAgent();

    // From Configurable    
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Agent
    virtual void start(const Observation &obs, Action *action);
    virtual void step(double tau, const Observation &obs, double reward, Action *action);
    virtual void end(double tau, const Observation &obs, double reward);

  protected:
    virtual void set_agent(SMAgent &agent, double tau, const Observation &obs, double reward, Action *action, const char* msg);
    virtual bool set_agent_main(double tau, const Observation &obs, double reward, Action *action, const char* msg);
    virtual void save(SMAgent &agent);
    virtual bool save_completed();
    virtual void report(SMAgent &agent);

    virtual void act(double tau, const Observation &obs, double reward, Action *action);
};

}

#endif /* GRL_LEO_STATE_MACHINE_AGENT_H_ */
