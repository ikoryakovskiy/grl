/** \file selective.h
 * \brief Selective master agent header file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-04-04
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

#ifndef GRL_SELECTIVE_MASTER_AGENT_H_
#define GRL_SELECTIVE_MASTER_AGENT_H_

#include <grl/predictor.h>
#include <grl/agent.h>

namespace grl
{

/// Master agent that treats timesteps in which a subagent doesn't run as smdp macro-steps.
class SelectiveMasterAgent : public Agent
{
  public:
    TYPEINFO("agent/master/selective", "Selective agent which decides wich agent to run based on agents' confidence")

  protected:
    std::vector<SubAgent*> agents_;
    std::vector<double> total_rewards_;
    SubAgent *current_agent_;
    int current_idx_;
    double total_reward_;
    double time_;
    
  public:
    SelectiveMasterAgent() : agents_(2), total_rewards_(0), current_agent_(NULL), current_idx_(0), total_reward_(0), time_(0)
    {
      agents_[0] = agents_[1] = NULL;
    }
  
    // From Configurable    
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);
    virtual void report(std::ostream &os);

    // From Agent
    virtual void start(const Observation &obs, Action *action);
    virtual void step(double tau, const Observation &obs, double reward, Action *action);
    virtual void end(double tau, const Observation &obs, double reward);
    
    virtual size_t selectSubAgent(double time, const Observation &obs, Action *action);
    virtual void stepSubAgent(int idx, double tau, const Observation &obs, double reward, Action *action);
};

}

#endif /* GRL_SELECTIVE_MASTER_AGENT_H_ */
