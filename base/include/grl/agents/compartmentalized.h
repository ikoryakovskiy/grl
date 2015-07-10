/** \file compartmentalized.h
 * \brief Compartmentalized sub-agent header file.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2015-06-16
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

#ifndef GRL_COMPARTMENTALIZED_AGENT_H_
#define GRL_COMPARTMENTALIZED_AGENT_H_

#include <grl/agent.h>

namespace grl
{

/// Fixed-policy agent.
class CompartmentalizedSubAgent : public SubAgent
{
  public:
    TYPEINFO("agent/sub/compartmentalized", "Sub agent that is valid in a fixed state-space region")

  protected:
    Agent *agent_;
    Vector min_, max_;
    
  public:
    CompartmentalizedSubAgent() : agent_(NULL) { }
  
    // From Configurable    
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Agent
    virtual CompartmentalizedSubAgent *clone() const;
    virtual void start(const Vector &obs, Vector *action);
    virtual void step(double tau, const Vector &obs, double reward, Vector *action);
    virtual void end(double tau, double reward);
    
    // From SubAgent
    double confidence(const Vector &obs) const;
};

}

#endif /* GRL_COMPARTMENTALIZED_AGENT_H_ */