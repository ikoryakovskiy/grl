/** \file leo_base.h
 * \brief Base agent header file for Leo
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

#ifndef GRL_LEO_BASE_AGENT_H_
#define GRL_LEO_BASE_AGENT_H_

#include <grl/agent.h>
#include <grl/signal.h>

namespace grl
{

class LeoBaseAgent : public Agent
{
  protected:
    VectorSignal *sub_ic_signal_;
    
  public:
    LeoBaseAgent() : sub_ic_signal_(NULL) { }
  
    // From Configurable    
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Agent
    virtual void start(const Observation &obs, Action *action) = 0;
    virtual void step(double tau, const Observation &obs, double reward, Action *action) = 0;
    virtual void end(double tau, const Observation &obs, double reward) = 0;

  protected:
    virtual bool unpack_ic(int *touchDown, int *groundContact, int *stanceLegLeft) const;
    virtual bool failed(const Observation &obs, bool stanceLegLeft) const;
};

}

#endif /* GRL_LEO_BASE_AGENT_H_ */
