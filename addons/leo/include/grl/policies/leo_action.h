/** \file action.h
 * \brief Action policy for Leo header file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-04-29
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

#ifndef GRL_LEO_ACTION_POLICY_H_
#define GRL_LEO_ACTION_POLICY_H_

#include <grl/policies/action.h>
#include <grl/projector.h>
#include <grl/representation.h>
#include <grl/discretizer.h>
#include <grl/signal.h>

namespace grl
{

/// Policy based on a direct action representation
class LeoActionPolicyLearn : public ActionPolicy
{
  public:
  TYPEINFO("mapping/policy/leo/action/learn", "Policy based on a direct action representation for Leo during learning")

  protected:
    VectorSignal *sub_sigma_signal_;

  public:
    LeoActionPolicyLearn() : sub_sigma_signal_(NULL) { }

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Policy
    virtual void act(const Observation &in, Action *out) const;
};

/// Policy based on a direct action representation
class LeoActionPolicyTest : public ActionPolicy
{
  public:
    TYPEINFO("mapping/policy/leo/action/test", "Policy based on a direct action representation for Leo during testing")

  protected:
    VectorSignal *pub_sigma_signal_, *sub_error_signal_;

  public:
    LeoActionPolicyTest() : pub_sigma_signal_(NULL), sub_error_signal_(NULL) { }
    
    // From Configurable  
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Policy
    virtual void act(const Observation &in, Action *out) const;
};

}

#endif /* GRL_LEO_ACTION_POLICY_H_ */
