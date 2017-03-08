/** \file leo_phantom.h
 * \brief Phantom environment header file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@gmail.com>
 * \date      2017-03-08
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

#ifndef GRL_LEO_PHANTOM_ENVIRONMENT_H_
#define GRL_LEO_PHANTOM_ENVIRONMENT_H_

#include <grl/environment.h>
#include <grl/importer.h>
#include <grl/exporter.h>
#include <time.h>

namespace grl
{

/// An environment which reads state transitions from importer.
/// Can be used when we want to see how agent responds to certain trajectories (can be real) to check approximator(s).
class LeoPhantomEnvironment: public Environment
{
  public:
    TYPEINFO("environment/leo/phantom", "Phantom LEO environment which reads state transitions from importer")
    LeoPhantomEnvironment(): importer_(NULL), exporter_(NULL), idx_(0), pub_ic_signal_(NULL), sub_transition_type_(NULL) {}

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Environment
    virtual void start(int test, Observation *obs);
    virtual double step(const Action &action, Observation *obs, double *reward, int *terminal);

  protected:
    Importer *importer_;
    Exporter *exporter_;
    std::vector<Vector> time_, state0_, contact_;
    int idx_;
    VectorSignal *pub_ic_signal_, *sub_transition_type_;
};

}

#endif /* GRL_LEO_PHANTOM_ENVIRONMENT_H_ */
