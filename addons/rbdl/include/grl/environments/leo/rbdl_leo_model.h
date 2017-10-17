/** \file rbdl_leo_model.h
 * \brief RBDL file for C++ description of Leo model.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2016-09-13
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

#ifndef RBDL_LEO_MODEL_H
#define RBDL_LEO_MODEL_H

#include <grl/environment.h>
#include <grl/environments/rbdl.h>
//#include <grl/butterworth.h>
//#include <../../agents/leo_sma.h>

namespace grl
{

class LeoSandboxModel: public Sandbox
{
  public:
    LeoSandboxModel() : target_dof_(4), target_env_(NULL), dynamics_(NULL), sub_true_action_(NULL), true_model_(NULL), sub_sma_state_(NULL) { }

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Model
    virtual void start(const Vector &hint, Vector *state) = 0;
    virtual double step(const Vector &action, Vector *next) = 0;

  protected:
    virtual void export_meshup_animation(double time, const Vector &state, const Vector &action = Vector()) const;

  protected:
    int target_dof_;
    Environment *target_env_;
    DynamicalModel dm_;
    RBDLDynamics *dynamics_;
    std::string animation_;

    VectorSignal *sub_true_action_;
    Model *true_model_;

    VectorSignal *sub_sma_state_;

    Vector target_state_, target_state_next_, target_action_;
};

}

#endif // RBDL_LEO_MODEL_H
