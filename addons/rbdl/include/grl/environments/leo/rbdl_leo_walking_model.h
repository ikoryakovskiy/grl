/** \file rbdl_leo_walking_model.h
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

#ifndef RBDL_LEO_WALKING_MODEL_H
#define RBDL_LEO_WALKING_MODEL_H

#include <grl/environments/leo/rbdl_leo_model.h>

namespace grl
{

class LeoWalkingSandboxModel : public LeoSandboxModel
{
  public:
    TYPEINFO("sandbox_model/leo_walk", "State transition model that integrates equations of motion and augments state vector with additional elements")

  public:
    LeoWalkingSandboxModel() : sub_ext_state_(NULL), mode_("vc"),
      active_constraint_set_(""), acting_constraint_set_(""),
      active_left_tip_contact_(0), active_right_tip_contact_(0), active_left_heel_contact_(0), active_right_heel_contact_(0),
      knee_mode_("fail_and_restart") { }

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Model
    virtual void start(const Vector &hint, Vector *state);
    virtual double step(const Vector &action, Vector *next);

  protected:
    virtual bool getCollisionPoints(const Vector &state);
    virtual void getConstraintSet(std::string &constraint_name, const int contacts, const int left_tip_contact, const int right_tip_contact, const int left_hip_contact, const int right_hip_contact);
    virtual void checkContactForces();
    virtual void getActingConstraintPoints(const Vector &state);
    virtual int getNumActingConstraintPoints();

  protected:
    Vector rbdl_addition_;
    VectorSignal *sub_ext_state_;
    std::string mode_, active_constraint_set_, acting_constraint_set_;
    double root_to_feet_height_ = -0.393865;
    int active_left_tip_contact_, active_right_tip_contact_, active_left_heel_contact_, active_right_heel_contact_, active_num_contacts_;
    int acting_left_tip_contact_, acting_right_tip_contact_, acting_left_heel_contact_, acting_right_heel_contact_, acting_num_contacts_;
    std::string knee_mode_;
};

}

#endif // RBDL_LEO_WALKING_MODEL_H
