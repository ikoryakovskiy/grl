/** \file leo_base.cpp
 * \brief Base agent source file for Leo
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

#include <grl/agents/leo_base.h>
#include <leo.h>

using namespace grl;

void LeoBaseAgent::request(ConfigurationRequest *config)
{
  config->push_back(CRP("sub_ic_signal", "signal/vector", "Subscriber to the contact signal", sub_ic_signal_, true));
  config->push_back(CRP("action_min", "vector.action_min", "Lower limit on actions", action_min_, CRP::System));
  config->push_back(CRP("action_max", "vector.action_max", "Upper limit on actions", action_max_, CRP::System));
}

void LeoBaseAgent::configure(Configuration &config)
{
  sub_ic_signal_ = (VectorSignal*)config["sub_ic_signal"].ptr();
  action_min_ = config["action_min"].v();
  action_max_ = config["action_max"].v();

  if (action_max_.size() != action_min_.size())
    throw bad_param("agent/leo/base:{action_min,action_max}");
}

bool LeoBaseAgent::failed(const Observation &obs, bool stanceLegLeft) const
{
  double torsoComstraint = 1; // 1
  double stanceComstraint = 0.36*M_PI; // 0.36*M_PI

  /*
  // Torso angle out of range
  if (fabs(obs[ljTorso]) > torsoComstraint)
  {
    std::cout << "[TERMINATION] Torso angle too large" << std::endl;
    return true;
  }

  // Stance leg angle out of range
  int hipStance = stanceLegLeft ? ljHipLeft : ljHipRight;
  if (fabs(obs[ljTorso] + obs[hipStance]) > stanceComstraint)
  {
    std::cout << "[TERMINATION] Stance leg angle too large" << std::endl;
    return true;
  }
  */

  if (fabs(obs[0]+obs[1]+obs[2]) > torsoComstraint)
  {
    //std::cout << "[TERMINATION] Torso angle too large" << std::endl;
    return true;
  }

  return false;
}

bool LeoBaseAgent::unpack_ic(int *touchDown, int *groundContact, int *stanceLegLeft) const
{
  if (sub_ic_signal_)
  {
    Vector signal = sub_ic_signal_->get();
    *touchDown     = (((int)signal[0] & lstSwlTouchDown ) != 0);
    *groundContact = (((int)signal[0] & lstGroundContact) != 0);
    *stanceLegLeft = (((int)signal[0] & lstStanceLeft   ) != 0);
    return true;
  }
  return false;
}

