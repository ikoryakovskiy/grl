/** \file leo_walkdynamic.h
 * \brief Leo agent header file to walk dynamically (time-based).
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2016-07-25
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

#ifndef GRL_LEO_WALKDYNAMIC_AGENT_H_
#define GRL_LEO_WALKDYNAMIC_AGENT_H_

#include <grl/agents/leo_base.h>

namespace grl
{

class LeoWalkdynamicAgent : public LeoBaseAgent
{
  public:
    TYPEINFO("agent/leo/walkdynamic", "Leo agent to controll leo walking dynamically")

  protected:
    double time_;

    double      mSwingStartTime;
    double      mParamAnkleStanceAngle, mParamAnklePushoffAngle, mParamInterHipAngle, mParamTransTorqueFactor, mParamTorsoAngle, mParamEarlySwingTime;
    bool        mEarlySwing;  // if true then earlyswing, if false then lateswing

  public:
    LeoWalkdynamicAgent() : time_(0.0)
    {
      mEarlySwing     = false;
      mSwingStartTime = 0;

      // Parameters
      mParamAnkleStanceAngle  =  0.02;
      mParamAnklePushoffAngle =  0.0;
      mParamInterHipAngle     =  0.65;
      mParamTransTorqueFactor = -0.60;
      mParamTorsoAngle        = -0.11; //-0.096;
      mParamEarlySwingTime    =  0.184;
    }
  
    // From Configurable    
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Agent
    virtual void start(const Observation &obs, Action *action);
    virtual void step(double tau, const Observation &obs, double reward, Action *action);
    virtual void end(double tau, const Observation &obs, double reward);

    // Own
    virtual void walk(double time, const Vector &obs, Vector *action);
    virtual bool isShoulderAngleSafe(Vector obs, int leftIsStance, double* safeShoulderAngle);
};

}

#endif /* GRL_LEO_WALKDYNAMIC_AGENT_H_ */
