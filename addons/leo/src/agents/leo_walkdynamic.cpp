/** \file leo_walkdynamic.cpp
 * \brief Leo agent source file to walk dynamically (time-based).
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

#include <leo.h>
#include <grl/agents/leo_walkdynamic.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoWalkdynamicAgent)

void LeoWalkdynamicAgent::request(ConfigurationRequest *config)
{
  LeoBaseAgent::request(config);
}

void LeoWalkdynamicAgent::configure(Configuration &config)
{
  LeoBaseAgent::configure(config);
}

void LeoWalkdynamicAgent::reconfigure(const Configuration &config)
{
}

void LeoWalkdynamicAgent::start(const Observation &obs, Action *action)
{
  time_ = 0;
  mSwingStartTime = -mParamEarlySwingTime; // starting time is important, we start after early swing (in the middle!)
  walk(time_, obs.v, &(action->v));
}

void LeoWalkdynamicAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  time_ += tau;
  walk(time_, obs.v, &(action->v));
  action->type = atGreedy;
}

void LeoWalkdynamicAgent::end(double tau, const Observation &obs, double reward)
{
}

void LeoWalkdynamicAgent::walk(double time, const Vector &obs, Vector *action)
{
  // ******************* Walk! ******************* //
  int touchDown, groundContact, leftIsStance;
  unpack_ic(&touchDown, &groundContact, &leftIsStance);

  if (touchDown)
    mSwingStartTime  = time;

  // Determine the right joint indices
  int hipStance, hipSwing, kneeStance, kneeSwing, ankleStance, ankleSwing;
  if (leftIsStance)
  {
    hipStance         = ljHipLeft;
    hipSwing          = ljHipRight;
    kneeStance        = ljKneeLeft;
    kneeSwing         = ljKneeRight;
    ankleStance       = ljAnkleLeft;
    ankleSwing        = ljAnkleRight;
  }
  else
  {
    hipSwing          = ljHipLeft;
    hipStance         = ljHipRight;
    kneeSwing         = ljKneeLeft;
    kneeStance        = ljKneeRight;
    ankleSwing        = ljAnkleLeft;
    ankleStance       = ljAnkleRight;
  }

  // Determine swing phase type
  if ((time - mSwingStartTime) < mParamEarlySwingTime)
    mEarlySwing = true;
  else
    mEarlySwing = false;

  // Determine appropriate torques

  // Ankles and knees
  double ankleSwingTorque     = 0.0;
  double kneeSwingTorque      = 0.0;
  double kneeStanceTorque     = 4.0*(0.0 - obs[kneeStance]);
  double swingFootFloorAngle  = obs[ljTorso] + obs[hipSwing] + obs[kneeSwing] + obs[ankleSwing];
  double stanceFootFloorAngle = obs[ljTorso] + obs[hipStance] + obs[kneeStance] + obs[ankleStance];
  if (mEarlySwing)
  {
    // Early swing
    // Ankle pushoff
    ankleSwingTorque  = 5.0*(mParamAnklePushoffAngle - obs[ankleSwing]);
    // Knee
    kneeSwingTorque   = -3.5;
  }
  else
  {
    // Late swing
    ankleSwingTorque  = 14.0*(0.10 - swingFootFloorAngle);
    kneeSwingTorque   = 15.0*(0.0 - obs[kneeSwing]);
  }

  double ankleStanceTorque  = 5.0*(mParamAnkleStanceAngle - stanceFootFloorAngle);
  ankleStanceTorque += 3.0*(0.04 - obs[ankleStance]);

  double transFactor  = std::min((double)(time - mSwingStartTime)/(0.13), 1.0);

  // Hip: control the inter hip angle to be 0.62 rad ideally
  double interHipAngleTorque       = 8.0*(mParamInterHipAngle - (obs[hipSwing] - obs[hipStance]));

  double hipStanceTorque      = transFactor*mParamTransTorqueFactor*interHipAngleTorque;
  double hipSwingTorque       = transFactor*interHipAngleTorque;

  // Torque to keep the upper body up right
  double stanceTorque   = 0.0;
  stanceTorque          = (-14.0)*(mParamTorsoAngle - obs[ljTorso]);
  hipStanceTorque       += stanceTorque;

  // Control arm to a certain angle: protect the hand from hitting the floor
  double shoulderTorque = 0;
  double safeShoulderAngle;
  if (isShoulderAngleSafe(obs, leftIsStance, &safeShoulderAngle))
    // Use low gain to maintain default shoulder angle
    shoulderTorque  = 3.5*(safeShoulderAngle - obs[ljShoulder]);
  else
    // Use higher gain to be safe in time
    shoulderTorque  = 14.0*(safeShoulderAngle - obs[ljShoulder]);

  // Set joint voltages, based on torques
  double torqueToVolt = XM430_VS_RX28_COEFF*14.0/3.3;

  (*action)[ankleStance] = torqueToVolt*ankleStanceTorque;
  (*action)[ankleSwing]  = torqueToVolt*ankleSwingTorque;
  (*action)[kneeStance]  = torqueToVolt*kneeStanceTorque;
  (*action)[kneeSwing]   = torqueToVolt*kneeSwingTorque;
  (*action)[hipStance]   = torqueToVolt*hipStanceTorque;
  (*action)[hipSwing]    = torqueToVolt*hipSwingTorque;
  (*action)[ljShoulder]  = torqueToVolt*shoulderTorque;

  for (int i = 0; i < ljNumDynamixels; i++)
    (*action)[i] = fmin(action_max_[i], fmax((*action)[i], action_min_[i]));
}

bool LeoWalkdynamicAgent::isShoulderAngleSafe(Vector obs, int leftIsStance, double* safeShoulderAngle)
{
  // Calculate hand height (not exact but estimate):
  // - assume that the stance leg is on the floor
  // - assume that the stance leg is straight

  // Some properties of the robot that we assume
  double footHeight           = 0.05;  // Distance from floor to ankle joint
  double legLength            = 0.22;
  double torsoLength          = 0.22;
  double armLength            = 0.32;
  double shoulderNeutralAngle =-0.40;
  double safeHandHeight       = 0.14;

  int hipStance;
  if (leftIsStance)
  {
    hipStance = ljHipLeft;
  }
  else
  {
    hipStance = ljHipRight;
  }
  double stanceLegAngle  = obs[ljTorso] + obs[hipStance];
  double armAngle      = -shoulderNeutralAngle + obs[ljTorso] + obs[ljShoulder];

  double shoulderHeight  = footHeight + legLength*cos(stanceLegAngle) + torsoLength*cos(obs[ljTorso]);
  double handHeight    = shoulderHeight - armLength*cos(armAngle);

  // Calculate desired shoulder angle
  if (safeShoulderAngle != NULL)
  {
    double cosangle = (shoulderHeight - safeHandHeight)/armLength;
    *safeShoulderAngle = shoulderNeutralAngle - acos(std::max(std::min(cosangle, 1.0), -1.0)) - obs[ljTorso];
  }

  if (handHeight > safeHandHeight)
    return true;
  else
    return false;
}
