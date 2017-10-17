/** \file rbdl_leo_walking_task.cpp
 * \brief RBDL file for C++ description of Leo task.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2016-06-30
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

#include <sys/stat.h>
#include <libgen.h>
#include <iomanip>

#include <rbdl/rbdl.h>
#include <rbdl/addons/luamodel/luamodel.h>

#include <grl/lua_utils.h>
#include <grl/environments/leo/rbdl_leo_walking_task.h>

#include <DynamixelSpecs.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoWalkingTask)

void LeoWalkingTask::request(ConfigurationRequest *config)
{
  Task::request(config);
  config->push_back(CRP("target_env", "environment", "Interaction environment", target_env_, true));
  config->push_back(CRP("timeout", "double.timeout", "Task timeout", timeout_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("randomize", "int.randomize", "Initialization from a random pose", randomize_, CRP::System, 0, 1));
  config->push_back(CRP("measurement_noise", "int.measurement_noise", "Adding measurement noise to observations", measurement_noise_, CRP::System, 0, 1));

}

void LeoWalkingTask::configure(Configuration &config)
{
  target_env_ = (Environment*)config["target_env"].ptr(); // Select a real enviromnent if needed
  timeout_ = config["timeout"];
  randomize_ = config["randomize"];
  measurement_noise_ = config["measurement_noise"];

  // Target observations: 2*target_dof + time
  std::vector<double> obs_min = {-1000, -1000, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI, -1000, -1000, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, 0};
  std::vector<double> obs_max = { 1000, 1000, M_PI,  M_PI, M_PI, M_PI, M_PI, M_PI, M_PI, 1000, 1000, 10*M_PI, 10*M_PI, 10*M_PI, 10*M_PI, 10*M_PI, 10*M_PI,  10*M_PI, 1};
  toVector(obs_min, target_obs_min_);
  toVector(obs_max, target_obs_max_);

  dof_ = 9;

  // Observations and actions exposed to an agent
  config.set("observation_dims", 2*dof_+1);
  Vector observation_min, observation_max;
  observation_min.resize(2*dof_+1);
  observation_max.resize(2*dof_+1);

  observation_min << target_obs_min_[rlwTorsoX], target_obs_min_[rlwTorsoZ], target_obs_min_[rlwTorsoAngle], target_obs_min_[rlwLeftHipAngle], target_obs_min_[rlwRightHipAngle], target_obs_min_[rlwLeftKneeAngle],
      target_obs_min_[rlwRightKneeAngle], target_obs_min_[rlwLeftAnkleAngle], target_obs_min_[rlwRightAnkleAngle], target_obs_min_[rlwTorsoXRate], target_obs_min_[rlwTorsoZRate], target_obs_min_[rlwTorsoAngleRate],
      target_obs_min_[rlwLeftHipAngleRate], target_obs_min_[rlwRightHipAngleRate], target_obs_min_[rlwLeftKneeAngleRate],
      target_obs_min_[rlwRightKneeAngleRate], target_obs_min_[rlwLeftAnkleAngleRate], target_obs_min_[rlwRightAnkleAngleRate], target_obs_min_[rlwTime];

  observation_max << target_obs_max_[rlwTorsoX], target_obs_max_[rlwTorsoZ], target_obs_max_[rlwTorsoAngle], target_obs_max_[rlwLeftHipAngle], target_obs_max_[rlwRightHipAngle], target_obs_max_[rlwLeftKneeAngle],
      target_obs_max_[rlwRightKneeAngle], target_obs_max_[rlwLeftAnkleAngle], target_obs_max_[rlwRightAnkleAngle], target_obs_max_[rlwTorsoXRate], target_obs_max_[rlwTorsoZRate], target_obs_max_[rlwTorsoAngleRate],
      target_obs_max_[rlwLeftHipAngleRate], target_obs_max_[rlwRightHipAngleRate], target_obs_max_[rlwLeftKneeAngleRate],
      target_obs_max_[rlwRightKneeAngleRate], target_obs_max_[rlwLeftAnkleAngleRate], target_obs_max_[rlwRightAnkleAngleRate], target_obs_max_[rlwTime];

  config.set("observation_min", observation_min);
  config.set("observation_max", observation_max);

  config.set("action_dims", dof_);
  config.set("action_min", ConstantVector(dof_, -LEO_MAX_DXL_VOLTAGE));
  config.set("action_max", ConstantVector(dof_, LEO_MAX_DXL_VOLTAGE));
  config.set("reward_min", VectorConstructor(-1000));
  config.set("reward_max", VectorConstructor( 1000));

  std::cout << "observation_min: " << config["observation_min"].v() << std::endl;
  std::cout << "observation_max: " << config["observation_max"].v() << std::endl;
  std::cout << "action_min: " << config["action_min"].v() << std::endl;
  std::cout << "action_max: " << config["action_max"].v() << std::endl;
}

void LeoWalkingTask::start(int test, Vector *state) const
{
  *state = ConstantVector(2*dof_+1, 0); // Same size for both tasts with FA and without

  if (target_env_)
  {
    // Obtain initial state from real Leo
    Observation obs;
    target_env_->start(0, &obs);
    *state << obs.v, VectorConstructor(0.0);
  }
  else
  {
    // Default initialization for walking pose
    *state <<
           0, 0, -0.101485, 0.100951, 0.819996, -0.00146549, -1.27, 4.11e-6, 2.26e-7,
           //0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0,
            0.0;  // rlwTime

    if ((randomize_) || (!test))
    {
      for (int ii=4; ii < dof_; ii+=2)
      {
        (*state)[ii] += RandGen::getUniform(-0.0872, 0.0872);
      }
      (*state)[rlwLeftKneeAngle] += RandGen::getUniform(-2*0.0872, 0);
      (*state)[rlwLeftHipAngle] += RandGen::getUniform(-0.0872, 0.0872);
      (*state)[rlwLeftAnkleAngle] +=RandGen::getUniform(-0.0872, 0.0872);
    }
  }

  CRAWL("Initial state: " << *state);
}

void LeoWalkingTask::observe(const Vector &state, Observation *obs, int *terminal) const
{
  grl_assert(state.size() == rlwStateDim);

  obs->v.resize(2*dof_);

  obs->v << state.head(2*dof_);

  //Adding measurement noise
  if (measurement_noise_)
  {
    for (int ij=2; ij<dof_; ++ij)
    {
      (obs->v)[ij] += RandGen::getUniform(-0.05,0.05);
      (obs->v)[ij+dof_] += RandGen::getUniform(-0.1,0.1);
    }
  }

  obs->absorbing = false;
  if ((timeout_> 0) && (state[rlwTime] >= timeout_))
    *terminal = 1;
  else if (isDoomedToFall(state))
  {
    obs->absorbing = false;
    *terminal = 2;
  }
  else
    *terminal = 0;
}

void LeoWalkingTask::evaluate(const Vector &state, const Action &action, const Vector &next, double *reward) const
{
  //State is previous state and next is the new state
  *reward = calculateReward(state, next); //Add the work with COM moving forward
  *reward += getEnergyUsage(state, next, action);
}

void LeoWalkingTask::report(std::ostream &os, const Vector &state) const
{
}

double LeoWalkingTask::calculateReward(const Vector &state, const Vector &next) const
{
  double reward = 0;
  double mRwDoomedToFall = -75;
  double mRwTime = -1.5;
  double mRwForward = 300;

// Time penalty
  reward += mRwTime;

  reward += mRwForward*(next[rlwComX] - state[rlwComX]);

//  // Negative reward for 'falling' (doomed to fall)
  if (isDoomedToFall(next))
  {
    reward += mRwDoomedToFall;
  }

  return reward;
}

bool LeoWalkingTask::isDoomedToFall(const Vector &state) const
{
  double torsoConstraint = 1; // 1
  double stanceConstraint = 0.36*M_PI; // 0.36*M_PI
  double torsoHeightConstraint = -0.15;

  if ((fabs(state[rlwTorsoAngle]) > torsoConstraint) || (fabs(state[rlwRightAnkleAngle]) > stanceConstraint) || (fabs(state[rlwLeftAnkleAngle]) > stanceConstraint)
      || (state[rlwTorsoZ] < torsoHeightConstraint) || (state[rlwRightKneeAngle] > 0) || (state[rlwLeftKneeAngle] > 0))
  {
    return true;
  }

  return false;
}

double LeoWalkingTask::getEnergyUsage(const Vector &state, const Vector &next, const Action &action) const
{
  double mRwEnergy = -2;
  double JointWork = 0;
  double mDesiredFrequency = 30;
  double I, U; // Electrical work: P = U*I

  for (int ii=3; ii<dof_; ii++)  //Start from 3 to ignore work by torso joint
  {
    // We take the joint velocity as the average of the previous and the current velocity measurement
    double omega = 0.5*(state[ii+dof_] + next[ii+dof_]);

    // We take the action that was executed the previous step.
    U = action.v[ii];
    I = (U - DXL_TORQUE_CONST*DXL_GEARBOX_RATIO*omega)/DXL_RESISTANCE;

  // Negative electrical work is not beneficial (no positive reward), but does not harm either.
    JointWork += std::max(0.0, U*I)/mDesiredFrequency;  // Divide power by frequency to get energy (work)
  }
  return mRwEnergy*JointWork;
}
