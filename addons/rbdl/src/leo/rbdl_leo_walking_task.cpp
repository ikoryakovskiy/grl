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
REGISTER_CONFIGURABLE(LeoBalancingTask)
REGISTER_CONFIGURABLE(LeoCrouchingTask)

void LeoWalkingTask::request(ConfigurationRequest *config)
{
  Task::request(config);
  config->push_back(CRP("target_env", "environment", "Interaction environment", target_env_, true));
  config->push_back(CRP("timeout", "double.timeout", "Task timeout", timeout_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("randomize", "double.randomize", "Random pose within +/- randomize degrees", randomize_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("measurement_noise", "int.measurement_noise", "Adding measurement noise to observations", measurement_noise_, CRP::System, 0, 1));
  config->push_back(CRP("rwForward", "double", "Task timeout", rwForward_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("knee_mode", "Select the mode knee constrain is handled", knee_mode_, CRP::Configuration, {"fail_and_restart", "punish_and_continue", "continue"}));
}

void LeoWalkingTask::configure(Configuration &config)
{
  target_env_ = (Environment*)config["target_env"].ptr(); // Select a real enviromnent if needed
  timeout_ = config["timeout"];
  randomize_ = config["randomize"];
  measurement_noise_ = config["measurement_noise"];
  knee_mode_ = config["knee_mode"].str();
  rwForward_ = config["rwForward"];

  // Target observations: 2*target_dof + time
  std::vector<double> obs_min = {-1000, -1000, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI, -1000, -1000, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI};
  std::vector<double> obs_max = { 1000, 1000, M_PI,  M_PI, M_PI, M_PI, M_PI, M_PI, M_PI, 1000, 1000, 10*M_PI, 10*M_PI, 10*M_PI, 10*M_PI, 10*M_PI, 10*M_PI,  10*M_PI};
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
      target_obs_min_[rlwRightKneeAngleRate], target_obs_min_[rlwLeftAnkleAngleRate], target_obs_min_[rlwRightAnkleAngleRate],
      -1000; // Forward promototion

  observation_max << target_obs_max_[rlwTorsoX], target_obs_max_[rlwTorsoZ], target_obs_max_[rlwTorsoAngle], target_obs_max_[rlwLeftHipAngle], target_obs_max_[rlwRightHipAngle], target_obs_max_[rlwLeftKneeAngle],
      target_obs_max_[rlwRightKneeAngle], target_obs_max_[rlwLeftAnkleAngle], target_obs_max_[rlwRightAnkleAngle], target_obs_max_[rlwTorsoXRate], target_obs_max_[rlwTorsoZRate], target_obs_max_[rlwTorsoAngleRate],
      target_obs_max_[rlwLeftHipAngleRate], target_obs_max_[rlwRightHipAngleRate], target_obs_max_[rlwLeftKneeAngleRate],
      target_obs_max_[rlwRightKneeAngleRate], target_obs_max_[rlwLeftAnkleAngleRate], target_obs_max_[rlwRightAnkleAngleRate],
      1000; // Forward promototion

  config.set("observation_min", observation_min);
  config.set("observation_max", observation_max);

  config.set("action_dims", dof_);
  config.set("action_min", ConstantVector(dof_, -LEO_MAX_DXL_VOLTAGE));
  config.set("action_max", ConstantVector(dof_, LEO_MAX_DXL_VOLTAGE));
  config.set("reward_min", -1000);
  config.set("reward_max",  1000);

  std::cout << "observation_min: " << config["observation_min"].v() << std::endl;
  std::cout << "observation_max: " << config["observation_max"].v() << std::endl;
  std::cout << "action_min: " << config["action_min"].v() << std::endl;
  std::cout << "action_max: " << config["action_max"].v() << std::endl;
}

void LeoWalkingTask::reconfigure(const Configuration &config)
{
  if (config.has("action"))
  {
    if (config["action"].str() == "update_rwForward")
    {
      rwForward_ = config["rwForward"];
      INFO("New forward reward weighting is " << rwForward_);
    }
    if (config["action"].str() == "update_rwTime")
    {
      rwTime_ = config["rwTime"];
      INFO("New time reward weighting is " << rwTime_);
    }
    if (config["action"].str() == "update_rwWork")
    {
      rwWork_ = config["rwWork"];
      INFO("New work reward weighting is " << rwWork_);
    }
  }
}

void LeoWalkingTask::initLeo(int test, Vector *state, int sym_rand) const
{
  test_ = test;

  if (target_env_)
  {
    // Obtain initial state from real Leo
    Observation obs;
    target_env_->start(0, &obs);
    *state << obs.v, VectorConstructor(0.0);
  }
  else if (!test)
  {
    double r = randomize_ * 3.1415/180.0;

    if (!sym_rand)
    {
      for (int ii = rlwLeftHipAngle; ii <= rlwRightAnkleAngle; ii++)
        (*state)[ii] += r * (2*drand48()-1);
    }
    else
    {
      for (int ii = rlwLeftHipAngle; ii <= rlwLeftAnkleAngle; ii+=2)
      {
        double delta = r * (2*drand48()-1);
        (*state)[ii] += delta;
        (*state)[ii+1] += delta;
      }
    }

    (*state)[rlwLeftKneeAngle] = fmin((*state)[rlwLeftKneeAngle], -0.02);
    (*state)[rlwLeftKneeAngle] = fmin((*state)[rlwRightKneeAngle], -0.02);
  }

  trialWork_ = 0;
  TRACE("Initial state: " << *state);
}

void LeoWalkingTask::start(int test, Vector *state) const
{
  // Default initialization of the walking pose
  *state = ConstantVector(2*dof_+1, 0);
  *state << 0, 0, 0, 0, 0.82, 0, -1.27, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0;  // + rlwTime
  initLeo(test, state);
}

void LeoWalkingTask::observe(const Vector &state, Observation *obs, int *terminal) const
{
  grl_assert(state.size() == rlwStateDim);

  obs->v.resize(2*dof_ + 1); // one for measurment of forward displacement of Leo
  obs->v << state.head(2*dof_), state[rlwComX] - state[rlwPrevComX];

  // Adding measurement noise
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
  else if (isDoomedToFall(state) || (isKneeBroken(state) && knee_mode_ == "fail_and_restart"))
  {
    if (isDoomedToFall(state) && !test_)
      falls_++; // increase number of falls only in learning trials
    obs->absorbing = false;
    *terminal = 2;
  }
  else
    *terminal = 0;
}

void LeoWalkingTask::evaluate(const Vector &state, const Action &action, const Vector &next, double *reward) const
{
  //grl_assert(state[rlwComX] == next[rlwPrevComX]-1);
  //throw Exception("LeoWalkingTask::evaluate Exception");
  if (state[rlwComX] != next[rlwPrevComX])
    ERROR("LeoWalkingTask::evaluate COM error: " << state[rlwComX] << ", " << next[rlwPrevComX]);

  // State is previous state and next is the new state
  double stepWork = getMotorWork(state, next, action);
  trialWork_ += stepWork;

  *reward = rwWork_*stepWork;
  *reward += calculateReward(state, next);
}

double LeoWalkingTask::calculateReward(const Vector &state, const Vector &next) const
{
  double reward = 0;

  // Time penalty
  reward += rwTime_;

  // Forward promotion
  reward += rwForward_*(next[rlwComX] - state[rlwComX]);

  // Negative reward
  if (isDoomedToFall(next) || (isKneeBroken(next) && knee_mode_ == "fail_and_restart"))
    reward += rwFail_;       // when failing the task due to 'fall' or 'broken knee'
  else if (!isDoomedToFall(next) && isKneeBroken(next) && knee_mode_ == "punish_and_continue")
    reward += rwBrokenKnee_; // when 'broken knee' is allowed but not preferred

  return reward;
}

bool LeoWalkingTask::isDoomedToFall(const Vector &state) const
{
  double torsoConstraint = 1;
  double anklesConstraint = 0.36*M_PI;
  double torsoHeightConstraint = -0.15;

  if ((fabs(state[rlwTorsoAngle]) > torsoConstraint) ||
      (fabs(state[rlwRightAnkleAngle]) > anklesConstraint) || (fabs(state[rlwLeftAnkleAngle]) > anklesConstraint) ||
      (state[rlwTorsoZ] < torsoHeightConstraint))
    return true;

  return false;
}

bool LeoWalkingTask::isKneeBroken(const Vector &state) const
{
  if ((state[rlwRightKneeAngle] > 0) || (state[rlwLeftKneeAngle] > 0))
    return true;
  return false;
}

double LeoWalkingTask::getMotorWork(const Vector &state, const Vector &next, const Action &action) const
{
  double motorWork = 0;
  double desiredFrequency = 30;
  double I, U; // Electrical work: P = U*I

  for (int ii=3; ii<dof_; ii++)  //Start from 3 to ignore work by torso joint
  {
    // We take the joint velocity as the average of the previous and the current velocity measurement
    double omega = 0.5*(state[ii+dof_] + next[ii+dof_]);

    // We take the action that was executed the previous step.
    U = action.v[ii];
    I = (U - DXL_TORQUE_CONST*DXL_GEARBOX_RATIO*omega)/DXL_RESISTANCE;

    // Negative electrical work is not beneficial (no positive reward), but does not harm either.
    motorWork += std::max(0.0, U*I)/desiredFrequency;  // Divide power by frequency to get energy (work)
  }
  return motorWork;
}

void LeoWalkingTask::report(std::ostream &os, const Vector &state) const
{
  const int pw = 15;
  std::stringstream progressString;

  progressString << std::fixed << std::setprecision(5) << std::right;

  // Number of cumulative falls since the birth of the agent
  progressString << std::setw(pw) << falls_;

  // Walked distance
  progressString << std::setw(pw) << state[rlwTorsoX];

  // Speed
  progressString << std::setw(pw) << state[rlwTorsoX]/state[rlwTime];

  // Energy usage
  progressString << std::setw(pw) << trialWork_;

  // Energy per traveled meter
  if (state[rlwTorsoX] > 0.001)
    progressString << std::setw(pw) << trialWork_/state[rlwTorsoX];
  else
    progressString << std::setw(pw) << 0.0;

  // Print task name and parameters
  progressString << std::setw(30) << d_type();
  progressString << std::setw(pw) << rwForward_;

  os << progressString.str();
}

////////////////////////////////////////////////////////////////////////////////////////////

void LeoBalancingTask::start(int test, Vector *state) const
{
  // Default initialization of the balancing pose
  *state = ConstantVector(2*dof_+1, 0);
  *state << 0, 0, 0, 0, 0, -0.02, -0.02, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0,
            0;  // + rlwTime

  initLeo(test, state);
}

double LeoBalancingTask::calculateReward(const Vector &state, const Vector &next) const
{
  double reward = 0;

  // Negative reward
  if (isDoomedToFall(next) || (isKneeBroken(next) && knee_mode_ == "fail_and_restart"))
    reward += rwFail_;       // when failing the task due to 'fall' or 'broken knee'
  else if (!isDoomedToFall(next) && isKneeBroken(next) && knee_mode_ == "punish_and_continue")
    reward += rwBrokenKnee_; // when 'broken knee' is allowed but not preferred

  return reward;
}

////////////////////////////////////////////////////////////////////////////////////////////

void LeoCrouchingTask::start(int test, Vector *state) const
{
  // Default initialization of the balancing pose
  *state = ConstantVector(2*dof_+1, 0);
  *state <<
      0, 0, 0,
      1.0586571916803691E+00,
      1.0586571916803691E+00,
     -2.1266836153365212E+00,
     -2.1266836153365212E+00,
      1.0680264236561250E+00, // ideally, flat feet on the ground
      1.0680264236561250E+00, // ideally, flat feet on the ground
      0, 0, 0,
      0, 0, 0, 0, 0, 0,
      0;  // + rlwTime
  initLeo(0, state, 1);
}

bool LeoCrouchingTask::isDoomedToFall(const Vector &state) const
{
  double torsoConstraint = 1;
  double anklesConstraint = 0.5*M_PI;   // allow ankles a greater bend
  double torsoHeightConstraint = -0.05; // torso cannot move much more down compared to when Leo is straight

  if ((fabs(state[rlwTorsoAngle]) > torsoConstraint) ||
      (fabs(state[rlwRightAnkleAngle]) > anklesConstraint) || (fabs(state[rlwLeftAnkleAngle]) > anklesConstraint) ||
      (state[rlwTorsoZ] < torsoHeightConstraint))
    return true;

  return false;
}
