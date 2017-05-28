/** \file rbdl_leo.cpp
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
#include <grl/environments/leo/rbdl_leo_task.h>

#include <DynamixelSpecs.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoSquattingTask)
REGISTER_CONFIGURABLE(LeoWalkingTask)

void LeoSquattingTask::request(ConfigurationRequest *config)
{
  Task::request(config);
  config->push_back(CRP("target_env", "environment", "Interaction environment", target_env_, true));
  config->push_back(CRP("timeout", "double.timeout", "Task timeout", timeout_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("randomize", "int.randomize", "Initialization from a random pose", randomize_, CRP::System, 0, 1));
  config->push_back(CRP("weight_nmpc", "double.weight_nmpc", "Weight on the NMPC cost (excluding shaping)", weight_nmpc_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("weight_nmpc_aux", "double.weight_nmpc_aux", "Weight on the part of NMPC cost with auxilary", weight_nmpc_aux_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("weight_nmpc_qd", "double.weight_nmpc_qd", "Weight on the part of NMPC cost which penalizes large velocities", weight_nmpc_qd_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("weight_shaping", "double.weight_shaping", "Weight on the shaping cost", weight_shaping_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("power", "double.power", "Power of objective functions comprising cost", power_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("setpoint_reward", "int.setpoint_reward", "If zero, reward at setpoint is given for setpoint at time t, otherwise - at t+1", setpoint_reward_, CRP::System, 0, 1));
  config->push_back(CRP("continue_after_fall", "int.continue_after_fall", "Continue exectution of the environemnt even after a fall of Leo", continue_after_fall_, CRP::System, 0, 1));
  config->push_back(CRP("sub_sim_state", "signal/vector", "Subscriber to external sigma", sub_sim_state_, true));
  config->push_back(CRP("gamma", "Discount rate (used in shaping)", gamma_));
  config->push_back(CRP("fixed_arm", "int.fixed_arm", "Require fixed arm, fa option", fixed_arm_, CRP::System, 0, 1));
}

void LeoSquattingTask::configure(Configuration &config)
{
  target_env_ = (Environment*)config["target_env"].ptr(); // Select a real enviromnent if needed
  timeout_ = config["timeout"];
  randomize_ = config["randomize"];
  weight_nmpc_ = config["weight_nmpc"];
  weight_nmpc_aux_ = config["weight_nmpc_aux"];
  weight_nmpc_qd_ = config["weight_nmpc_qd"];
  weight_shaping_ = config["weight_shaping"];
  power_ = config["power"];
  setpoint_reward_ = config["setpoint_reward"];
  continue_after_fall_ = config["continue_after_fall"];
  sub_sim_state_ = (VectorSignal*)config["sub_sim_state"].ptr();
  gamma_ = config["gamma"];
  fixed_arm_ = config["fixed_arm"];

  // Target observations: 2*target_dof + time
  std::vector<double> obs_min = {-M_PI, -M_PI, -M_PI, -M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, 0};
  std::vector<double> obs_max = { M_PI,  M_PI,  M_PI,  M_PI,  10*M_PI,  10*M_PI,  10*M_PI,  10*M_PI, 1};
  toVector(obs_min, target_obs_min_);
  toVector(obs_max, target_obs_max_);

  dof_ = fixed_arm_ ? 3 : 4;

  // Observations and actions exposed to an agent
  config.set("observation_dims", 2*dof_+1);
  Vector observation_min, observation_max;
  observation_min.resize(2*dof_+1);
  observation_max.resize(2*dof_+1);
  if (fixed_arm_)
  {
    observation_min << target_obs_min_[rlsLeftAnkleAngle], target_obs_min_[rlsLeftKneeAngle], target_obs_min_[rlsLeftHipAngle],
        target_obs_min_[rlsLeftAnkleAngleRate], target_obs_min_[rlsLeftKneeAngleRate], target_obs_min_[rlsLeftHipAngleRate], target_obs_min_[rlsTime];
    observation_max << target_obs_max_[rlsLeftAnkleAngle], target_obs_max_[rlsLeftKneeAngle], target_obs_max_[rlsLeftHipAngle],
        target_obs_max_[rlsLeftAnkleAngleRate], target_obs_max_[rlsLeftKneeAngleRate], target_obs_max_[rlsLeftHipAngleRate], target_obs_max_[rlsTime];
  }
  else
  {
    observation_min << target_obs_min_[rlsLeftAnkleAngle], target_obs_min_[rlsLeftKneeAngle], target_obs_min_[rlsLeftHipAngle], target_obs_min_[rlsLeftArmAngle],
        target_obs_min_[rlsLeftAnkleAngleRate], target_obs_min_[rlsLeftKneeAngleRate], target_obs_min_[rlsLeftHipAngleRate], target_obs_min_[rlsLeftArmAngleRate], target_obs_min_[rlsTime];
    observation_max << target_obs_max_[rlsLeftAnkleAngle], target_obs_max_[rlsLeftKneeAngle], target_obs_max_[rlsLeftHipAngle],  target_obs_max_[rlsLeftArmAngle],
        target_obs_max_[rlsLeftAnkleAngleRate], target_obs_max_[rlsLeftKneeAngleRate], target_obs_max_[rlsLeftHipAngleRate], target_obs_max_[rlsLeftArmAngleRate], target_obs_max_[rlsTime];
  }

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

void LeoSquattingTask::start(int test, Vector *state) const
{
  *state = ConstantVector(2*(4)+1, 0); // Same size for both tasts with FA and without

  if (target_env_)
  {
    // Obtain initial state from real Leo
    Observation obs;
    target_env_->start(0, &obs);
    *state << obs.v, VectorConstructor(0.0);
  }
  else
  {
    // Default initialization in a sitted pose
    *state <<
           1.0586571916803691E+00,
          -2.1266836153365212E+00,
           1.2680264236561250E+00,
          -2.5999999999984957E-01,
          -0.0,
          -0.0,
          -0.0,
          -0.0,  // end of rlsDofDim
           0.0;  // rlsTime

    if (randomize_)
    {
      // sample angles
      const double upLegLength  = 0.1160;
      const double loLegLength  = 0.1085;
      double a, b, c, d, hh;
      do
      {
        a = RandGen::getUniform(-1.65,  1.48)*randomize_;
        b = RandGen::getUniform(-2.53,  0.00)*randomize_;
        c = RandGen::getUniform(-0.61,  2.53)*randomize_;
        d = RandGen::getUniform(-0.10,  0.10)*randomize_;
        hh = loLegLength*cos(a) + upLegLength*cos(a+b);
      }
      while (fabs(a + b + c) > 3.1415/2.0 || hh < 0.07);

      (*state)[rlsLeftAnkleAngle] = a;
      (*state)[rlsLeftKneeAngle] = b;
      (*state)[rlsLeftHipAngle] = c;
      (*state)[rlsLeftArmAngle] = d;

      TRACE("Hip height: " << hh);
    }
  }

  task_reward_ = 0;
  subtask_reward_ = 0;
  subtasks_rewards_.clear();
  CRAWL("Initial state: " << *state);
}

void LeoSquattingTask::observe(const Vector &state, Observation *obs, int *terminal) const
{
  grl_assert(state.size() == stsStateDim);

  obs->v.resize(2*dof_+1);
  if (fixed_arm_)
  {
    obs->v << state[rlsLeftAnkleAngle], state[rlsLeftKneeAngle], state[rlsLeftHipAngle],
              state[rlsLeftAnkleAngleRate], state[rlsLeftKneeAngleRate], state[rlsLeftHipAngleRate], state[rlsRefRootZ];
  }
  else
  {
    obs->v << state[rlsLeftAnkleAngle], state[rlsLeftKneeAngle], state[rlsLeftHipAngle], state[rlsLeftArmAngle],
              state[rlsLeftAnkleAngleRate], state[rlsLeftKneeAngleRate], state[rlsLeftHipAngleRate], state[rlsLeftArmAngleRate], state[rlsRefRootZ];
  }

  if ((timeout_> 0) && (state[rlsTime] >= timeout_))
    *terminal = 1;
  else if (failed(state))
    *terminal = 2;
  else
    *terminal = 0;
/*
  // debugging (until first switch)
  if (state[rlsRefRootZ] == 0.28)
  {
    TRACE("Terminate on first switch.");
    *terminal = 1;
  }
*/
}

void LeoSquattingTask::evaluate(const Vector &state, const Action &action, const Vector &next, double *reward) const
{
  grl_assert(state.size() == stsStateDim);
  grl_assert(action.size() == dof_);
  grl_assert(next.size() == stsStateDim);

  if (failed(next))
  {
    *reward = -100;
    task_reward_ += -100;
    subtask_reward_ += -100;
    return;
  }

  double cost_nmpc = 0, cost_nmpc_aux = 0, cost_nmpc_qd = 0;
  double refRootZ = setpoint_reward_ ? next[rlsRefRootZ] : state[rlsRefRootZ];

  // calculate support center from feet positions
  double suppport_center = 0.5 * (next[rlsLeftTipX] + next[rlsLeftHeelX]);

  // track: || root_z - h_ref ||_2^2
  cost_nmpc +=  pow(50.0 * fabs(next[rlsRootZ] - refRootZ), power_);

  // track: || com_x - support center_x ||_2^2
  cost_nmpc_aux +=  pow( 100.00 * fabs(next[rlsComX] - suppport_center), power_);

  //double velW = 10.0; // 10.0
  //cost +=  pow( velW * next[rlsComVelocityX], 2);
  //cost +=  pow( velW * next[rlsComVelocityZ], 2);

  //cost +=  pow( 100.00 * next[rlsAngularMomentumY], 2);

  // NOTE: sum of lower body angles is equal to angle between ground slope
  //       and torso. Minimizing deviation from zero keeps torso upright
  //       during motion execution.
  cost_nmpc_aux += pow(30.00 * fabs( next[rlsLeftAnkleAngle] + next[rlsLeftKneeAngle] + next[rlsLeftHipAngle] - (0.15) ), power_); // desired torso angle

  // regularize torso
  // is this a good way for torso? Results in a very high penalty, and very weird behaviour
  //cost += pow(60.00 * (next[rlsAnkleAngleRate] + next[rlsKneeAngleRate] + next[rlsHipAngleRate]), 2);

  // regularize: || qdot ||_2^2
  // res[res_cnt++] = 6.00 * sd[QDOTS["arm"]]; // arm
  double rateW = 5.0; // 6.0
  if (!fixed_arm_)
    cost_nmpc_qd += pow(rateW * fabs(next[rlsLeftArmAngleRate]), power_); // arm, added to cost only if nmpc adds it
  cost_nmpc_qd += pow(rateW * fabs(next[rlsLeftHipAngleRate]), power_); // hip_left
  cost_nmpc_qd += pow(rateW * fabs(next[rlsLeftKneeAngleRate]), power_); // knee_left
  cost_nmpc_qd += pow(rateW * fabs(next[rlsLeftAnkleAngleRate]), power_); // ankle_left

  // regularize: || u ||_2^2
  // res[res_cnt++] = 0.01 * u[TAUS["arm"]]; // arm

  TRACE(cost_nmpc);
  TRACE(cost_nmpc_aux);
  TRACE(cost_nmpc_qd);

  // reward is a negative of cost
  double immediate_reward = -weight_nmpc_*(cost_nmpc + weight_nmpc_aux_*(cost_nmpc_aux + weight_nmpc_qd_*cost_nmpc_qd));
  task_reward_ += immediate_reward;
  subtask_reward_ += immediate_reward;

  if (sub_sim_state_)
  {
    // use reward based on the simulated state if requested
    Vector sim_state = sub_sim_state_->get();
    Vector x = next.block(0, 0, 1, dof_) - sim_state.block(0, 0, 1, dof_);
    *reward = - sqrt(x.cwiseProduct(x).sum());
  }
  else
    *reward = immediate_reward;

  TRACE(*reward);

  // adding shaping
  if (weight_shaping_ != 0.0)
  {
    double F0 = -fabs(state[rlsRootZ] - refRootZ); // distance to setpoint at time (t)
    double F1 = -fabs(next [rlsRootZ] - refRootZ); // distance to setpoint at time (t+1)
    double shaping = gamma_*F1 - F0; // positive reward for getting closer to the setpoint
    TRACE(state[rlsRootZ] << ", " << next[rlsRootZ] << " -> " << refRootZ);
    TRACE(F1 << " - " << F0 << " = " << shaping);
    *reward += weight_shaping_*shaping;
    TRACE(*reward);
  }

  // record rewards when switching happens
  if (state[rlsRefRootZ] != next[rlsRefRootZ])
  {
    subtasks_rewards_.push_back(subtask_reward_);
    subtask_reward_ = 0;
  }
}

int LeoSquattingTask::failed(const Vector &state) const
{
  if (std::isnan(state[rlsRootZ]))
    ERROR("NaN value of root, try to reduce integration period to cope with this.");

  if (continue_after_fall_)
    return 0;

  double torsoAngle = state[rlsLeftAnkleAngle] + state[rlsLeftKneeAngle] + state[rlsLeftHipAngle];
  if (fabs(torsoAngle) > 1.0) // > 57 deg
  {
    TRACE("Terminate on large torso.");
    return 1;
  }
  //
  else if (state[rlsLeftAnkleAngleRate] < target_obs_min_[rlsLeftAnkleAngleRate])
  {
    TRACE("Terminate on large negative ankle angle rate.");
    return 1;
  }
  else if (state[rlsLeftAnkleAngleRate] > target_obs_max_[rlsLeftAnkleAngleRate])
  {
    TRACE("Terminate on large positive ankle angle rate.");
    return 1;
  }
  //
  else if (state[rlsLeftKneeAngleRate]  < target_obs_min_[rlsLeftKneeAngleRate])
  {
    TRACE("Terminate on large negative knee angle rate.");
    return 1;
  }
  else if (state[rlsLeftKneeAngleRate]  > target_obs_max_[rlsLeftKneeAngleRate])
  {
    TRACE("Terminate on large positive knee angle rate.");
    return 1;
  }
  //
  else if (state[rlsLeftHipAngleRate]   < target_obs_min_[rlsLeftHipAngleRate])
  {
    TRACE("Terminate on large negative hip angle rate.");
    return 1;
  }
  else if (state[rlsLeftHipAngleRate]   > target_obs_max_[rlsLeftHipAngleRate])
  {
    TRACE("Terminate on large positive hip angle rate.");
    return 1;
  }
  //
  else if (state[rlsLeftArmAngleRate]   < target_obs_min_[rlsLeftArmAngleRate])
  {
    TRACE("Terminate on large negative ankle angle rate.");
    return 1;
  }
  else if (state[rlsLeftArmAngleRate]   > target_obs_max_[rlsLeftArmAngleRate])
  {
    TRACE("Terminate on large positive ankle angle rate.");
    return 1;
  }
  //
  else if (state[rlsRootZ] < 0)
  {
    TRACE("Terminate on root point going under the ground.");
    return 1;
  }
  else
    return 0;
}

void LeoSquattingTask::report(std::ostream &os, const Vector &state) const
{
  const int pw = 15;
  std::stringstream progressString;
  progressString << std::fixed << std::setprecision(3) << std::right;
  progressString << std::setw(pw) << state[rlsRootZ];
  progressString << std::setw(pw) << state[stsSquats];
  progressString << std::setw(pw) << task_reward_;

  // append cumulative reward in case of timeout termination
  if (subtask_reward_ != 0)
    subtasks_rewards_.push_back(subtask_reward_);

  int max_size = 6;
  int size = std::min(max_size, static_cast<int>(subtasks_rewards_.size()));

  for (int i = 0; i < size; i++)
    progressString << std::setw(pw) << subtasks_rewards_[i];

  for (int i = size; i < max_size; i++)
    progressString << std::setw(pw) << std::numeric_limits<double>::quiet_NaN();

  os << progressString.str();
}

///////////////////////////////////////////////////////////

void LeoWalkingTask::request(ConfigurationRequest *config)
{
  Task::request(config);
  config->push_back(CRP("target_env", "environment", "Interaction environment", target_env_, true));
  config->push_back(CRP("timeout", "double.timeout", "Task timeout", timeout_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("randomize", "int.randomize", "Initialization from a random pose", randomize_, CRP::System, 0, 1));
  config->push_back(CRP("sub_sim_state", "signal/vector", "Subscriber to external sigma", sub_sim_state_, true));
}

void LeoWalkingTask::configure(Configuration &config)
{
  target_env_ = (Environment*)config["target_env"].ptr(); // Select a real enviromnent if needed
  timeout_ = config["timeout"];
  randomize_ = config["randomize"];
  sub_sim_state_ = (VectorSignal*)config["sub_sim_state"].ptr();

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

  observation_min << target_obs_min_[rlsTorsoX], target_obs_min_[rlsTorsoZ], target_obs_min_[rlsTorsoAngle], target_obs_min_[rlsLeftHipAngle], target_obs_min_[rlsRightHipAngle], target_obs_min_[rlsLeftKneeAngle],
      target_obs_min_[rlsRightKneeAngle], target_obs_min_[rlsLeftAnkleAngle], target_obs_min_[rlsRightAnkleAngle], target_obs_min_[rlsTorsoXRate], target_obs_min_[rlsTorsoZRate], target_obs_min_[rlsTorsoAngleRate],
      target_obs_min_[rlsLeftHipAngleRate], target_obs_min_[rlsRightHipAngleRate], target_obs_min_[rlsLeftKneeAngleRate],
      target_obs_min_[rlsRightKneeAngleRate], target_obs_min_[rlsLeftAnkleAngleRate], target_obs_min_[rlsRightAnkleAngleRate], target_obs_min_[rlsTime];

  observation_max << target_obs_max_[rlsTorsoX], target_obs_max_[rlsTorsoZ], target_obs_max_[rlsTorsoAngle], target_obs_max_[rlsLeftHipAngle], target_obs_max_[rlsRightHipAngle], target_obs_max_[rlsLeftKneeAngle],
      target_obs_max_[rlsRightKneeAngle], target_obs_max_[rlsLeftAnkleAngle], target_obs_max_[rlsRightAnkleAngle], target_obs_max_[rlsTorsoXRate], target_obs_max_[rlsTorsoZRate], target_obs_max_[rlsTorsoAngleRate],
      target_obs_max_[rlsLeftHipAngleRate], target_obs_max_[rlsRightHipAngleRate], target_obs_max_[rlsLeftKneeAngleRate],
      target_obs_max_[rlsRightKneeAngleRate], target_obs_max_[rlsLeftAnkleAngleRate], target_obs_max_[rlsRightAnkleAngleRate], target_obs_max_[rlsTime];

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
  int randomize;

  if (test)
    randomize = 0;
  else
    randomize = randomize_;

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
            0.0;  // rlsTime

    if (randomize)
    {
      for (int ii=4; ii < dof_; ii+=2)
      {
        (*state)[ii] += RandGen::getUniform(-0.0872, 0.0872);
        (*state)[dof_+ii] += RandGen::getUniform(-0.1,0.1);
      }
     }
  }

  CRAWL("Initial state: " << *state);
}

void LeoWalkingTask::observe(const Vector &state, Observation *obs, int *terminal) const
{
  grl_assert(state.size() == rlsStateDim);

  obs->v.resize(2*dof_);
  obs->v << state.head(2*dof_);

  if ((timeout_> 0) && (state[rlsTime] >= timeout_))
    *terminal = 1;
  else if (isDoomedToFall(state))
    *terminal = 2;
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

  reward += mRwForward*(next[rlsComX] - state[rlsComX]);

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
  double torsoHeightConstraint = -0.13;

  if ((fabs(state[rlsTorsoAngle]) > torsoConstraint) || (fabs(state[rlsRightAnkleAngle]) > stanceConstraint) || (fabs(state[rlsLeftAnkleAngle]) > stanceConstraint) || (fabs(state[rlsTorsoZ]) < torsoHeightConstraint))
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
