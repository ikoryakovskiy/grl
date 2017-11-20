/** \file rbdl_leo_squatting_task.cpp
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
#include <grl/environments/leo/rbdl_leo_squatting_task.h>

#include <DynamixelSpecs.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoSquattingTask)

void LeoSquattingTask::request(ConfigurationRequest *config)
{
  Task::request(config);
  config->push_back(CRP("target_env", "environment", "Interaction environment", target_env_, true));
  config->push_back(CRP("timeout", "double.timeout", "Task timeout", timeout_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("randomize", "double.randomize", "Initialization from a random pose", randomize_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("weight_nmpc", "double.weight_nmpc", "Weight on the NMPC cost (excluding shaping)", weight_nmpc_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("weight_nmpc_aux", "double.weight_nmpc_aux", "Weight on the part of NMPC cost with auxilary", weight_nmpc_aux_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("weight_nmpc_qd", "double.weight_nmpc_qd", "Weight on the part of NMPC cost which penalizes large velocities", weight_nmpc_qd_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("weight_shaping", "double.weight_shaping", "Weight on the shaping cost", weight_shaping_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("power", "double.power", "Power of objective functions comprising cost", power_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("use_mef", "int.use_mef", "Use MEF instead of NMPC cost", use_mef_, CRP::System, 0, 1));
  config->push_back(CRP("setpoint_reward", "int.setpoint_reward", "If zero, reward at setpoint is given for setpoint at time t, otherwise - at t+1", setpoint_reward_, CRP::System, 0, 1));
  config->push_back(CRP("continue_after_fall", "int.continue_after_fall", "Continue exectution of the environemnt even after a fall of Leo", continue_after_fall_, CRP::System, 0, 1));
  config->push_back(CRP("gamma", "Discount rate (used in shaping)", gamma_));
  config->push_back(CRP("fixed_arm", "int.fixed_arm", "Require fixed arm, fa option", fixed_arm_, CRP::System, 0, 1));
  config->push_back(CRP("lower_height", "double.lower_height", "Lower bound of root height to switch direction", lower_height_, CRP::Configuration, 0.0, DBL_MAX));
  config->push_back(CRP("upper_height", "double.upper_height", "Upper bound of root height to switch direction", upper_height_, CRP::Configuration, 0.0, DBL_MAX));
  config->push_back(CRP("friction_compensation", "int.fixed_arm", "Require friction compensation", friction_compensation_, CRP::System, 0, 1));
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
  use_mef_ = config["use_mef"];
  power_ = config["power"];
  setpoint_reward_ = config["setpoint_reward"];
  continue_after_fall_ = config["continue_after_fall"];
  gamma_ = config["gamma"];
  fixed_arm_ = config["fixed_arm"];
  friction_compensation_ = config["friction_compensation"];

  // Target observations: 2*target_dof + time
  std::vector<double> obs_min = {-M_PI, -M_PI, -M_PI, -M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, 0, 10};
  std::vector<double> obs_max = { M_PI,  M_PI,  M_PI,  M_PI,  10*M_PI,  10*M_PI,  10*M_PI,  10*M_PI, 1, 80};
  toVector(obs_min, target_obs_min_);
  toVector(obs_max, target_obs_max_);

  dof_ = fixed_arm_ ? 3 : 4;

  // Observations and actions exposed to an agent
  config.set("observation_dims", 2*dof_+2);
  Vector observation_min, observation_max;
  observation_min.resize(2*dof_+2);
  observation_max.resize(2*dof_+2);
  if (fixed_arm_)
  {
    observation_min << target_obs_min_[rlsAnkleAngle], target_obs_min_[rlsKneeAngle], target_obs_min_[rlsHipAngle],
        target_obs_min_[rlsAnkleAngleRate], target_obs_min_[rlsKneeAngleRate], target_obs_min_[rlsHipAngleRate],
        target_obs_min_[rlsTime], target_obs_min_[rlsTemperature];
    observation_max << target_obs_max_[rlsAnkleAngle], target_obs_max_[rlsKneeAngle], target_obs_max_[rlsHipAngle],
        target_obs_max_[rlsAnkleAngleRate], target_obs_max_[rlsKneeAngleRate], target_obs_max_[rlsHipAngleRate],
        target_obs_max_[rlsTime], target_obs_max_[rlsTemperature];
  }
  else
  {
    observation_min << target_obs_min_[rlsAnkleAngle], target_obs_min_[rlsKneeAngle], target_obs_min_[rlsHipAngle], target_obs_min_[rlsArmAngle],
        target_obs_min_[rlsAnkleAngleRate], target_obs_min_[rlsKneeAngleRate], target_obs_min_[rlsHipAngleRate], target_obs_min_[rlsArmAngleRate],
        target_obs_min_[rlsTime], target_obs_min_[rlsTemperature];
    observation_max << target_obs_max_[rlsAnkleAngle], target_obs_max_[rlsKneeAngle], target_obs_max_[rlsHipAngle],  target_obs_max_[rlsArmAngle],
        target_obs_max_[rlsAnkleAngleRate], target_obs_max_[rlsKneeAngleRate], target_obs_max_[rlsHipAngleRate], target_obs_max_[rlsArmAngleRate],
        target_obs_max_[rlsTime], target_obs_max_[rlsTemperature];
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

void LeoSquattingTask::reconfigure(const Configuration &config)
{
  if (config.has("action") && config["action"].str() == "statclr")
  {
    task_reward_ = 0;
    subtask_reward_ = 0;
    subtasks_rewards_.clear();
  }
}

void LeoSquattingTask::start(int test, Vector *state) const
{
  *state = ConstantVector(4*2+3, 0); // Same size for both tasks with FA and without
  test_ = test;

  if (target_env_)
  {
    // Obtain initial state from real Leo
    Observation obs;
    target_env_->start(0, &obs); // 4 angles, 4 velocities, 1 rlsTemperature
    *state << obs.v.head(4*2), VectorConstructor(0.0), obs.v.tail(1), VectorConstructor(upper_height_); // 4 angles, 4 velocities, 1 rlsTime, 1 rlsTemperature
  }
  else
  {
    Rand rand;
    int low_start;

    if (test == 0)
      low_start = rand.getInteger(2); // testing: random setpoint
    else
      low_start = test % 2; // learning: setpoint selection is determined by test parameter value (test = 1 => low start)

    if (low_start)
    {
      // initialization in a sitted pose
      *state <<
             1.0586571916803691E+00,
            -2.1266836153365212E+00,
             1.2680264236561250E+00,
            -2.5999999999984957E-01,
            -0.0,
            -0.0,
            -0.0,
            -0.0,           // end of rlsDofDim
             0.0,           // rlsTime
            25.0,           // rlsTemperature
            upper_height_;  // rlsRefRootZ
    }
    else
    {
      // initialization in a standing pose
      *state <<
             0.458644,
            -1.19249,
             1.03384,
            -0.191774,
            -0.0,
            -0.0,
            -0.0,
            -0.0,           // end of rlsDofDim
             0.0,           // rlsTime
            25.0,           // rlsTemperature
            lower_height_;  // rlsRefRootZ
    }

    if (test == 0 && randomize_)
    {
      // sample angles
      const double upLegLength  = 0.1160;
      const double loLegLength  = 0.1085;
      double a, b, c, d, hh;
      do
      {
        a = (*state)[rlsAnkleAngle] + RandGen::getUniform(-1, 1) * randomize_;
        b = (*state)[rlsKneeAngle]  + RandGen::getUniform(-1, 1) * randomize_;
        c = (*state)[rlsHipAngle]   + RandGen::getUniform(-1, 1) * randomize_;
        d = (*state)[rlsArmAngle]   + RandGen::getUniform(-1, 1) * randomize_;
        hh = loLegLength*cos(a) + upLegLength*cos(a+b);
      }
      while (fabs(a + b + c) > 3.1415/2.0 || hh < 0.07 || b >= -0.02); // knee can have only negative values

      (*state)[rlsAnkleAngle] = a;
      (*state)[rlsKneeAngle] = b;
      (*state)[rlsHipAngle] = c;
      (*state)[rlsArmAngle] = d;

      TRACE("Hip height: " << hh);
    }
  }

  task_reward_ = 0;
  subtask_reward_ = 0;
  subtasks_rewards_.clear();
  CRAWL("Initial state: " << *state);
}

bool LeoSquattingTask::actuate(const Vector &state, const Action &action, Vector *actuation) const
{
  *actuation = action;

  if (friction_compensation_)
  {
    // *** HACK TO MAKE REAL LEO SQUAT IN VOLTAGE CONTROL USING NMPC/MLRTI ***
    if (target_env_)
    {
    // Gearbox effeciency is 75%
    *actuation *= 0.75;

    // Coulomb friction
    double f = 0.25*DXL_RESISTANCE/(DXL_TORQUE_CONST*DXL_GEARBOX_RATIO); // = 0.f * 3.420806273
    if (fabs(state[rlsRefRootZ] - 0.28) < 0.00001)
      *actuation += VectorConstructor(+1, -1, +1, 0)*f*1.5;
    else
      *actuation += VectorConstructor(-1, +1, -1, 0)*f;
    }
  }

  return true;
}

void LeoSquattingTask::observe(const Vector &state, Observation *obs, int *terminal) const
{
  grl_assert(state.size() == rlsStateDim);

  obs->v.resize(2*dof_+2);
  if (fixed_arm_)
  {
    obs->v << state[rlsAnkleAngle], state[rlsKneeAngle], state[rlsHipAngle],
              state[rlsAnkleAngleRate], state[rlsKneeAngleRate], state[rlsHipAngleRate],
              state[rlsTemperature], state[rlsRefRootZ];
  }
  else
  {
    obs->v << state[rlsAnkleAngle], state[rlsKneeAngle], state[rlsHipAngle], state[rlsArmAngle],
              state[rlsAnkleAngleRate], state[rlsKneeAngleRate], state[rlsHipAngleRate], state[rlsArmAngleRate],
              state[rlsTemperature], state[rlsRefRootZ];
  }

  if ((timeout_> 0) && (state[rlsTime] >= timeout_))
  {
    *terminal = 1;
    obs->absorbing = false;
  }
  else if (failed(state))
  {
    *terminal = 2;
    obs->absorbing = true;
  }
  else
 {
    *terminal = 0;
    obs->absorbing = false;
  }

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
  grl_assert(state.size() == rlsStateDim);
  grl_assert(action.size() == dof_);
  grl_assert(next.size() == rlsStateDim);

  // increment failures
  if (!test_)
  {
    if (failed(next) == 2)
      falls_++;
    if (failed(next))
      rl_failures_++;
    if (nmpc_failed(next))
      nmpc_failures_++;
  }

  if (failed(next))
  {
    *reward = -100;
    task_reward_ += -100;
    subtask_reward_ += -100;
    return;
  }

  // Since action is not used in the cost function, there is no need to add the friction bias
  //actuate(next, action, &actuation);

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
  cost_nmpc_aux += pow(50.00 * fabs( next[rlsAnkleAngle] + next[rlsKneeAngle] + next[rlsHipAngle] - (0.3) ), power_); // desired torso angle

  // regularize torso
  // is this a good way for torso? Results in a very high penalty, and very weird behaviour
  //cost += pow(60.00 * (next[rlsAnkleAngleRate] + next[rlsKneeAngleRate] + next[rlsHipAngleRate]), 2);

  // regularize: || qdot ||_2^2
  // res[res_cnt++] = 6.00 * sd[QDOTS["arm"]]; // arm
  double rateW = 3.0; // 6.0
  if (!fixed_arm_)
    cost_nmpc_qd += pow(rateW * fabs(next[rlsArmAngleRate]), power_); // arm, added to cost only if nmpc adds it
  cost_nmpc_qd += pow(rateW * fabs(next[rlsHipAngleRate]), power_); // hip_left
  cost_nmpc_qd += pow(rateW * fabs(next[rlsKneeAngleRate]), power_); // knee_left
  cost_nmpc_qd += pow(rateW * fabs(next[rlsAnkleAngleRate]), power_); // ankle_left

  // regularize: || u ||_2^2
  // res[res_cnt++] = 0.01 * u[TAUS["arm"]]; // arm

  TRACE(cost_nmpc);
  TRACE(cost_nmpc_aux);
  TRACE(cost_nmpc_qd);

  // reward is a negative of cost
  double immediate_reward = -weight_nmpc_*(cost_nmpc + weight_nmpc_aux_*(cost_nmpc_aux + weight_nmpc_qd_*cost_nmpc_qd));
  task_reward_ += immediate_reward;
  subtask_reward_ += immediate_reward;

  if (use_mef_)
    *reward = next[rlsMEF];
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

bool LeoSquattingTask::invert(const Observation &obs, Vector *state) const
{
  state->resize(obs.size());
  *state = obs;
}

int LeoSquattingTask::failed(const Vector &state) const
{
  if (std::isnan(state[rlsRootZ]))
    ERROR("NaN value of root, try to reduce integration period to cope with this.");

  if (continue_after_fall_)
    return 0;

  double torsoAngle = state[rlsAnkleAngle] + state[rlsKneeAngle] + state[rlsHipAngle];
  if (fabs(torsoAngle) > 1.0) // > 57 deg
  {
    TRACE("Terminate on large torso.");
    return 2;
  }
  //
  else if (state[rlsAnkleAngleRate] < target_obs_min_[rlsAnkleAngleRate])
  {
    TRACE("Terminate on large negative ankle angle rate.");
    return 1;
  }
  else if (state[rlsAnkleAngleRate] > target_obs_max_[rlsAnkleAngleRate])
  {
    TRACE("Terminate on large positive ankle angle rate.");
    return 1;
  }
  //
  else if (state[rlsKneeAngleRate]  < target_obs_min_[rlsKneeAngleRate])
  {
    TRACE("Terminate on large negative knee angle rate.");
    return 1;
  }
  else if (state[rlsKneeAngleRate]  > target_obs_max_[rlsKneeAngleRate])
  {
    TRACE("Terminate on large positive knee angle rate.");
    return 1;
  }
  //
  else if (state[rlsHipAngleRate]   < target_obs_min_[rlsHipAngleRate])
  {
    TRACE("Terminate on large negative hip angle rate.");
    return 1;
  }
  else if (state[rlsHipAngleRate]   > target_obs_max_[rlsHipAngleRate])
  {
    TRACE("Terminate on large positive hip angle rate.");
    return 1;
  }
  //
  else if (state[rlsArmAngleRate]   < target_obs_min_[rlsArmAngleRate])
  {
    TRACE("Terminate on large negative ankle angle rate.");
    return 1;
  }
  else if (state[rlsArmAngleRate]   > target_obs_max_[rlsArmAngleRate])
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

int LeoSquattingTask::nmpc_failed(const Vector &state) const
{
  if (state[rlsAnkleAngle] < -1.57 || state[rlsAnkleAngle] > 1.45 ||
      state[rlsKneeAngle] < -2.53 || state[rlsKneeAngle] > -0.02 ||
      state[rlsHipAngle] < -0.61 || state[rlsHipAngle] > 2.53 ||
      state[rlsRootX] < state[rlsLeftHeelX] || state[rlsRootX] > state[rlsLeftTipX])
    return 1;
  return 0;
}

void LeoSquattingTask::report(std::ostream &os, const Vector &state) const
{
  const int pw = 15;
  std::stringstream progressString;
  progressString << std::fixed << std::setprecision(5) << std::right;
  progressString << std::setw(pw) << state[rlsRootZ];
  progressString << std::setw(pw) << state[rlsSquats];
  progressString << std::setw(pw) << task_reward_;

  // append cumulative reward in case of timeout termination
  if (subtask_reward_ != 0)
    subtasks_rewards_.push_back(subtask_reward_);   

  int max_size = 6;
  int size = std::min(max_size, static_cast<int>(subtasks_rewards_.size()));

  if (size && fabs(subtasks_rewards_[0]) < 0.01)
    ERROR("Wrong calculations!");

  for (int i = 0; i < size; i++)
    progressString << std::setw(pw) << subtasks_rewards_[i];

  for (int i = size; i < max_size; i++)
    progressString << std::setw(pw) << std::numeric_limits<double>::quiet_NaN();

  progressString << std::setw(pw) << falls_;
  progressString << std::setw(pw) << rl_failures_;
  progressString << std::setw(pw) << nmpc_failures_;

  os << progressString.str();
}
