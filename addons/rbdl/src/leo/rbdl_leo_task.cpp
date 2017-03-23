/** \file rbdl_leo_task.cpp
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

void LeoSquattingTask::request(ConfigurationRequest *config)
{
  Task::request(config);
  config->push_back(CRP("timeout", "double.timeout", "Task timeout", timeout_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("randomize", "int.randomize", "Initialization from a random pose", randomize_, CRP::System, 0, 1));
  config->push_back(CRP("weight", "double.weight", "Weight on the NMPC cost (excluding shaping)", weight_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("weight2", "double.weight", "Weight on the part of NMPC cost (excluding regularizations)", weight2_, CRP::System, 0.0, DBL_MAX));
}

void LeoSquattingTask::configure(Configuration &config)
{
  timeout_ = config["timeout"];
  randomize_ = config["randomize"];
  weight_ = config["weight"];
  weight2_ = config["weight2"];

  // Target observations: 2*target_dof + time
  std::vector<double> obs_min = {-M_PI, -M_PI, -M_PI, -M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, 0};
  std::vector<double> obs_max = { M_PI,  M_PI,  M_PI,  M_PI,  10*M_PI,  10*M_PI,  10*M_PI,  10*M_PI, 1};
  toVector(obs_min, target_obs_min_);
  toVector(obs_max, target_obs_max_);

  // Observations and actions exposed to an agent
  config.set("observation_dims", 2*dof_+1);
  Vector observation_min, observation_max;
  observation_min.resize(2*dof_+1);
  observation_min << target_obs_min_[rlsAnkleAngle], target_obs_min_[rlsKneeAngle], target_obs_min_[rlsHipAngle],
      target_obs_min_[rlsAnkleAngleRate], target_obs_min_[rlsKneeAngleRate], target_obs_min_[rlsHipAngleRate], target_obs_min_[rlsTime];
  observation_max.resize(2*dof_+1);
  observation_max << target_obs_max_[rlsAnkleAngle], target_obs_max_[rlsKneeAngle], target_obs_max_[rlsHipAngle],
      target_obs_max_[rlsAnkleAngleRate], target_obs_max_[rlsKneeAngleRate], target_obs_max_[rlsHipAngleRate], target_obs_max_[rlsTime];
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
  *state = ConstantVector(2*(dof_+1)+1, 0);

  // sitted pose
  *state <<
         1.0586571916803691E+00,
        -2.1266836153365212E+00,
         1.0680264236561250E+00,
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

    (*state)[rlsAnkleAngle] = a;
    (*state)[rlsKneeAngle] = b;
    (*state)[rlsHipAngle] = c;
    (*state)[rlsArmAngle] = d;

    TRACE("Hip height: " << hh);
  }

  CRAWL("Initial state: " << *state);
}

void LeoSquattingTask::observe(const Vector &state, Observation *obs, int *terminal) const
{
  grl_assert(state.size() == stsStateDim);

  // arm is auto-actuated => exclude angle and angle rate from observations
  obs->v.resize(2*dof_+1);
  obs->v << state[rlsAnkleAngle], state[rlsKneeAngle], state[rlsHipAngle],
            state[rlsAnkleAngleRate], state[rlsKneeAngleRate], state[rlsHipAngleRate], state[rlsRefRootZ];

  if ((timeout_> 0) && (state[rlsTime] >= timeout_))
    *terminal = 1;
  else if (failed(state))
    *terminal = 2;
  else
    *terminal = 0;
}

void LeoSquattingTask::evaluate(const Vector &state, const Action &action, const Vector &next, double *reward) const
{
  grl_assert(state.size() == stsStateDim);
  grl_assert(action.size() == dof_);
  grl_assert(next.size() == stsStateDim);

  if (failed(next))
  {
    *reward = -100;
    return;
  }

  double cost_nmpc = 0, cost_nmpc_reg = 0;

  // calculate support center from feet positions
  double suppport_center = 0.5 * (next[rlsLeftTipX] + next[rlsLeftHeelX]);

  // track: || root_z - h_ref ||_2^2
  cost_nmpc +=  pow(50.0 * (next[rlsRootZ] - next[rlsRefRootZ]), 2);

  // track: || com_x - support center_x ||_2^2
  cost_nmpc +=  pow( 100.00 * (next[rlsComX] - suppport_center), 2);

  //double velW = 10.0; // 10.0
  //cost +=  pow( velW * next[rlsComVelocityX], 2);
  //cost +=  pow( velW * next[rlsComVelocityZ], 2);

  //cost +=  pow( 100.00 * next[rlsAngularMomentumY], 2);

  // NOTE: sum of lower body angles is equal to angle between ground slope
  //       and torso. Minimizing deviation from zero keeps torso upright
  //       during motion execution.
  cost_nmpc_reg += pow(30.00 * ( next[rlsAnkleAngle] + next[rlsKneeAngle] + next[rlsHipAngle] - (0.15) ), 2); // desired torso angle

  // regularize torso
  // is this a good way for torso? Results in a very high penalty, and very weird behaviour
  //cost += pow(60.00 * (next[rlsAnkleAngleRate] + next[rlsKneeAngleRate] + next[rlsHipAngleRate]), 2);

  // regularize: || qdot ||_2^2
  // res[res_cnt++] = 6.00 * sd[QDOTS["arm"]]; // arm
  double rateW = 5.0; // 6.0
  cost_nmpc_reg += pow(rateW * next[rlsHipAngleRate], 2); // hip_left
  cost_nmpc_reg += pow(rateW * next[rlsKneeAngleRate], 2); // knee_left
  cost_nmpc_reg += pow(rateW * next[rlsAnkleAngleRate], 2); // ankle_left

  // regularize: || u ||_2^2
  // res[res_cnt++] = 0.01 * u[TAUS["arm"]]; // arm
//  cost += pow(0.01 * action[0], 2); // hip_left
//  cost += pow(0.01 * action[1], 2); // knee_left
//  cost += pow(0.01 * action[2], 2); // ankle_left

  double shaping = 0;

  double w = 10.0;
  double F1, F0;

  F0 = pow(w * (state[rlsRootZ] - next[rlsRefRootZ]), 2); // distance to setpoint at time (t)
  F1 = pow(w * (next [rlsRootZ] - next[rlsRefRootZ]), 2); // distance to setpoint at time (t+1)

  shaping += F0 - F1; // positive reward for getting closer to the setpoint

  TRACE(state[rlsRootZ] << ", " << next[rlsRootZ] << " -> " << next[rlsRefRootZ]);
  TRACE(F1 << " - " << F0 << " = " << shaping);

  // reward is a negative of cost
  *reward = -weight_*(cost_nmpc + weight2_*cost_nmpc_reg) + shaping;
}

int LeoSquattingTask::failed(const Vector &state) const
{
  if (std::isnan(state[rlsRootZ]))
    ERROR("NaN value of root, try to reduce integration period to cope with this.");

  double torsoAngle = state[rlsAnkleAngle] + state[rlsKneeAngle] + state[rlsHipAngle];
  if (fabs(torsoAngle) > 1.0 || // > 57 deg
      // penalty for high joint velocities
      (state[rlsAnkleAngleRate] < target_obs_min_[rlsAnkleAngleRate]) ||
      (state[rlsAnkleAngleRate] > target_obs_max_[rlsAnkleAngleRate]) ||
      (state[rlsKneeAngleRate]  < target_obs_min_[rlsKneeAngleRate])  ||
      (state[rlsKneeAngleRate]  > target_obs_max_[rlsKneeAngleRate])  ||
      (state[rlsHipAngleRate]   < target_obs_min_[rlsHipAngleRate])   ||
      (state[rlsHipAngleRate]   > target_obs_max_[rlsHipAngleRate])   ||
      (state[rlsArmAngleRate]   < target_obs_min_[rlsArmAngleRate])   ||
      (state[rlsArmAngleRate]   > target_obs_max_[rlsArmAngleRate])   ||
      (state[rlsRootZ] < 0)
      )
    return 1;
  else
  {
    return 0;
  }
}

void LeoSquattingTask::report(std::ostream &os, const Vector &state) const
{
  const int pw = 15;
  std::stringstream progressString;
  progressString << std::fixed << std::setprecision(3) << std::right;
  progressString << std::setw(pw) << state[rlsRootZ];
  progressString << std::setw(pw) << state[stsSquats];
  os << progressString.str();
}
