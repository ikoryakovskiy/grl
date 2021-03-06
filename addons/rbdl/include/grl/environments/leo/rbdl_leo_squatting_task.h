/** \file rbdl_leo_squatting_task.h
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
 
#ifndef GRL_RBDL_LEO_SQUATTING_TASK_H_
#define GRL_RBDL_LEO_SQUATTING_TASK_H_

#include <functional>
#include <grl/environment.h>
#include <grl/environments/rbdl.h>

namespace grl
{

enum RbdlLeoSquattingState
{
  rlsAnkleAngle,
  rlsKneeAngle,
  rlsHipAngle,
  rlsArmAngle,

  rlsAnkleAngleRate,
  rlsKneeAngleRate,
  rlsHipAngleRate,
  rlsArmAngleRate,

  rlsTime,  // end of rbdl state

  rlsTemperature,
  rlsRefRootZ,

  rlsLeftTipX,
  rlsLeftTipY,
  rlsLeftTipZ,

  rlsLeftHeelX,
  rlsLeftHeelY,
  rlsLeftHeelZ,

  rlsRootX,
  rlsRootY,
  rlsRootZ,

  rlsMass,

  rlsComX,
  rlsComY,
  rlsComZ,

  rlsComVelocityX,
  rlsComVelocityY,
  rlsComVelocityZ,

  rlsAngularMomentumX,
  rlsAngularMomentumY,
  rlsAngularMomentumZ,

  rlsMEF,
  rlsSMAState,

  rlsModelStateDim
};

enum SquattingTaskState
{
  rlsSquats = rlsModelStateDim,
  rlsStateDim
};

class LeoSquattingTask : public Task
{
  public:
    TYPEINFO("task/leo_squatting", "Task specification for Leo squatting without an auto-actuated arm by default")

  public:
    LeoSquattingTask() : target_env_(NULL), timeout_(0), test_(0), weight_nmpc_(0.0001), weight_nmpc_aux_(1.0), weight_nmpc_qd_(1.0), weight_shaping_(0.0),
      use_mef_(0), falls_(0), rl_failures_(0), nmpc_failures_(0), task_reward_(0.0), subtask_reward_(0.0), power_(2.0), randomize_(0.), dof_(4), continue_after_fall_(0), setpoint_reward_(1), gamma_(0.97),
      fixed_arm_(false), lower_height_(0.28), upper_height_(0.35), friction_compensation_(false) { }

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Task
    virtual void start(int test, Vector *state) const;
    virtual void observe(const Vector &state, Observation *obs, int *terminal) const;
    virtual void evaluate(const Vector &state, const Action &action, const Vector &next, double *reward) const;
    virtual void report(std::ostream &os, const Vector &state) const;
    virtual bool actuate(const Vector &state, const Action &action, Vector *actuation) const;
    virtual bool invert(const Observation &obs, Vector *state) const;

  protected:
    virtual int failed(const Vector &state) const;
    virtual int nmpc_failed(const Vector &state) const;

  protected:
    Environment *target_env_;
    double timeout_;
    mutable int test_;
    double weight_nmpc_, weight_nmpc_aux_, weight_nmpc_qd_, weight_shaping_;
    int use_mef_;
    mutable int falls_, rl_failures_, nmpc_failures_;
    mutable double task_reward_;
    mutable double subtask_reward_;
    mutable std::vector<double> subtasks_rewards_;
    double power_;
    double randomize_;
    int dof_;
    Vector target_obs_min_, target_obs_max_;
    int continue_after_fall_;
    int setpoint_reward_;
    double gamma_;
    bool fixed_arm_;
    double lower_height_, upper_height_;
    bool friction_compensation_;
};

}

#endif /* GRL_RBDL_LEO_SQUATTING_TASK_H_ */
