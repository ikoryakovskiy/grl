/** \file rbdl_leo_task.h
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
 
#ifndef GRL_RBDL_LEO_TASK_H_
#define GRL_RBDL_LEO_TASK_H_

#include <functional>
#include <grl/environment.h>
#include <grl/environments/rbdl.h>

namespace grl
{

enum RbdlLeoState
{
  rlsAnkleAngle,
  rlsKneeAngle,
  rlsHipAngle,
  rlsArmAngle,

  rlsAnkleAngleRate,
  rlsKneeAngleRate,
  rlsHipAngleRate,
  rlsArmAngleRate,

  rlsTime,
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

  rlsStateDim = rlsAngularMomentumZ + 1
};

enum SquattingTaskState
{
  stsSquats = rlsStateDim,
  stsStateDim
};

class LeoSquattingTask : public Task
{
  public:
    TYPEINFO("task/leo_squatting", "Task specification for Leo squatting with an auto-actuated arm")

  public:
    LeoSquattingTask() : target_env_(NULL), timeout_(0), weight_nmpc_(0.0001), weight_nmpc_aux_(1.0), weight_shaping_(0.0), randomize_(0), dof_(3), gamma_(0.95) { }

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Task
    virtual void start(int test, Vector *state) const;
    virtual void observe(const Vector &state, Observation *obs, int *terminal) const;
    virtual void evaluate(const Vector &state, const Action &action, const Vector &next, double *reward) const;
    virtual void report(std::ostream &os, const Vector &state) const;

  protected:
    virtual int failed(const Vector &state) const;

  protected:
    Environment *target_env_;
    double timeout_;
    double weight_nmpc_, weight_nmpc_aux_, weight_shaping_;
    int randomize_;
    int dof_;
    double gamma_;
    Vector target_obs_min_, target_obs_max_;
};

}

#endif /* GRL_RBDL_LEO_TASK_H_ */
