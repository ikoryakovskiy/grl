/** \file rbdl_leo_walking task.h
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
 
#ifndef GRL_RBDL_LEO_WALKING_TASK_H_
#define GRL_RBDL_LEO_WALKING_TASK_H_

#include <functional>
#include <grl/environment.h>
#include <grl/environments/rbdl.h>

namespace grl
{

// TODO change rlw
enum RbdlLeoWalkingState
{
    rlwTorsoX,
    rlwTorsoZ,
    rlwTorsoAngle,
    rlwLeftHipAngle,
    rlwRightHipAngle,
    rlwLeftKneeAngle,
    rlwRightKneeAngle,
    rlwLeftAnkleAngle,
    rlwRightAnkleAngle,
//    rlwArmAngle,

    rlwTorsoXRate,
    rlwTorsoZRate,
    rlwTorsoAngleRate,
    rlwLeftHipAngleRate,
    rlwRightHipAngleRate,
    rlwLeftKneeAngleRate,
    rlwRightKneeAngleRate,
    rlwLeftAnkleAngleRate,
    rlwRightAnkleAngleRate,
//    rlwArmAngleRate,

    rlwTime,

    // Contact points locations
    rlwLeftTipX,
    rlwLeftTipY,
    rlwLeftTipZ,
    rlwRightTipX,
    rlwRightTipY,
    rlwRightTipZ,
    rlwLeftHeelX,
    rlwLeftHeelY,
    rlwLeftHeelZ,
    rlwRightHeelX,
    rlwRightHeelY,
    rlwRightHeelZ,

    rlwLeftTipVelX,
    rlwLeftTipVelY,
    rlwLeftTipVelZ,
    rlwRightTipVelX,
    rlwRightTipVelY,
    rlwRightTipVelZ,
    rlwLeftHeelVelX,
    rlwLeftHeelVelY,
    rlwLeftHeelVelZ,
    rlwRightHeelVelX,
    rlwRightHeelVelY,
    rlwRightHeelVelZ,

    rlwComX,
    rlwComY,
    rlwComZ,

    rlwPrevComX,

    rlwModelStateDim
};

enum WalkingTaskState
{
  rlwStateDim = rlwModelStateDim
};

class LeoWalkingTask : public Task
{
  public:
    TYPEINFO("task/leo_walking", "Task specification for Leo walking with all joints actuated (except for shoulder)")

  public:
    LeoWalkingTask() : test_(0), target_env_(NULL), randomize_(.0), measurement_noise_(0), dof_(4), timeout_(0), falls_(0),
      trialWork_(.0), knee_mode_("fail_and_restart"), rwForward_(300.), rwTime_(-1.5), rwWork_(-2), rwFail_(-75), rwBrokenKnee_(-0.1) { }

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Task
    virtual void start(int test, Vector *state) const;
    virtual void observe(const Vector &state, Observation *obs, int *terminal) const;
    virtual void evaluate(const Vector &state, const Action &action, const Vector &next, double *reward) const;
    virtual void report(std::ostream &os, const Vector &state) const;

  protected:
    mutable int test_;
    Environment *target_env_;
    double randomize_;
    int measurement_noise_;
    int dof_;
    double timeout_;
    Vector target_obs_min_, target_obs_max_;
    mutable int falls_;
    mutable double trialWork_;
    std::string knee_mode_;

    // reward weights
    double rwForward_;
    double rwTime_;
    double rwWork_;
    double rwFail_;
    double rwBrokenKnee_;

  protected:
    virtual double calculateReward(const Vector &state, const Vector &next) const;
    virtual bool isDoomedToFall(const Vector &state) const;
    virtual bool isKneeBroken(const Vector &state) const;
    virtual double getMotorWork(const Vector &state, const Vector &next, const Action &action) const;
    virtual void initLeo(int test, Vector *state, int sym_rand = 0) const;
};

class LeoBalancingTask : public LeoWalkingTask
{
  public:
    TYPEINFO("task/leo_balancing", "Task specification for Leo standing straight and balancing with all joints actuated (except for shoulder) and both feet on the floor")

  public:
    LeoBalancingTask() {}

    // From Task
    virtual void start(int test, Vector *state) const;

  protected:
    virtual double calculateReward(const Vector &state, const Vector &next) const;
};

class LeoCrouchingTask : public LeoBalancingTask
{
  public:
    TYPEINFO("task/leo_crouching", "Task specification for Leo crouching and balancing with all joints actuated (except for shoulder) and both feet on the floor")

  public:
    LeoCrouchingTask() {}

    // From Task
    virtual void start(int test, Vector *state) const;

  protected:
    virtual bool isDoomedToFall(const Vector &state) const;
  //  virtual double calculateReward(const Vector &state, const Vector &next) const;
};

}

#endif /* GRL_RBDL_LEO_WALKING_TASK_H_ */
