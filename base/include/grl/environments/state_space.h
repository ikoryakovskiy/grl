/** \file state_space.h
 * \brief  * \brief State-space environment header file.
 *
 * \author    Ivan Koryakovskiy
 * \date      2017-07-02
 *
 * \copyright \verbatim
 * Copyright (c) 2017, Wouter Caarls
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
 
#ifndef GRL_STATE_SPACE_ENVIRONMENT_H_
#define GRL_STATE_SPACE_ENVIRONMENT_H_

#include <grl/environment.h>

namespace grl
{

class StateSpaceModelBase : public Model
{
  protected:
    double tau_;
    double coulomb_;
    Eigen::Matrix2d A_;
    Eigen::Vector2d B_;

  public:
    StateSpaceModelBase() : tau_(1), coulomb_(0) { }

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Model
    virtual double step(const Vector &state, const Vector &actuation, Vector *next) const;

  private:
    virtual double coulomb_friction(double xd, double uc, double kc) const;
    virtual double coulomb_friction_new(double xd, double uc, double kc) const;
    virtual double coulomb_friction_tanh(double xd, double uc, double kc) const;
};

/// Generic state-space model
class StateSpaceModel : public StateSpaceModelBase
{
  public:
    TYPEINFO("model/1dss/generic", "Generic 1-DOF model defined by matrixes A and B")

  public:
    StateSpaceModel() { }

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
};

/// Car model
class CarStateSpaceModel : public StateSpaceModelBase
{
  public:
    TYPEINFO("model/1dss/car", "Car model with (optional) static friction")

  protected:

    double m_, viscous_;
  
  public:
    CarStateSpaceModel() : m_(1), viscous_(1.0) { }
  
    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
};

/// Regulator task
class StateSpaceRegulatorTask : public RegulatorTask
{
  public:
    TYPEINFO("task/1dss/regulator", "Car regulator task")
    
    double timeout_;

  public:
    StateSpaceRegulatorTask() : timeout_(20.)
    {
      start_ = VectorConstructor(0., 0.);
      goal_ = VectorConstructor(5., 0.);
      stddev_ = VectorConstructor(0., 0.);
      q_ = 2*VectorConstructor(1., 0.);
      r_ = 2*VectorConstructor(0.01);
    }
  
    // From Configurable
    void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Task
    virtual void observe(const Vector &state, Observation *obs, int *terminal) const;
    virtual bool invert(const Observation &obs, Vector *state) const;
};

}

#endif /* GRL_STATE_SPACE_ENVIRONMENT_H_ */
