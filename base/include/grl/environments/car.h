/** \file car.h
 * \brief  * \brief Car accelerating and decelerating on a flat surface with (optional) static friction,  environment header file.
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
 
#ifndef GRL_CAR_ENVIRONMENT_H_
#define GRL_CAR_ENVIRONMENT_H_

#include <grl/environment.h>

namespace grl
{

/// Car model
class CarModel : public Model
{
  public:
    TYPEINFO("model/car", "Car model with (optional) static friction")

  protected:
    double tau_;
    double m_, mu_;
  
  public:
    CarModel() : tau_(1), m_(1), mu_(0) { }
  
    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Model
    virtual double step(const Vector &state, const Vector &actuation, Vector *next) const;

    virtual double friction(double xd, double uc, double kc) const;
};

/// Regulator task
class CarRegulatorTask : public RegulatorTask
{
  public:
    TYPEINFO("task/car/regulator", "Car regulator task")
    
    double timeout_;

  public:
    CarRegulatorTask() : timeout_(20.)
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

#endif /* GRL_CAR_ENVIRONMENT_H_ */
