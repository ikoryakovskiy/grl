/** \file car.cpp
 * \brief Car accelerating and decelerating on a flat surface with (optional) static friction, environment source file.
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

#include <grl/environments/car.h>

using namespace grl;

REGISTER_CONFIGURABLE(CarModel)
REGISTER_CONFIGURABLE(CarRegulatorTask)

void CarModel::request(ConfigurationRequest *config)
{
  config->push_back(CRP("control_step", "double.control_step", "Control step time", tau_, CRP::Configuration, 0.001, DBL_MAX));
  config->push_back(CRP("mass", "Car mass", m_, CRP::Configuration, 0., DBL_MAX));
  config->push_back(CRP("friction", "Static friction coeffecient between car and ground", mu_, CRP::Configuration, 0., DBL_MAX));
}

void CarModel::configure(Configuration &config)
{
  tau_ = config["control_step"];
  m_ = config["mass"];
  mu_ = config["friction"];
}

double CarModel::friction(double xd, double uc, double kc) const
{
  // adapted from
  // K. A. J. Verbert, R. Toth and R. Babuska, "Adaptive Friction Compensation: A Globally Stable Approach,"
  // in IEEE/ASME Transactions on Mechatronics, vol. 21, no. 1, pp. 351-363, Feb. 2016.
  double zero_tolerance = 1E-11;

  if (xd > zero_tolerance || (fabs(xd) <= zero_tolerance && uc > kc))
    return kc;

  if (xd < -zero_tolerance || (fabs(xd) <= zero_tolerance && uc < -kc))
    return -kc;

  if (fabs(xd) <= zero_tolerance && fabs(uc) <= kc)
    return uc;

  throw Exception("Unexpected friction model condition");
}

double CarModel::step(const Vector &state, const Vector &actuation, Vector *next) const
{
  next->resize(3);

  double a = actuation[0] - friction(state[1], actuation[0], mu_);

  (*next)[0] = state[0] + tau_*state[1] + a*tau_*tau_/(2*m_);
  (*next)[1] = state[1] + a*tau_/m_;
  (*next)[2] = state[2] + tau_;

  return tau_;
}

void CarRegulatorTask::request(ConfigurationRequest *config)
{
  RegulatorTask::request(config);
  config->push_back(CRP("timeout", "Timeout until the task is restarted", (double)timeout_, CRP::Configuration, 0.0, DBL_MAX));
}

void CarRegulatorTask::configure(Configuration &config)
{
  RegulatorTask::configure(config);
  
  if (q_.size() != 2)
    throw bad_param("task/mountain/regulator:q");
  if (r_.size() != 1)
    throw bad_param("task/mountain/regulator:r");

  config.set("observation_min", VectorConstructor(-10., -10.));
  config.set("observation_max", VectorConstructor(10., 10.));
  config.set("action_min", VectorConstructor(-10));
  config.set("action_max", VectorConstructor(10));

  timeout_ = config["timeout"];
}

void CarRegulatorTask::observe(const Vector &state, Observation *obs, int *terminal) const
{
  if (state.size() != 3)
  {
    ERROR("Received state size " << state.size() << ", expected 3");
    throw Exception("task/mountain/regulator requires dynamics/mountain");
  }
    
  obs->v.resize(2);
  for (size_t ii=0; ii < 2; ++ii)
    (*obs)[ii] = state[ii];
  obs->absorbing = false;

  *terminal = state[2] >= timeout_;
}

bool CarRegulatorTask::invert(const Observation &obs, Vector *state) const
{
  *state = extend(obs, VectorConstructor(0.));
  
  return true;
}
