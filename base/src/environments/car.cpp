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

REGISTER_CONFIGURABLE(StateSpaceModel)
REGISTER_CONFIGURABLE(CarStateSpaceModel)
REGISTER_CONFIGURABLE(StateSpaceRegulatorTask)

void StateSpaceModelBase::request(ConfigurationRequest *config)
{
  config->push_back(CRP("control_step", "double.control_step", "Control step time", tau_, CRP::Configuration, 0.001, DBL_MAX));
  config->push_back(CRP("coulomb", "Coulomb friction coeffecient in a joint", coulomb_, CRP::Configuration, 0., DBL_MAX));
}

void StateSpaceModelBase::configure(Configuration &config)
{
  tau_ = config["control_step"];
  coulomb_ = config["coulomb"];
}

double StateSpaceModelBase::coulomb_friction(double xd, double uc, double kc) const
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

double StateSpaceModelBase::step(const Vector &state, const Vector &actuation, Vector *next) const
{
  next->resize(3);
  double a = actuation[0] - coulomb_friction(state[1], actuation[0], coulomb_);
  (*next)[0] = A_(0,0) * state[0] + A_(0,1) * state[1] + B_[0] * a;
  (*next)[1] = A_(1,0) * state[0] + A_(1,1) * state[1] + B_[1] * a;
  (*next)[2] = state[2] + tau_;
  return tau_;
}

////////////////////////

void StateSpaceModel::request(ConfigurationRequest *config)
{
  StateSpaceModelBase::request(config);
  config->push_back(CRP("A", "Row-major matrix A, from x_ = Ax + Bu", VectorConstructor(1, 1, 0, 1), CRP::Configuration));
  config->push_back(CRP("B", "Vector B, from x_ = Ax + Bu", VectorConstructor(0, 1), CRP::Configuration));
}

void StateSpaceModel::configure(Configuration &config)
{
  StateSpaceModelBase::configure(config);

  Vector A = config["A"].v();
  Vector B = config["B"].v();

  if (A.size() != 4)
    throw bad_param("model/1dss:A");
  if (B.size() != 2)
    throw bad_param("model/1dss:B");

  A_ = Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>>(A.data());
  B_ = Eigen::Map<Eigen::Vector2d>(B.data());
}

////////////////////////

void CarStateSpaceModel::request(ConfigurationRequest *config)
{
  config->push_back(CRP("mass", "Car mass", m_, CRP::Configuration, 0., DBL_MAX));
  config->push_back(CRP("coulomb", "Coulomb friction coeffecient between the car and a ground", coulomb_, CRP::Configuration, 0., DBL_MAX));
  config->push_back(CRP("viscous", "Viscous friction coeffecient", viscous_, CRP::Configuration, 0., DBL_MAX));
}

void CarStateSpaceModel::configure(Configuration &config)
{
  m_ = config["mass"];
  coulomb_ = config["coulomb"];
  viscous_ = config["viscous"];

  if (viscous_ == 0.0)
  {
    A_ << 1, tau_, 0, 1;
    B_ << tau_*tau_/(2*m_), tau_/m_;
  }
  else
  {
    double e = exp(-viscous_*tau_/m_);
    A_ << 1, m_*(1-e)/viscous_, 0, e;
    B_ << tau_/viscous_ - m_*(1-e)/(viscous_*viscous_), (1-e)/viscous_;
  }
}

void StateSpaceRegulatorTask::request(ConfigurationRequest *config)
{
  RegulatorTask::request(config);
  config->push_back(CRP("timeout", "Timeout until the task is restarted", (double)timeout_, CRP::Configuration, 0.0, DBL_MAX));
}

void StateSpaceRegulatorTask::configure(Configuration &config)
{
  RegulatorTask::configure(config);
  
  if (q_.size() != 2)
    throw bad_param("task/car/regulator:q");
  if (r_.size() != 1)
    throw bad_param("task/car/regulator:r");

  config.set("observation_min", VectorConstructor(-10., -10.));
  config.set("observation_max", VectorConstructor(10., 10.));
  config.set("action_min", VectorConstructor(-1000));
  config.set("action_max", VectorConstructor(1000));

  timeout_ = config["timeout"];
}

void StateSpaceRegulatorTask::observe(const Vector &state, Observation *obs, int *terminal) const
{
  if (state.size() != 3)
  {
    ERROR("Received state size " << state.size() << ", expected 3");
    throw Exception("task/car/regulator requires dynamics/mountain");
  }
    
  obs->v.resize(2);
  for (size_t ii=0; ii < 2; ++ii)
    (*obs)[ii] = state[ii];
  obs->absorbing = false;

  *terminal = state[2] >= timeout_;
}

bool StateSpaceRegulatorTask::invert(const Observation &obs, Vector *state) const
{
  *state = extend(obs, VectorConstructor(0.));
  
  return true;
}
