/** \file lowpass.cpp
 * \brief Post-policy that filters output source file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-06-23
 *
 * \copyright \verbatim
 * Copyright (c) 2016, Wouter Caarls
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

#include <grl/policies/lowpass.h>
#include <stdlib.h>

using namespace grl;

REGISTER_CONFIGURABLE(LowpassPolicy)

LowpassPolicy::~LowpassPolicy()
{
  for (int i = 0; i < filters_.size(); i++)
    delete filters_[i];
}

void LowpassPolicy::request(ConfigurationRequest *config)
{
  std::string order = std::to_string(order_);
  config->push_back(CRP("order", "Save policy to 'output' at the end of event", order, CRP::Configuration, {"0", "1", "2", "3"}));
  config->push_back(CRP("sampling", "double", "Sampling frequency, Hz", sampling_, CRP::Configuration, 1.0, 1000.0));
  config->push_back(CRP("cutoff", "double", "Cut-off frequency, Hz", cutoff_, CRP::Configuration, 1.0, 1000.0));
  config->push_back(CRP("action_dims", "int.action_dims", "Number of action dimensions", action_dims_, CRP::Configuration, 1, INT_MAX));
  config->push_back(CRP("magnitude", "Binary magnitude of the effect of the filter", magnitude_, CRP::Configuration));

  config->push_back(CRP("policy", "mapping/policy", "Policy to inject noise into", policy_));
}

void LowpassPolicy::configure(Configuration &config)
{
  policy_ = (Policy*)config["policy"].ptr();

  std::string order = config["order"];
  order_ = atoi(order.c_str());
  sampling_ = config["sampling"];
  cutoff_ = config["cutoff"];
  action_dims_ = config["action_dims"];
  magnitude_ = config["magnitude"].v();

  if (magnitude_.size() == 0)
    magnitude_ = ConstantVector(action_dims_, 1.0);

  if (magnitude_.size() != action_dims_)
    grl::bad_param("mapping/policy/post/lowpass:action_dims");

  if (order_)
  {
    for (int i = 0; i < action_dims_; i++)
    {
      CFilterBase *f = NULL;
      if (magnitude_[i])
      {
        switch(order_)
        {
          case 1:
            f = new CButterworthFilter<1>();
            break;
          case 2:
            f = new CButterworthFilter<2>();
            break;
          case 3:
            f = new CButterworthFilter<3>();
            break;
          default:
            throw bad_param("policy/lowpass:order");
        }
        f->init(sampling_, cutoff_);
      }
      filters_.push_back(f);
    }
  }
}

void LowpassPolicy::reconfigure(const Configuration &config)
{
}

void LowpassPolicy::act(const Observation &in, Action *out) const
{
  policy_->act(in, out);

  if (action_dims_ != out->size())
    throw bad_param("policy/lowpass:action_dims");

  for (int i = 0; i < action_dims_; i++)
    if (magnitude_[i] && order_)
      (*out)[i] = filters_[i]->filter( (*out)[i] );
}

void LowpassPolicy::act(double time, const Observation &in, Action *out)
{
  if (time == 0.)
  {
    for (int i = 0; i < action_dims_; i++)
      if (magnitude_[i] && order_)
        filters_[i]->clear();
  }

  policy_->act(time, in, out);

  if (action_dims_ != out->size())
    throw bad_param("policy/lowpass:action_dims");

  for (int i = 0; i < action_dims_; i++)
    if (magnitude_[i] && order_)
      (*out)[i] = filters_[i]->filter( (*out)[i] );
}
