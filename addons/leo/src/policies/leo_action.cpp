/** \file action.cpp
 * \brief Action policy for Leo source file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-04-29
 *
 * \copyright \verbatim
 * Copyright (c) 2015, Wouter Caarls
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

#include <grl/policies/leo_action.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoActionPolicyLearn)
REGISTER_CONFIGURABLE(LeoActionPolicyTest)

void LeoActionPolicyLearn::request(ConfigurationRequest *config)
{
  ActionPolicy::request(config);
  config->push_back(CRP("sub_sigma_signal", "signal/vector", "Subscriber to external sigma", sub_sigma_signal_, true));
}

void LeoActionPolicyLearn::configure(Configuration &config)
{
  ActionPolicy::configure(config);
  sub_sigma_signal_ = (VectorSignal*)config["sub_sigma_signal"].ptr();
}

void LeoActionPolicyLearn::act(const Observation &in, Action *out) const
{
  Vector smart_sigma = sigma_;

  if (sub_sigma_signal_)
  {
    if (sub_sigma_signal_->get().size())
      smart_sigma = sub_sigma_signal_->get();
  }

  ProjectionPtr p = projector_->project(in);
  representation_->read(p, &out->v);
  out->type = atGreedy;

  // Some representations may not always return a value.
  if (!out->size())
    *out = (min_+max_)/2;

  for (size_t ii=0; ii < out->size(); ++ii)
  {
    if (smart_sigma[ii])
    {
      (*out)[ii] += RandGen::getNormal(0., smart_sigma[ii]);
      out->type = atExploratory;
    }

    (*out)[ii] = fmin(fmax((*out)[ii], min_[ii]), max_[ii]);
  }
}

///////////////////////////////////////////

void LeoActionPolicyTest::request(ConfigurationRequest *config)
{
  ActionPolicy::request(config);
  config->push_back(CRP("sub_error_signal", "signal/vector", "Subscriber to an error signal", sub_error_signal_, true));
  config->push_back(CRP("pub_sigma_signal", "signal/vector", "Publisher of sigma", pub_sigma_signal_, true));
}

void LeoActionPolicyTest::configure(Configuration &config)
{
  ActionPolicy::configure(config);
  sub_error_signal_ = (VectorSignal*)config["sub_error_signal"].ptr();
  pub_sigma_signal_ = (VectorSignal*)config["pub_sigma_signal"].ptr();
}

void LeoActionPolicyTest::act(const Observation &in, Action *out) const
{
  if (pub_sigma_signal_ && sub_error_signal_)
  {
    Vector smart_sigma = 1000.0*sub_error_signal_->get();
    pub_sigma_signal_->set(smart_sigma);
  }

  ProjectionPtr p = projector_->project(in);
  representation_->read(p, &out->v);
  out->type = atGreedy;
  
  // Some representations may not always return a value.
  if (!out->size())
    *out = (min_+max_)/2;

  for (size_t ii=0; ii < out->size(); ++ii)
  {
    if (sigma_[ii])
    {
      (*out)[ii] += RandGen::getNormal(0., sigma_[ii]);
      out->type = atExploratory;
    }

    (*out)[ii] = fmin(fmax((*out)[ii], min_[ii]), max_[ii]);
  }
}
