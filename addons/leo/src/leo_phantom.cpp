/** \file leo_phantom.cpp
 * \brief Phantom environment source file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@gmail.com>
 * \date      2017-03-08
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

#include <grl/environments/leo/leo_phantom.h>
#include <iomanip>
#include <unistd.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoPhantomEnvironment)

void LeoPhantomEnvironment::request(ConfigurationRequest *config)
{
  config->push_back(CRP("importer", "importer", "Importer with time as the first column", importer_));
  config->push_back(CRP("exporter", "exporter", "Optional exporter for transition log (supports time, state, observation, action, reward, terminal)", exporter_, true));
  config->push_back(CRP("sub_transition_type", "signal/vector", "Subscriber to the transition type", sub_transition_type_, true));
  config->push_back(CRP("pub_ic_signal", "signal/vector", "Publisher of the initialization and contact signal", pub_ic_signal_, true));
}

void LeoPhantomEnvironment::configure(Configuration &config)
{
  Vector time, state0, state1, action, reward, terminal, transition_type, contact;
  importer_ = (Importer*)config["importer"].ptr();
  importer_->open();
  while (importer_->read({&time, &state0, &state1, &action, &reward, &terminal, &transition_type, &contact}))
  {
    time_.push_back(time);
    state0_.push_back(state0);
    contact_.push_back(contact);
  }

  if (state0_.empty())
  {
    ERROR("Could not import file");
    throw bad_param("environment/phantom:importer");
  }

  exporter_ = (Exporter*) config["exporter"].ptr();
  if (exporter_)
    exporter_->init({"time", "state0", "state1", "action", "reward", "terminal", "transition_type", "contact"});

  sub_transition_type_ = (VectorSignal*)config["sub_transition_type"].ptr();
  pub_ic_signal_ = (VectorSignal*)config["pub_ic_signal"].ptr();
}

void LeoPhantomEnvironment::reconfigure(const Configuration &config)
{
}

void LeoPhantomEnvironment::start(int test, Observation *obs)
{
  idx_= 0;

  // copy observation
  obs->v.resize(state0_[idx_].size());
  obs->v = state0_[idx_];
  CRAWL(obs->v);

  // copy contact
  if (pub_ic_signal_)
  {
    pub_ic_signal_->set(contact_[idx_]);
    CRAWL(contact_[idx_]);
  }

  if (exporter_)
    exporter_->open("test", 0);
}

double LeoPhantomEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  TRACE(state0_[idx_]);
  TRACE(action);

  *reward = 0;

  if (idx_+2 < time_.size())
    *terminal = 0;
  else
    *terminal = 1;

  if (exporter_)
    exporter_->write({time_[idx_], state0_[idx_], state0_[idx_+1], action.v,
                      grl::VectorConstructor(*reward), grl::VectorConstructor(*terminal),
                      grl::VectorConstructor(atUndefined), contact_[idx_]
                     });

  // Prepare new observation
  obs->v.resize(state0_[idx_+1].size());
  obs->v = state0_[idx_+1];
  CRAWL(obs->v);

  // Prepare new contact
  if (pub_ic_signal_)
  {
    pub_ic_signal_->set(contact_[idx_+1]);
    CRAWL(contact_[idx_+1]);
  }

  double tau = (time_[idx_+1][0] - time_[idx_][0]);
  ++idx_;
  return tau;
}


