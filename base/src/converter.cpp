/** \file converter.cpp
 * \brief Class which is capable of remapping states and actions.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2016-01-01
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

#include <grl/environment.h>
#include <grl/converter.h>

using namespace grl;

REGISTER_CONFIGURABLE(StateActionConverter)
REGISTER_CONFIGURABLE(ConvertingEnvironment)

// From Configurable
void StateActionConverter::request(ConfigurationRequest *config)
{
  config->push_back(CRP("state_in", "Comma-separated list of state elements in the input vector", ""));
  config->push_back(CRP("state_out", "Comma-separated list of state elements in the output vector", ""));
  config->push_back(CRP("action_in", "Comma-separated list of action elements observed in the input vector", ""));
  config->push_back(CRP("action_out", "Comma-separated list of action elements provided in the output vector", ""));
  config->push_back(CRP("filling", "Filling of the missing values", ""));
}

void StateActionConverter::configure(Configuration &config)
{
  const std::vector<std::string> state_in = cutLongStr( config["state_in"].str() );
  const std::vector<std::string> state_out = cutLongStr( config["state_out"].str() );
  const std::vector<std::string> action_in = cutLongStr( config["action_in"].str() );
  const std::vector<std::string> action_out = cutLongStr( config["action_out"].str() );
  std::string filling_str = config["filling"].str();

  char* p;
  double filling = strtod(filling_str.c_str(), &p);
  if (!(*p))
  {
    fill_ = true;
    filling_ = filling;
  }

  state_in_size_ = state_in.size();
  action_out_size_ = action_out.size();

  prepare(state_in, state_out, state_map_);
  prepare(action_in, action_out, action_map_);

  INFO("State map: " << state_map_);
  INFO("Action map: " << action_map_);
}

// Own
void StateActionConverter::convert(const Vector &state_in, Vector &state_out, const Vector &action_in, Vector &action_out) const
{
  convert_state(state_in, state_out);
  convert_action(action_in, action_out);
}

void StateActionConverter::convert_state(const Vector &state_in, Vector &state_out) const
{
  if (state_out.size() < state_map_.size())
    state_out.resize(state_map_.size());

  if (state_map_.size() > 0)
  {
    for (int i = 0; i < state_map_.size(); i++)
      if (state_map_[i] >= 0)
        state_out[i] = state_in[ state_map_[i] ];
      else
        state_out[i] = filling_;
  }
  else
  {
    state_out = state_in;
  }
}

void StateActionConverter::convert_action(const Vector &action_in, Vector &action_out) const
{
  if (action_out.size() < action_map_.size())
    action_out.resize(action_map_.size());

  if (action_map_.size() > 0)
  {
    for (int i = 0; i < action_map_.size(); i++)
      if (action_map_[i] >= 0)
        action_out[i] = action_in[ action_map_[i] ];
      else
        action_out[i] = filling_;
  }
  else
  {
    action_out = action_in;
  }
}

void StateActionConverter::convert_action_invert(const Vector &action_out, Vector &action_in) const
{
  // a bit costly operation compared to forward conversion, but ok if used occasionally
  int len = 0;
  for (int i = 0; i < action_map_.size(); i++)
    len += (action_map_[i] >= 0);

  if (action_in.size() < len)
    action_in.resize(len);

  if (action_map_.size() > 0)
  {
    int j = 0;
    for (int i = 0, base = 0; i < action_map_.size(); i++)
      if (action_map_[i] >= 0)
        action_in[j++] = action_out[ base + action_map_[i] ];
      else
        base++;
  }
  else
  {
    action_in = action_out;
  }
}

void StateActionConverter::prepare(const std::vector<std::string> in, const std::vector<std::string> out, std::vector<int> &map) const
{
  std::vector<std::string>::const_iterator it_out = out.begin();
  std::vector<std::string>::const_iterator it_in = in.begin();
  for (; it_out < out.end(); it_out++)
  {
    bool found = false;
    it_in = in.begin();
    for (int i = 0; it_in < in.end(); it_in++, i++)
    {
      if (*it_out == *it_in)
      {
        TRACE("Adding to the observation vector (physical state): " << *it_out);
        map.push_back(i);
        found = true;
        break;
      }
    }

    if (!found)
    {
      if (!fill_)
        throw Exception("Field '" + *it_out + "' is not matched with any input");
      else
        map.push_back(-1); // Will indicate that we need to put "filling_" here
    }
  }
}

////////////////////////////////////////////////////////////////////////////

void ConvertingEnvironment::request(ConfigurationRequest *config)
{
  config->push_back(CRP("environment", "environment", "Environment in which the agent acts", environment_));
  config->push_back(CRP("converter", "converter", "Convert states and actions if needed", converter_, true));

  config->push_back(CRP("observation_dims", "int.observation_dims", 0, CRP::Provided));
  config->push_back(CRP("observation_min", "vector.observation_min", Vector(), CRP::Provided));
  config->push_back(CRP("observation_max", "vector.observation_max", Vector(), CRP::Provided));
  config->push_back(CRP("action_dims", "int.action_dims", 0, CRP::Provided));
  config->push_back(CRP("action_min", "vector.action_min", Vector(), CRP::Provided));
  config->push_back(CRP("action_max", "vector.action_max", Vector(), CRP::Provided));
  config->push_back(CRP("reward_min", "double.reward_min", 0.0, CRP::Provided, -DBL_MAX, DBL_MAX));
  config->push_back(CRP("reward_max", "double.reward_max", 0.0, CRP::Provided, -DBL_MAX, DBL_MAX));

  config->push_back(CRP("target_observation_min", "vector.observation_min", Vector(), CRP::System));
  config->push_back(CRP("target_observation_max", "vector.observation_max", Vector(), CRP::System));
  config->push_back(CRP("target_action_min", "vector.action_min", Vector(), CRP::System));
  config->push_back(CRP("target_action_max", "vector.action_max", Vector(), CRP::System));
  config->push_back(CRP("target_reward_min", "double.reward_min", 0.0, CRP::System, -DBL_MAX, DBL_MAX));
  config->push_back(CRP("target_reward_max", "double.reward_max", 0.0, CRP::System, -DBL_MAX, DBL_MAX));
}

void ConvertingEnvironment::configure(Configuration &config)
{
  environment_ = (Environment*)config["environment"].ptr();
  converter_ = (StateActionConverter*)config["converter"].ptr();

  if (converter_)
  {
    target_obs_.v.resize(converter_->get_state_in_size());
    target_action_.v.resize(converter_->get_action_out_size());

    Vector obs, action;
    converter_->convert_state(config["target_observation_min"], obs);
    config.set("observation_min", obs);
    converter_->convert_state(config["target_observation_max"], obs);
    config.set("observation_max", obs);
    config.set("observation_dims", obs.size());

    converter_->convert_action_invert(config["target_action_min"], action);
    config.set("action_min", action);
    converter_->convert_action_invert(config["target_action_max"], action);
    config.set("action_max", action);
    config.set("action_dims", action.size());

    config.set("reward_min", config["target_reward_min"]);
    config.set("reward_max", config["target_reward_max"]);
  }
}

ConvertingEnvironment &ConvertingEnvironment::copy(const Configurable &obj)
{
  const ConvertingEnvironment& se = dynamic_cast<const ConvertingEnvironment&>(obj);
  environment_ = se.environment_;
  converter_ = se.converter_;
  return *this;
}

void ConvertingEnvironment::start(int test, Observation *obs)
{
  environment_->start(test, &target_obs_);
  TRACE("Target observation " << target_obs_);
  if (converter_)
  {
    converter_->convert_state(target_obs_.v, obs->v);
    obs->absorbing = target_obs_.absorbing;
  }
  else
    *obs = target_obs_;
}

double ConvertingEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  if (converter_)
    converter_->convert_action(action.v, target_action_.v);
  else
    target_action_ = action;

  TRACE("Target action " << target_action_);
  double tau = environment_->step(target_action_, &target_obs_, reward, terminal);
  TRACE("Target observation " << target_obs_);

  if (converter_)
  {
    converter_->convert_state(target_obs_.v, obs->v);
    obs->absorbing = target_obs_.absorbing;
  }
  else
    *obs = target_obs_;

  return tau;
}

void ConvertingEnvironment::report(std::ostream &os) const
{
  environment_->report(os);
}
