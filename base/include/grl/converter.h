/** \file converter.h
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
 
#ifndef GRL_CONVERTER_H_
#define GRL_CONVERTER_H_

#include <grl/configurable.h>

namespace grl
{

/// Converts signals.
class StateActionConverter : public Configurable
{
  public:
    TYPEINFO("converter/state_action_converter", "Configurable which is capable of remapping states and actions")

  protected:
    std::vector<int> state_map_, action_map_;
    int state_in_size_, action_out_size_;
    bool fill_;
    double filling_;

  public:
    StateActionConverter() : state_in_size_(0), action_out_size_(0), fill_(false), filling_(.0) { }
    virtual ~StateActionConverter() { }

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config) { }

    // Own
    virtual void convert(const Vector &state_in, Vector &state_out, const Vector &action_in, Vector &action_out) const;
    virtual void convert_state(const Vector &state_in, Vector &state_out) const;
    virtual void convert_action(const Vector &action_in, Vector &action_out) const;
    virtual void convert_action_invert(const Vector &action_out, Vector &action_in) const;
    virtual void prepare(const std::vector<std::string> in, const std::vector<std::string> out, std::vector<int> &map) const;

    virtual int get_state_in_size() { return state_in_size_; }
    virtual int get_action_out_size() { return action_out_size_; }
};

}

#endif /* GRL_CONVERTER_H_ */
