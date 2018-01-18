/** \file rbdl_leo_model.cpp
 * \brief RBDL file for C++ description of Leo model.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2016-09-13
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

#include <grl/environments/leo/rbdl_leo_model.h>
#include <DynamixelSpecs.h>
#include <iomanip>

using namespace grl;

void LeoSandboxModel::request(ConfigurationRequest *config)
{
  dm_.request(config);
  config->pop_back();
  config->push_back(CRP("target_dof", "int.target_dof", "Number of degrees of freedom of the target model", target_dof_, CRP::Configuration, 0, INT_MAX));
  config->push_back(CRP("target_env", "environment", "Interaction environment", target_env_, true));
  config->push_back(CRP("dynamics", "dynamics/rbdl", "Equations of motion", dm_.dynamics_));
  config->push_back(CRP("true_model", "model", "True dynamical model", true_model_, true));

  config->push_back(CRP("animation", "Save current state or full animation", animation_, CRP::Configuration, {"nope", "full", "immediate"}));
}

void LeoSandboxModel::configure(Configuration &config)
{
  target_dof_ = config["target_dof"];
  target_env_ = (Environment*)config["target_env"].ptr(); // Select a real enviromnent if needed

  dm_.configure(config);
  dynamics_ = (RBDLDynamics*) dm_.dynamics_;

  true_model_ = (Model*)config["true_model"].ptr();

  animation_ = config["animation"].str();
  if (animation_ == "full")
  {
    // remove file because will be append otherwice
    std::remove("sd_leo.csv");
    std::remove("u_leo.csv");
    std::remove("aux_leo.csv");
  }
}

void LeoSandboxModel::export_meshup_animation(double time, const Vector &state, const Vector &action,
                                              const std::vector<std::string> &aux) const
{
  if (animation_ == "nope")
    return;

  std::ios_base::openmode om = (animation_ == "full" && action.size() != 0)?(std::ios_base::app):(std::ios_base::trunc);

  std::ofstream data_stream;
  data_stream.open("sd_leo.csv", om);
  if (!data_stream || 2*target_dof_ > state.size())
  {
    std::cerr << "Error opening file sd_leo.csv" << std::endl;
    abort();
  }
  data_stream << time << ", ";
  for (int i = 0; i < 2*target_dof_-1; i++)
    data_stream << state[i] << ", ";
  data_stream << state[2*target_dof_-1] << std::endl;
  data_stream.close();

  data_stream.open ("u_leo.csv", om);
  if (!data_stream || target_dof_ > action.size())
  {
    std::cerr << "Error opening file u_leo.csv" << std::endl;
    abort();
  }
  data_stream << time << ", ";
  for (int i = 0; i < target_dof_-1; i++)
    data_stream << action[i] << ", ";
  data_stream << action[target_dof_-1] << std::endl;
  data_stream.close();

  if (aux.size())
  {
    data_stream.open ("aux_leo.csv", om);
    if (!data_stream)
    {
      std::cerr << "Error opening file aux_leo.csv" << std::endl;
      abort();
    }
    data_stream << time << ", ";
    for (int i = 0; i < aux.size()-1; i++)
      data_stream << aux[i] << ", ";
    data_stream << aux[aux.size()-1] << std::endl;
    data_stream.close();
  }
}
