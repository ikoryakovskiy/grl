/** \file py_env.cpp
 * \brief Python access to grl environments.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-12-08
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

#include <grl/environments/py_env.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen.h>
#include <string.h>

using namespace grl;

namespace py = pybind11;

py::tuple PyEnv::init(const std::string &file)
{
  //std::cout << "PyEnv::init" << std::endl;
  if (first)
  {
    loadPlugins();
    first = false;
  }

  if (env)
  {
    // Implemented for Spider kernels which continue running the kernel
    std::cerr << "Re-initializing." << std::endl;
    fini();
  }

  if (file == "")
    std::cerr << "Missing configuration file name." << std::endl;

  Configurator *conf, *envconf;
  if (!(conf = loadYAML(file)) || !(configurator = conf->instantiate()) || !(envconf = configurator->find("environment")))
  {
    safe_delete(&conf);
    safe_delete(&configurator);
    std::cerr << "Configuration file is not valid." << std::endl;
    return py::make_tuple();
  }

  safe_delete(&conf);
  env = dynamic_cast<Environment*>(envconf->ptr());

  if (!env)
  {
    safe_delete(&configurator);
    std::cerr << "Configuration file does not specify a valid environment." << std::endl;
    return py::make_tuple();
  }

  if (envconf->find("observation_min"))
    path = "";
  else if (envconf->find("task/observation_min"))
    path = "task/";
  else
  {
    std::cerr << "Could not determine task specification." << std::endl;
    return py::make_tuple();
  }

  observation_dims = (*envconf)[path+"observation_dims"];
  action_dims = (*envconf)[path+"action_dims"];

  std::cout << "Observation dims: " << observation_dims << std::endl;
  std::cout << "Action dims: " << action_dims << std::endl;

  started = false;

  // Process output
  return py::make_tuple(observation_dims, (*envconf)[path+"observation_min"].v(), (*envconf)[path+"observation_max"].v(),
      action_dims, (*envconf)[path+"action_min"].v(), (*envconf)[path+"action_max"].v());
}

void PyEnv::seed(int seed)
{
  //std::cout << "PyEnv::seed" << std::endl;
  srand(seed);
  srand48(seed);
}

Vector PyEnv::start(int test)
{
  //std::cout << "PyEnv::start" << std::endl;
  if (!env)
    std::cerr << "Not initialized." << std::endl;

  // Run environment
  Observation obs;

  env->start(test, &obs);
  started = true;

  // Process output
  return obs.v;
}

py::tuple PyEnv::step(const Vector &action)
{
  //std::cout << "PyEnv::step" << std::endl;

  if (!env)
    std::cerr << "Not initialized." << std::endl;

  if (!started)
    std::cerr << "Environment not started." << std::endl;

  if (action.size() != action_dims)
    std::cerr << "Invalid action size." << std::endl;

  //std::cout << "PyEnv::step ready" << std::endl;

  // Run environment
  Observation obs;
  double reward;
  int terminal;
  double tau = env->step(action, &obs, &reward, &terminal);

  //std::cout << "PyEnv::step done" << std::endl;

  // Process output
  return py::make_tuple(obs.v, reward, terminal, tau);
}

void PyEnv::fini()
{
  //std::cout << "PyEnv::fini" << std::endl;
  if (!env)
    std::cerr << "Not initialized." << std::endl;

  safe_delete(&configurator);
  env = NULL;
  first=true;
}

PYBIND11_MODULE(py_env, m)
{
  py::class_<PyEnv> py_env_class(m, "PyEnv");
  py_env_class
      .def(py::init<>())
      .def("init",  &PyEnv::init)
      .def("seed",  &PyEnv::seed)
      .def("start", &PyEnv::start)
      .def("step",  &PyEnv::step)
      .def("fini",  &PyEnv::fini);
}
