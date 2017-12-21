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
  if (first_)
  {
    loadPlugins();
    first_ = false;
  }

  if (env_)
  {
    // Implemented for Spider kernels which continue running the kernel
    std::cerr << "Re-initializing." << std::endl;
    fini();
  }

  if (file == "")
    std::cerr << "Missing configuration file name." << std::endl;

  Configurator *conf, *envconf;
  if (!(conf = loadYAML(file)) || !(configurator_ = conf->instantiate()) || !(envconf = configurator_->find("environment")))
  {
    safe_delete(&conf);
    safe_delete(&configurator_);
    std::cerr << "Configuration file is not valid." << std::endl;
    return py::make_tuple();
  }

  safe_delete(&conf);
  env_ = dynamic_cast<Environment*>(envconf->ptr());

  if (!env_)
  {
    safe_delete(&configurator_);
    std::cerr << "Configuration file does not specify a valid environment." << std::endl;
    return py::make_tuple();
  }

  std::string path;
  if (envconf->find("observation_min"))
    path = "";
  else if (envconf->find("task/observation_min"))
    path = "task/";
  else
  {
    std::cerr << "Could not determine task specification." << std::endl;
    return py::make_tuple();
  }

  observation_dims_ = (*envconf)[path+"observation_dims"];
  action_dims_ = (*envconf)[path+"action_dims"];

  INFO("PyEnv: Observation dims: " << observation_dims_);
  INFO("PyEnv: Action dims: " << action_dims_);

  started_ = false;

  // Process output
  return py::make_tuple(observation_dims_, (*envconf)[path+"observation_min"].v(), (*envconf)[path+"observation_max"].v(),
      action_dims_, (*envconf)[path+"action_min"].v(), (*envconf)[path+"action_max"].v());
}

void PyEnv::seed(int seed)
{
  srand(seed);
  srand48(seed);
}

Vector PyEnv::start(int test)
{
  if (!env_)
    std::cerr << "Not initialized." << std::endl;

  test_ = test;

  // Run environment
  Observation obs;

  env_->start(test, &obs);
  started_ = true;

  // Process output
  return obs.v;
}

py::tuple PyEnv::step(const Vector &action)
{
  if (!env_)
    std::cerr << "Not initialized." << std::endl;

  if (!started_)
    std::cerr << "Environment not started." << std::endl;

  if (action.size() != action_dims_)
    std::cerr << "Invalid action size." << std::endl;

  // Run environment
  Observation obs;
  double reward;
  int terminal;
  env_->step(action, &obs, &reward, &terminal);

  // Pass extra info if episode is testing and has finished
  std::ostringstream oss;
  if (((test_ == report_idx_) || report_idx_ == 2) && terminal)
    env_->report(oss);

  // Process output
  return py::make_tuple(obs.v, reward, terminal, oss.str());
}

void PyEnv::fini()
{
  if (!env_)
    std::cerr << "Not initialized." << std::endl;

  safe_delete(&configurator_);
  env_ = NULL;
  first_=true;
}

void PyEnv::report(int idx)
{
  report_idx_ = idx;
}


PYBIND11_MODULE(py_env, m)
{
  py::class_<PyEnv> py_env_class(m, "PyEnv");
  py_env_class
      .def(py::init<>())
      .def("init",  &PyEnv::init, "Initialize envirnoment", py::return_value_policy::copy,
           py::arg("file").none(false), py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>())
      .def("seed",  &PyEnv::seed, "Seed envirnoment", py::arg("seed").none(false),
           py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>())
      .def("start", &PyEnv::start, "Start envirnoment", py::return_value_policy::copy,
           py::arg("test").none(false), py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>())
      .def("step",  &PyEnv::step, "Start envirnoment", py::return_value_policy::copy,
           py::arg("py_action").noconvert(), py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>())
      .def("fini",  &PyEnv::fini, "Finish envirnoment",
           py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>())
      .def("report",  &PyEnv::report, "Report in episodes ['learn', 'test', 'all']", py::arg("idx").none(false),
           py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>());
}
