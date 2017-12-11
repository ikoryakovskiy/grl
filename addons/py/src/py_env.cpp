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

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <string.h>

#include <grl/grl.h>
#include <grl/configuration.h>
#include <grl/environment.h>

using namespace grl;

namespace py = pybind11;

static Configurator *g_configurator=NULL;
static Environment *g_env=NULL;
static int g_action_dims=0;
static int g_observation_dims=0;
static bool g_started=false;
static std::string g_path;
static bool g_first=true;

void py_fini();

void py_init(const std::string &file)
{
  if (g_first)
  {
    loadPlugins();
    g_first = false;
  }

  if (g_env)
  {
    std::cerr << "Re-initializing." << std::endl;
    py_fini();
  }

  if (file == "")
    std::cerr << "Missing configuration file name." << std::endl;

  Configurator *conf, *envconf;
  if (!(conf = loadYAML(file)) || !(g_configurator = conf->instantiate()) || !(envconf = g_configurator->find("environment")))
  {
    safe_delete(&conf);
    safe_delete(&g_configurator);
    return;
  }

  safe_delete(&conf);
  g_env = dynamic_cast<Environment*>(envconf->ptr());

  if (!g_env)
  {
    safe_delete(&g_configurator);
    std::cerr << "Configuration file does not specify a valid environment." << std::endl;
  }

  if (envconf->find("observation_min"))
    g_path = "";
  else if (envconf->find("task/observation_min"))
    g_path = "task/";
  else
    std::cerr << "Could not determine task specification." << std::endl;

  g_observation_dims = (*envconf)[g_path+"observation_dims"];
  g_action_dims = (*envconf)[g_path+"action_dims"];

  std::cout << "Observation dims: " << g_observation_dims << std::endl;
  std::cout << "Action dims: " << g_action_dims << std::endl;

  g_started = false;
}

Vector py_start(int test)
{
  if (!g_env)
    std::cerr << "Not initialized." << std::endl;

  // Run environment
  Observation obs;

  g_env->start(test, &obs);
  g_started = true;


  // Process output
  return obs.v;
}

py::tuple py_step(const Vector &action)
{
  if (!g_env)
    std::cerr << "Not initialized." << std::endl;

  if (!g_started)
    std::cerr << "Environment not started." << std::endl;

  if (action.size() != g_action_dims)
    std::cerr << "Invalid action size." << std::endl;

  // Run environment
  Observation obs;
  double reward;
  int terminal;
  double tau = g_env->step(action, &obs, &reward, &terminal);

  // Process output
  return py::make_tuple(obs.v, reward, terminal, tau);
}

void py_fini()
{
  if (!g_env)
    std::cerr << "Not initialized." << std::endl;

  safe_delete(&g_configurator);
  g_env = NULL;
  g_first=true;
}

PYBIND11_MODULE(py_env, m)
{
    m.doc() = "pybind11 plugin for GRL";
    m.def("init", &py_init, "Initialize envirnoment", py::arg("file").none(false),
          py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>());
    m.def("start", &py_start, "Start envirnoment", py::arg("test").none(false) = 0,
          py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>());
    m.def("step", &py_step, "Start envirnoment", py::arg("py_action").noconvert(),
          py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>());
    m.def("fini", &py_fini, "Finish envirnoment",
          py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>());
}
