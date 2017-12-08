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

#include <string.h>
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include <grl/grl.h>

#include <grl/configuration.h>
#include <grl/environment.h>

using namespace grl;

namespace py = pybind11;

static Configurator *g_configurator=NULL;
static Environment *g_env=NULL;
static int g_action_dims=0;
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

  g_action_dims = (*envconf)[g_path+"action_dims"];
  g_started = false;
}

py::array_t<double, py::array::c_style> py_start(int test)
{
  if (!g_env)
    std::cerr << "Not initialized." << std::endl;

  // Run environment
  Observation obs;

  g_env->start(test, &obs);
  g_started = true;

  //std::cout << obs << std::endl;

  // Process output
  std::vector<ssize_t> shape { obs.size() };
  return py::array(shape, obs.v.data());
}

/*
void py_step()
{
  py::scoped_estream_redirect output;

  if (!g_env)
    std::cerr << "Not initialized." << std::endl;

  Action action;

  if (!g_started)
    std::cerr << "Environment not started.");

  // Verify input
  if (nrhs < 2 || !mxIsDouble(prhs[1]))
    std::cerr << "Missing action.");

  // Prepare input
  int elements = mxGetNumberOfElements(prhs[1]);

  if (elements != g_action_dims)
    mexErrMsgTxt("Invalid action size.");

  action.v.resize(elements);
  for (size_t ii=0; ii < elements; ++ii)
    action[ii] = mxGetPr(prhs[1])[ii];

  // Run environment
  Observation obs;
  double reward;
  int terminal;
  double tau = g_env->step(action, &obs, &reward, &terminal);

  // Process output
  plhs[0] = vectorToArray(obs);
  if (nlhs > 1)
    plhs[1] = mxCreateDoubleScalar(reward);
  if (nlhs > 2)
    plhs[2] = mxCreateDoubleScalar(terminal);
  if (nlhs > 3)
    plhs[3] = mxCreateDoubleScalar(tau);
}
*/

void py_fini()
{
  if (!g_env)
    std::cerr << "Not initialized." << std::endl;

  safe_delete(&g_configurator);
  g_env = NULL;
  g_first=true;
}

/*
void py_envv(int nlhs, mxArray *plhs[ ],
                 int nrhs, const mxArray *prhs[ ])
{
  MexMemString func;
  static bool first=true;
  
  if (first)
  {
    loadPlugins();
    first = false;
  }

  if (nrhs < 1 || !mxIsChar(prhs[0]) || !(func = mxArrayToString(prhs[0])))
    mexErrMsgTxt("Missing function name.");

  if (!strcmp(func, "init"))
  {
    MexMemString file;
  
    if (g_env)
      mexErrMsgTxt("Already initialized.");
      
    if (nrhs < 2 || !mxIsChar(prhs[1]) || !(file = mxArrayToString(prhs[1])))
      mexErrMsgTxt("Missing configuration file name.");
      
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
      mexErrMsgTxt("Configuration file does not specify a valid environment.");
    }

    plhs[0] = taskSpecToStruct(*envconf);
    
    g_action_dims = mxGetPr(mxGetField(plhs[0], 0, "action_dims"))[0];
    g_started = false;

    mexLock();

    return;
  }
  
  if (!g_env)
    mexErrMsgTxt("Not initialized.");

  if (!strcmp(func, "fini"))
  {
    safe_delete(&g_configurator);
    g_env = NULL;
    mexUnlock();
  }
  else if (!strcmp(func, "start"))
  {
    // Run environment
    Observation obs;
    
    // TODO: READ TEST ARGUMENT
    g_env->start(0, &obs);
    g_started = true;
    
    // Process output
    plhs[0] = vectorToArray(obs);
  }
  else if (!strcmp(func, "step"))
  {
    Action action;
    
    if (!g_started)
      mexErrMsgTxt("Environment not started.");

    // Verify input    
    if (nrhs < 2 || !mxIsDouble(prhs[1]))
      mexErrMsgTxt("Missing action.");
      
    // Prepare input
    int elements = mxGetNumberOfElements(prhs[1]);
    
    if (elements != g_action_dims)
      mexErrMsgTxt("Invalid action size.");
    
    action.v.resize(elements);
    for (size_t ii=0; ii < elements; ++ii)
      action[ii] = mxGetPr(prhs[1])[ii];
    
    // Run environment
    Observation obs;
    double reward;
    int terminal;
    double tau = g_env->step(action, &obs, &reward, &terminal);
    
    // Process output
    plhs[0] = vectorToArray(obs);
    if (nlhs > 1) 
      plhs[1] = mxCreateDoubleScalar(reward);
    if (nlhs > 2)
      plhs[2] = mxCreateDoubleScalar(terminal);
    if (nlhs > 3)
      plhs[3] = mxCreateDoubleScalar(tau);
  }
  else
    mexErrMsgTxt("Unknown command.");
}
*/

PYBIND11_MODULE(py_env, m)
{
    m.doc() = "pybind11 plugin for GRL";
    m.def("init", &py_init, "Initialize envirnoment", py::arg("file"),
          py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>());
    m.def("start", &py_start, "Start envirnoment", py::arg("test") = 0,
          py::return_value_policy::copy,
          py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>());
}
