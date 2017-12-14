/** \file py_env.h
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
 
#ifndef GRL_PY_ENVIRONMENT_H_
#define GRL_PY_ENVIRONMENT_H_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <grl/grl.h>
#include <grl/configuration.h>
#include <grl/environment.h>

namespace grl
{

class PyEnv
{
  public:
    Configurator *configurator_;
    Environment *env_;
    int action_dims_;
    int observation_dims_;
    bool started_;
    bool first_;

  public:
    PyEnv() : configurator_(NULL), env_(NULL), action_dims_(0), observation_dims_(0), started_(false), first_(true) { }
  
    pybind11::tuple init(const std::string &file);
    void seed(int seed);
    Vector start(int test);
    pybind11::tuple step(const Vector &action);
    void fini();
};

}

#endif /* GRL_PY_ENVIRONMENT_H_ */
