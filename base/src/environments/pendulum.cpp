/** \file pendulum.cpp
 * \brief Pendulum environment source file.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2015-01-22
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

#include <grl/environments/pendulum.h>

using namespace grl;

REGISTER_CONFIGURABLE(PendulumDynamics)
REGISTER_CONFIGURABLE(PendulumSwingupTask)
REGISTER_CONFIGURABLE(PendulumRegulatorTask)

void PendulumDynamics::request(ConfigurationRequest *config)
{
  config->push_back(CRP("param", "Model parameters", std::string(""), CRP::Configuration));
}

void PendulumDynamics::configure(Configuration &config)
{
  J_ = 0.000191;
  m_ = 0.055;
  g_ = 9.81;
  l_ = 0.042;
  b_ = 0.000003;
  K_ = 0.0536;
  R_ = 9.5;

  std::string param = config["param"];

  if (config["param"].str() == "random")
  {
    J_ += 0.1*J_*RandGen::getUniform(-1.0, 1.0);
    m_ += 0.1*m_*RandGen::getUniform(-1.0, 1.0);
    l_ += 0.1*l_*RandGen::getUniform(-1.0, 1.0);
    b_ += 0.1*b_*RandGen::getUniform(-1.0, 1.0);
    K_ += 0.1*K_*RandGen::getUniform(-1.0, 1.0);
    R_ += 0.1*R_*RandGen::getUniform(-1.0, 1.0);
  }
  else
  {
    int pos = param.find_last_of(':');
    if (pos != std::string::npos)
    {
      std::string file = param.substr(0, pos);
      std::string sn = param.substr(pos+1, param.length()-pos);

      char* p = NULL;
      long n = strtol(sn.c_str(), &p, 10);
      if (p == NULL)
      {
        // load file and read parameters from the specified line
        std::ifstream infile(file);

        std::string line;
        int i = 0;
        while (std::getline(infile, line))
        {
          if (i++ == n)
          {
            std::istringstream iss(line);
            iss >> J_ >> m_ >> l_ >> b_ >> K_ >> R_;
            break;
          }
        }
      }
    }
  }
}

void PendulumDynamics::reconfigure(const Configuration &config)
{
}

void PendulumDynamics::eom(const Vector &state, const Vector &actuation, Vector *xd) const
{
  if (state.size() != 3 || actuation.size() != 1)
    throw Exception("dynamics/pendulum requires a task/pendulum subclass");

  double a   = state[0];
  double ad  = state[1];
  double add = (1/J_)*(m_*g_*l_*sin(a)-b_*ad-(K_*K_/R_)*ad+(K_/R_)*actuation[0]);
  
  xd->resize(3);
  (*xd)[0] = ad;
  (*xd)[1] = add;
  (*xd)[2] = 1;
}

void PendulumSwingupTask::request(ConfigurationRequest *config)
{
  Task::request(config);

  config->push_back(CRP("timeout", "Episode timeout", T_, CRP::Configuration, 0., DBL_MAX));
  config->push_back(CRP("randomization", "Level of start state randomization", randomization_, CRP::Configuration, 0., 1.));
  config->push_back(CRP("wrap_angle", "Wrap angle of the pendulum", wrap_angle_, CRP::Configuration, 0, 1));
}

void PendulumSwingupTask::configure(Configuration &config)
{
  T_ = config["timeout"];
  randomization_ = config["randomization"];
  wrap_angle_ = config["wrap_angle"];

  config.set("observation_dims", 2);
  config.set("action_dims", 1);
  config.set("action_min", VectorConstructor(-3));
  config.set("action_max", VectorConstructor(3));
  config.set("reward_max", 0);

  if (wrap_angle_)
  {
    config.set("observation_min", VectorConstructor(0., -12*M_PI));
    config.set("observation_max", VectorConstructor(2*M_PI, 12*M_PI));
    config.set("reward_min", -5*pow(M_PI, 2) - 0.1*pow(12*M_PI, 2) - 1*pow(3, 2));
  }
  else
  {
    config.set("observation_min", VectorConstructor(-M_PI, -12*M_PI));
    config.set("observation_max", VectorConstructor(2*M_PI, 12*M_PI));
    config.set("reward_min", -5*pow(2*M_PI, 2) - 0.1*pow(12*M_PI, 2) - 1*pow(3, 2));
  }
}

void PendulumSwingupTask::reconfigure(const Configuration &config)
{
}

void PendulumSwingupTask::start(int test, Vector *state) const
{
  state->resize(3);
  (*state)[0] = M_PI+randomization_*(test==0)*RandGen::get()*2*M_PI;
  (*state)[1] = 0;
  (*state)[2] = 0;
}

void PendulumSwingupTask::observe(const Vector &state, Observation *obs, int *terminal) const
{
  if (state.size() != 3)
    throw Exception("task/pendulum/swingup requires dynamics/pendulum");

  double a = state[0];
  if (wrap_angle_)
  {
    a = fmod(state[0]+M_PI, 2*M_PI);
    if (a < 0) a += 2*M_PI;
  }

  obs->v.resize(2);
  (*obs)[0] = a;
  (*obs)[1] = state[1];
  obs->absorbing = false;
  
  if (wrap_angle_)
  {
    if (state[0] > 2*M_PI || state[0] < -M_PI)
      *terminal = 2;
  }
  else if (state[1] > 20 || state[1] < -20) // #divyam For DDPG
  {
    *terminal = 2;
    obs->absorbing = true;
  }
  else if (state[2] > T_)
    *terminal = 1;
  else
    *terminal = 0;
}

void PendulumSwingupTask::evaluate(const Vector &state, const Action &action, const Vector &next, double *reward) const
{
  if (state.size() != 3 || action.size() != 1 || next.size() != 3)
    throw Exception("task/pendulum/swingup requires dynamics/pendulum");

  double a = next[0];
  if (wrap_angle_)
  {
    a = fmod(fabs(next[0]), 2*M_PI);
    if (a > M_PI) a -= 2*M_PI;
  }

  // For DDPG:
  // *reward = -5*pow(a, 2) - .1*pow(b, 2) - .01*pow(action[0], 2);
  *reward = -5*pow(a, 2) - 0.1*pow(next[1], 2) - 1*pow(action[0], 2);
}

bool PendulumSwingupTask::invert(const Observation &obs, Vector *state) const
{
  *state = obs;
  if (wrap_angle_)
    (*state)[0] -= M_PI;
  *state = extend(*state, VectorConstructor(0.));
  
  return true;
}

// Regulator

void PendulumRegulatorTask::request(ConfigurationRequest *config)
{
  RegulatorTask::request(config);
}

void PendulumRegulatorTask::configure(Configuration &config)
{
  RegulatorTask::configure(config);
  
  if (q_.size() != 2)
    throw bad_param("task/pendulum/regulator:q");
  if (r_.size() != 1)
    throw bad_param("task/pendulum/regulator:r");

  config.set("observation_min", VectorConstructor(-M_PI, -12*M_PI));
  config.set("observation_max", VectorConstructor( M_PI,  12*M_PI));
  config.set("action_min", VectorConstructor(-3));
  config.set("action_max", VectorConstructor(3));
}

void PendulumRegulatorTask::reconfigure(const Configuration &config)
{
  RegulatorTask::reconfigure(config);
}

void PendulumRegulatorTask::observe(const Vector &state, Observation *obs, int *terminal) const
{
  if (state.size() != 3)
    throw Exception("task/pendulum/regulator requires dynamics/pendulum");
    
  obs->v.resize(2);
  for (size_t ii=0; ii < 2; ++ii)
    (*obs)[ii] = state[ii];
  obs->absorbing = false;

  *terminal = state[2] > 3;
}

bool PendulumRegulatorTask::invert(const Observation &obs, Vector *state) const
{
  *state = extend(obs, VectorConstructor(0.));
  
  return true;
}
