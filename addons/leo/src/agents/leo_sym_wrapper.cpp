/** \file leo_sym_wrapper.cpp
 * \brief Leo agent wrapper source file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-02-07
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

#include <grl/agents/leo_sym_wrapper.h>
#include <leo.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoSymWrapperAgent)

#define TARGET_OBSERVATION_SIZE 10
#define TARGET_ACTION_SIZE       3

void LeoSymWrapperAgent::request(ConfigurationRequest *config)
{
  config->push_back(CRP("agent", "agent", "Target agent with reduced state-action space due to symmetry", agent_));
  config->push_back(CRP("sub_ic_signal", "signal/vector", "Publisher of the initialization and contact signal", sub_ic_signal_));
}

void LeoSymWrapperAgent::configure(Configuration &config)
{
  agent_ = (Agent*)config["agent"].ptr();
  sub_ic_signal_ = (VectorSignal*)config["sub_ic_signal"].ptr();
}

void LeoSymWrapperAgent::reconfigure(const Configuration &config)
{
}

void LeoSymWrapperAgent::start(const Observation &obs, Action *action)
{
  Observation obs_agent = ConstantVector(TARGET_OBSERVATION_SIZE, 0);
  Action act_agent = ConstantVector(TARGET_ACTION_SIZE, 0);

  action->v.resize(ljNumJoints);

  int stl = stanceLegLeft();
  parseStateForAgent(obs, &obs_agent, stl);
  agent_->start(obs_agent, &act_agent);
  parseActionForEnvironment(act_agent, obs, action, stl);
}

void LeoSymWrapperAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  Observation obs_agent = ConstantVector(TARGET_OBSERVATION_SIZE, 0);
  Action act_agent = ConstantVector(TARGET_ACTION_SIZE, 0);

  int stl = stanceLegLeft();
  parseStateForAgent(obs, &obs_agent, stl);
  agent_->step(tau, obs_agent, reward, &act_agent);
  parseActionForEnvironment(act_agent, obs, action, stl);
}

void LeoSymWrapperAgent::end(double tau, const Observation &obs, double reward)
{
  Observation obs_agent = ConstantVector(TARGET_OBSERVATION_SIZE, 0);

  int stl = stanceLegLeft();
  parseStateForAgent(obs, &obs_agent, stl);
  agent_->end(tau, obs_agent, reward);
}

int LeoSymWrapperAgent::stanceLegLeft() const
{
  double stl = 0;
  Vector signal = sub_ic_signal_->get();
  if ((int)signal[0] & lstStanceLeft)
    stl = 1;

  // update signal because of symmetrical state
  if ((int)signal[0] & lstSwlTouchDown)
  {
    double contact = signal[0];
    Vector ti_actuator = VectorConstructor(-1, 1, 0, 2, 2, -1, -1);
    Vector ti_actuator_sym = VectorConstructor(-1, 0, 1, 2, 2, -1, -1);

    signal.resize(1+ti_actuator.size()+ti_actuator_sym.size());
    if (stl)
      signal << contact, ti_actuator, ti_actuator_sym;
    else
      signal << contact, ti_actuator_sym, ti_actuator;
    sub_ic_signal_->set(signal);

    INFO("LeoSymWrapperAgent : TouchDown");
  }

  return stl;
}

void LeoSymWrapperAgent::parseStateForAgent(const Observation &obs, Observation *obs_agent, int stl) const
{
  // environment
  //   obs: hipleft, hipright, kneeleft, kneeright, ankleleft, ankleright, shoulder, torso_boom
  //   obs_agent: hipleft, hipright, kneeleft, kneeright, shoulder

  CRAWL(obs);

  (*obs_agent)[ljTorso] = obs[ljTorso];
  (*obs_agent)[TARGET_OBSERVATION_SIZE/2+ljTorso] = obs[ljNumJoints+ljTorso];

  if (stl)
  {
    (*obs_agent)[ljHipLeft]   = obs[ljHipLeft];
    (*obs_agent)[ljHipRight]  = obs[ljHipRight];
    (*obs_agent)[ljKneeLeft]  = obs[ljKneeLeft];
    (*obs_agent)[ljKneeRight] = obs[ljKneeRight];

    (*obs_agent)[TARGET_OBSERVATION_SIZE/2+ljHipLeft]   = obs[ljNumJoints+ljHipLeft];
    (*obs_agent)[TARGET_OBSERVATION_SIZE/2+ljHipRight]  = obs[ljNumJoints+ljHipRight];
    (*obs_agent)[TARGET_OBSERVATION_SIZE/2+ljKneeLeft]  = obs[ljNumJoints+ljKneeLeft];
    (*obs_agent)[TARGET_OBSERVATION_SIZE/2+ljKneeRight] = obs[ljNumJoints+ljKneeRight];
  }
  else
  {
    (*obs_agent)[ljHipRight]  = obs[ljHipLeft];
    (*obs_agent)[ljHipLeft]   = obs[ljHipRight];
    (*obs_agent)[ljKneeRight] = obs[ljKneeLeft];
    (*obs_agent)[ljKneeLeft]  = obs[ljKneeRight];

    (*obs_agent)[TARGET_OBSERVATION_SIZE/2+ljHipRight]  = obs[ljNumJoints+ljHipLeft];
    (*obs_agent)[TARGET_OBSERVATION_SIZE/2+ljHipLeft]   = obs[ljNumJoints+ljHipRight];
    (*obs_agent)[TARGET_OBSERVATION_SIZE/2+ljKneeRight] = obs[ljNumJoints+ljKneeLeft];
    (*obs_agent)[TARGET_OBSERVATION_SIZE/2+ljKneeLeft]  = obs[ljNumJoints+ljKneeRight];
  }

  obs_agent->absorbing = obs.absorbing;

  TRACE(*obs_agent);
}

void LeoSymWrapperAgent::parseActionForEnvironment(const Action &act_agent, const Observation &obs, Action *action, int stl) const
{
  // agent
  //   observe: hipleft, hipright, kneeleft, kneeright, ankleleft, ankleright, shoulder, torso_boom
  //   actuate: hipleft, hipright, kneeleft, kneeright, ankleleft, ankleright, shoulder

  TRACE(act_agent);

  if (stl)
  {
    (*action)[ljHipLeft]   = act_agent[0];
    (*action)[ljHipRight]  = act_agent[1];
    (*action)[ljKneeLeft]  = autoActuateKnees(obs[ljKneeLeft]); // auto actuate stance leg, which is left now
    (*action)[ljKneeRight] = act_agent[2];
  }
  else
  {
    (*action)[ljHipLeft]   = act_agent[1];
    (*action)[ljHipRight]  = act_agent[0];
    (*action)[ljKneeLeft]  = act_agent[2];
    (*action)[ljKneeRight] = autoActuateKnees(obs[ljKneeRight]); // auto actuate stance leg, which is right now
  }

  double actionAnkleLeft, actionAnkleRight;
  autoActuateAnkles_FixedPos(obs[ljAnkleLeft],  &actionAnkleLeft,
                             obs[ljAnkleRight], &actionAnkleRight);
  (*action)[ljAnkleLeft] = actionAnkleLeft;
  (*action)[ljAnkleRight] = actionAnkleRight;

  (*action)[ljShoulder] = autoActuateArm(obs[ljShoulder]);

  action->type = act_agent.type;

  CRAWL(*action);
}

double LeoSymWrapperAgent::autoActuateArm(double armObs) const
{
  double armTorque = 5.0*(preProgShoulderAngle_ - armObs);
  const double torqueToVoltage  = 14.0/3.3;
  return torqueToVoltage*armTorque;
}

double LeoSymWrapperAgent::autoActuateKnees(double stanceKneeObs) const
{
  double kneeStanceTorque = 5.0*(preProgStanceKneeAngle_ - stanceKneeObs);
  const double torqueToVoltage = 14.0/3.3;
  return torqueToVoltage*kneeStanceTorque;
}

void LeoSymWrapperAgent::autoActuateAnkles_FixedPos(double leftAnkleObs, double *leftAnkleAction, double rightAnkleObs, double *rightAnkleAction) const
{
  double K = 10.0;
  double leftAnkleTorque    = K*(preProgAnkleAngle_ - leftAnkleObs);
  double rightAnkleTorque   = K*(preProgAnkleAngle_ - rightAnkleObs);

  const double torqueToVoltage  = 14.0/3.3;
  *leftAnkleAction = leftAnkleTorque*torqueToVoltage;
  *rightAnkleAction = rightAnkleTorque*torqueToVoltage;
}
