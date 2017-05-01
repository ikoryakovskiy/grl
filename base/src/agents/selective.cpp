/** \file selective.cpp
 * \brief Selective master agent source file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-04-04
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

#include <grl/agents/selective.h>
#include <cmath>
#include <iomanip>

using namespace grl;

REGISTER_CONFIGURABLE(SelectiveMasterAgent)

// *** SelectiveMasterAgent ***

void SelectiveMasterAgent::request(ConfigurationRequest *config)
{
  config->push_back(CRP("agent1", "agent/sub", "First subagent", agents_[0]));
  config->push_back(CRP("agent2", "agent/sub", "Second subagent", agents_[1]));
  config->push_back(CRP("sub_sigma_signal", "signal/vector", "Subscriber to external sigma", sub_sigma_signal_, true));
}

void SelectiveMasterAgent::configure(Configuration &config)
{
  agents_[0] = (SubAgent*)config["agent1"].ptr();
  agents_[1] = (SubAgent*)config["agent2"].ptr();
  sub_sigma_signal_ = (VectorSignal*)config["sub_sigma_signal"].ptr();
}

void SelectiveMasterAgent::reconfigure(const Configuration &config)
{

}

void SelectiveMasterAgent::report(std::ostream &os)
{  
  const int pw = 15;
  std::stringstream progressString;
  progressString << std::fixed << std::setprecision(3) << std::right;

  //std::cout << "   " << &total_rewards_ << std::endl;

  // append cumulative reward in case of timeout termination
  if (total_reward_ != 0)
    total_rewards_.push_back(total_reward_);

  int max_size = 6;
  int size = std::min(max_size, static_cast<int>(total_rewards_.size()));

  for (int i = 0; i < size; i++)
    progressString << std::setw(pw) << total_rewards_[i];

  for (int i = size; i < max_size; i++)
    progressString << std::setw(pw) << std::numeric_limits<double>::quiet_NaN();

  // Sigma used by the combination
  // TODO: Ask Wouter if report() in Policy is possible
  if (sub_sigma_signal_)
  {
    Vector smart_sigma = sub_sigma_signal_->get();
    progressString << std::setw(pw) << smart_sigma[0];
  }

  os << progressString.str();
}

void SelectiveMasterAgent::start(const Observation &obs, Action *action)
{
  current_idx_ = selectSubAgent(0, obs, action);

  current_agent_ = agents_[current_idx_];
  current_agent_->start(obs, action);

  time_ = 0;
  total_reward_ = 0;
  total_rewards_.clear();
}

void SelectiveMasterAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  time_ += tau;
  int idx = selectSubAgent(time_, obs, action);
  stepSubAgent(idx, tau, obs, reward, action);
}

void SelectiveMasterAgent::end(double tau, const Observation &obs, double reward)
{
  current_agent_->end(tau, obs, reward);
  total_reward_ += reward;
  total_rewards_.push_back(total_reward_);
  total_reward_ = 0;
}

size_t SelectiveMasterAgent::selectSubAgent(double time, const Observation &obs, Action *action)
{
  // Find most confident agent
  double maxconf = agents_[0]->confidence(obs);
  size_t maxconfa = 0;

  for (size_t ii=1; ii < agents_.size(); ++ii)
  {
    double confidence = agents_[ii]->confidence(obs);
    if (confidence > maxconf)
    {
      maxconf = confidence;
      maxconfa = ii;
    }
  }
  return maxconfa;
}

void SelectiveMasterAgent::stepSubAgent(int idx, double tau, const Observation &obs, double reward, Action *action)
{
  SubAgent *agent = agents_[idx];
  if (current_agent_ != agent)
  {
    current_agent_->end(tau, obs, reward);          // finish previous agent
    current_agent_ = agent;                         // switch to the new agent
    current_agent_->start(obs, action);             // start it to obtain action
    TRACE("Changing subAgents");
  }
  else
    current_agent_->step(tau, obs, reward, action); // or simply continue

  // record rewards based on changes of the index
  // therefore, also allow to use the same agent, but record it's performance at each stage
  total_reward_ += reward;
  if (current_idx_ != idx)
  {
    total_rewards_.push_back(total_reward_);
    total_reward_ = 0;
    current_idx_ = idx;
  }
}
