/** \file leo_sma.cpp
 * \brief State-machine agent source file for Leo
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

#include <sys/resource.h>
#include <errno.h>
#include <grl/agents/leo_sma.h>
#include <leo.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoStateMachineAgent)

struct th_data
{
  th_data() : agent(NULL), tt(0), output(""), completed(true), quit(false) {}
  Agent *agent;
  int tt;
  std::string output;
  bool completed;
  bool quit;
};

th_data save_data;
pthread_cond_t save_cond;
pthread_mutex_t save_mtx;

void *save_thread(void*)
{
  // use "sudo nice -n -20 ./grld ..." for best performance, takes 1.55s. Otherwice it takes 3.78s
  if (setpriority(PRIO_PROCESS, 0, 19) == -1)
  {
    ERROR("agent/leo/sma: failed to lower priority of the saving thread");
    throw Exception(std::strerror(errno));
  }

  while (1)
  {
    pthread_mutex_lock(&save_mtx); // lock the mutex
    if (save_data.quit)
    {
        pthread_mutex_unlock(&save_mtx);
        break;
    }
    save_data.completed = true;
    pthread_cond_wait(&save_cond, &save_mtx); // wait for save signal
    if (save_data.quit)
    {
        pthread_mutex_unlock(&save_mtx);
        break;
    }
    th_data data = save_data;
    pthread_mutex_unlock(&save_mtx); // unlock the mutex

    // give PID controller to stabalize Leo before we start saving
    sleep(4);

    // saving policy
    std::ostringstream oss;
    oss << data.output << "-" << data.tt << "-";
    Configuration saveconfig;
    saveconfig.set("action", "save");
    saveconfig.set("file", oss.str().c_str());
    data.agent->walk(saveconfig);

    // give some time OS to finalize writing
    sleep(1);
  }
  return NULL;
}

void LeoStateMachineAgent::request(ConfigurationRequest *config)
{
  config->push_back(CRP("environment", "environment", "Environment in which the agent acts", environment_, true));

  config->push_back(CRP("main_steps", "Number of time steps for the main agent to operate", (int)steps_, CRP::Configuration, 0, INT_MAX));
  config->push_back(CRP("main_timeout", "Number of trials to take (switched to agent_prepare after the timout or fail)", (double)timeout_, CRP::Configuration, 0.0, DBL_MAX));
  config->push_back(CRP("test_interval", "Number of episodes after which testing happens of the main controller happens", (int)test_interval_, CRP::Configuration, -1, INT_MAX));
  config->push_back(CRP("output", "Output base filename", output_));

  LeoBaseAgent::request(config);
  config->push_back(CRP("agent_prepare", "agent", "Prepare agent", agent_prepare_.a, false));
  config->push_back(CRP("agent_standup", "agent", "Safe standup agent", agent_standup_.a, false));
  config->push_back(CRP("agent_starter", "agent", "Starting agent", agent_starter_.a, true));
  config->push_back(CRP("agent_main", "agent", "Main learning agent", agent_main_.a, false));
  config->push_back(CRP("agent_test", "agent", "Main testing agent", agent_test_.a, true));

  config->push_back(CRP("upright_trigger", "trigger", "Trigger which finishes stand-up phase and triggers preparation agent", upright_trigger_, false));
  config->push_back(CRP("feet_on_trigger", "trigger", "Trigger which checks for foot contact to ensure that robot is prepared to walk", feet_on_trigger_, true));
  config->push_back(CRP("feet_off_trigger", "trigger", "Trigger which checks for foot contact to detect lifts of the robot", feet_off_trigger_, true));
  config->push_back(CRP("starter_trigger", "trigger", "Trigger which initiates a preprogrammed walking at the beginning", starter_trigger_, true));

  config->push_back(CRP("pub_sma_state", "signal/vector", "Publisher of the type of the agent currently used by state machine", pub_sma_state_, true));
}

void LeoStateMachineAgent::configure(Configuration &config)
{ 
  environment_ = (Environment*)config["environment"].ptr();

  steps_ = config["main_steps"];
  timeout_ = config["main_timeout"];
  test_interval_ = config["test_interval"];
  output_ = config["output"].str();

  LeoBaseAgent::configure(config);

  agent_prepare_.a = (Agent*)config["agent_prepare"].ptr();
  agent_prepare_.s = SMA_PREPARE;
  agent_standup_.a = (Agent*)config["agent_standup"].ptr();
  agent_standup_.s = SMA_STANDUP;
  agent_starter_.a = (Agent*)config["agent_starter"].ptr();
  agent_starter_.s = SMA_STARTER;
  agent_main_.a = (Agent*)config["agent_main"].ptr();
  agent_main_.s = SMA_MAIN;
  agent_test_.a = (Agent*)config["agent_test"].ptr();
  agent_test_.s = SMA_TEST;

  if (test_interval_ >= 0 && !agent_test_.a)
    throw bad_param("agent/leo/sma:agent_test");

  upright_trigger_ = (Trigger*)config["upright_trigger"].ptr();
  feet_on_trigger_ = (Trigger*)config["feet_on_trigger"].ptr();
  feet_off_trigger_ = (Trigger*)config["feet_off_trigger"].ptr();
  starter_trigger_ = (Trigger*)config["starter_trigger"].ptr();

  pub_sma_state_ = (VectorSignal*)config["pub_sma_state"].ptr();

  if (!output_.empty())
  {
    if (pthread_cond_init(&save_cond, NULL))
      throw Exception("agent/leo/sma: cannot initialize condition");
    thread_ = new pthread_t();
    if (pthread_create(thread_, NULL, save_thread, NULL))
      throw Exception("agent/leo/sma cannot create thread");
  }
}

void LeoStateMachineAgent::reconfigure(const Configuration &config)
{
}

LeoStateMachineAgent::~LeoStateMachineAgent()
{
  if (ofs_.is_open())
    ofs_.close();

  if (thread_)
  {
    pthread_mutex_lock(&save_mtx); // lock the mutex
    save_data.quit = true;
    if (save_data.completed)
      pthread_cond_signal(&save_cond); // request quit
    pthread_mutex_unlock(&save_mtx); // unlock the mutex
    pthread_join(*thread_, NULL);
    pthread_cond_destroy(&save_cond);
    grl::safe_delete(&thread_);
  }
}

void LeoStateMachineAgent::start(const Observation &obs, Action *action)
{
  time_ = 0.;
  int touchDown, groundContact, stanceLegLeft;
  unpack_ic(&touchDown, &groundContact, &stanceLegLeft);
  if (failed(obs))
    agent_ = agent_standup_; // standing up from lying position
  else
    agent_ = agent_prepare_; // prepare from hanging position

  // clear sandbox evnironment history
  if (environment_)
  {
    Configuration config;
    config.set("action", "statclr");
    config.set("sma_state", agent_.s);
    environment_->reconfigure(config);
  }

  // start agent
  agent_.a->start(obs, action);

  for (int i = 0; i < action_max_.size(); i++)
    (*action)[i] = fmin(action_max_[i], fmax((*action)[i], action_min_[i]));

  if (pub_sma_state_)
    pub_sma_state_->set(VectorConstructor(agent_.s));

  if (!output_.empty())
  {
    std::ostringstream oss;
    oss << output_ << ".txt";
    ofs_.open(oss.str().c_str());
  }
}

void LeoStateMachineAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  time_ += tau;
  main_total_reward_ += reward;

  act(tau, obs, reward, action);

  for (int i = 0; i < action_max_.size(); i++)
    (*action)[i] = fmin(action_max_[i], fmax((*action)[i], action_min_[i]));

  if (pub_sma_state_)
    pub_sma_state_->set(VectorConstructor(agent_.s));
}

void LeoStateMachineAgent::end(double tau, const Observation &obs, double reward)
{
  std::cout << "End should not be called here!" << std::endl;
}

void LeoStateMachineAgent::act(double tau, const Observation &obs, double reward, Action *action)
{
  // obtain contact information for symmetrical switching
  // note that groundContact is not reliable when Leo is moving,
  // but is reliable when Leo is standing still.
  // Therefore we use a trigger to make sure the contact is lost over some amount of time.
  // Also, note that if "sub_ic_signal_" is not used, then robot is assumed to be on the ground (i.e. ready for operation)
  int touchDown = 0, groundContact = 1, stanceLegLeft = 1;
  unpack_ic(&touchDown, &groundContact, &stanceLegLeft);
  Vector gc = VectorConstructor(groundContact);

  // if Leo looses ground contact
  if (feet_off_trigger_ && feet_off_trigger_->check(time_, gc))
  {
    if (failed(obs))
      // lost contact due to fall => try to standup
      set_agent(agent_standup_, tau, obs, reward, action, "Lost ground contact, need to stand up!");
    else
    {
      // lost contact due to lift => move prepare to continue walking
      if (agent_ != agent_standup_) // wait for standing up if needed
        set_agent(agent_prepare_, tau, obs, reward, action, "Lost ground contact, already upright!");
    }
  }

  // if Leo fell down and we are not trying to stand up already, then try!
  if (failed(obs))
    return set_agent(agent_standup_, tau, obs, reward, action, "Main terminated due to fall. Leo needs to standup.");

  // if timeout
  if ((agent_ == agent_main_ || agent_ == agent_test_) && (time_ - main_time_ > timeout_))
    return set_agent(agent_prepare_, tau, obs, reward, action, "Main terminated due to timeout. Prepare for the new episode.");

  if (agent_ == agent_prepare_)
  {
    // if Leo is in the upright position wait for the contact before we start the starter
    // or the main agent
    if (feet_on_trigger_->check(time_, gc) && save_completed())
    {
      if (agent_starter_.a && starter_trigger_ && !starter_trigger_->check(time_, Vector()))
        return set_agent(agent_starter_, tau, obs, reward, action, "Starter!");
      else
      {
        if (set_agent_main(tau, obs, reward, action, "Main directly!"))
          return;
      }
    }
  }

  if (agent_ == agent_starter_)
  {
    // run starter agent for some time, it helps the main agent to start
    if (starter_trigger_->check(time_, Vector()))
    {
      if (set_agent_main(tau, obs, reward, action, "Main!"))
        return;
    }
  }

  if (agent_ == agent_standup_)
  {
    // try to stand up (body should be in the upright position)
    if (upright_trigger_->check(time_, obs))
      return set_agent(agent_prepare_, tau, obs, reward, action, "Prepare!");
  }

  agent_.a->step(tau, obs, reward, action);

  if (agent_ == agent_main_)
    ss_++;

  if (!save_completed())
  {
    tstat_.addValue(t_.elapsed());
    t_.restart();
    std::cout << "Saving delay " << tstat_.toStr("s") << std::endl;
  }
}

void LeoStateMachineAgent::set_agent(SMAgent &agent, double tau, const Observation &obs, double reward, Action *action, const char* msg)
{
  if (agent_ != agent)
  {
    // finish previous agent
    agent_.a->end(tau, obs, reward);

    // end of agent_test or agent_main => report and increment trials
    int agent_tm = (agent_ == agent_test_) || (agent_ == agent_main_);
    if (agent_tm)
    {
      report(agent_);
      tt_++;
    }

    // save policy after every N-th test episode or every failure or at the end of learning
    if (!output_.empty())
    {
      int save_after = (test_interval_+1)*3; // 3rd test episode
      if ((agent_ == agent_test_ && tt_%save_after == 0)
          || (agent_tm && (failed(obs) || (steps_ && ss_ >= steps_)) ))
        save(agent_);
    }

    // reset reward
    main_total_reward_ = 0;

    // clear sandbox evnironment history
    if (environment_)
    {
      Configuration config;
      config.set("action", "statclr");
      config.set("sma_state", agent.s);
      environment_->reconfigure(config);
    }

    // start new agent and obtain action
    agent_ = agent;
    agent_.a->start(obs, action);
    INFO(msg);
  }
}

bool LeoStateMachineAgent::set_agent_main(double tau, const Observation &obs, double reward, Action *action, const char* msg)
{
  if (!steps_ || ss_ < steps_)
  {
    // New experiment with the main agent
    int test = (test_interval_ >= 0 && tt_%(test_interval_+1) == test_interval_);

    if (test)
      INFO("Test interval, episode " << tt_);

    SMAgent agent = test ? agent_test_ : agent_main_;

    timer init_t; // add start-up time (critical for NMPC because it is long, 1-2s) => account for it
    set_agent(agent, tau, obs, reward, action, msg);  
    main_time_ = time_ + init_t.elapsed();
    return true;
  }
  return false;
}

void LeoStateMachineAgent::save(SMAgent &agent)
{
  if (thread_)
  {
    INFO("Saving at episode " << tt_);
    pthread_mutex_lock(&save_mtx); // lock the mutex
    save_data.agent = agent.a;
    save_data.tt = tt_;
    save_data.output = output_;
    save_data.completed = false;
    pthread_cond_signal(&save_cond); // request saving
    pthread_mutex_unlock(&save_mtx); // unlock the mutex

    t_.restart();
  }
}

bool LeoStateMachineAgent::save_completed()
{
  if (thread_)
  {
    pthread_mutex_lock(&save_mtx);
    bool completed = save_data.completed;
    pthread_mutex_unlock(&save_mtx);
    return completed;
  }
  return true;
}

void LeoStateMachineAgent::report(SMAgent &agent)
{
  std::ostringstream oss;
  oss << std::setprecision(3) << std::fixed << std::setw(15) << tt_ << std::setw(15) << ss_ << std::setw(15) << main_total_reward_;
  agent.a->report(oss);

  if (environment_)
    environment_->report(oss);

  INFO(oss.str());
  if (ofs_.is_open())
    ofs_ << oss.str() << std::endl;
}
