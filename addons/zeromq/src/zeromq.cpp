/** \file zeromq.cpp
 * \brief ZeroMQ policy source file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@gmail.com>
 * \date      2016-02-09
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

#include <zeromq.h>
#include <iomanip>
#include <unistd.h>

using namespace grl;

REGISTER_CONFIGURABLE(ZeromqPubSubCommunicator)
REGISTER_CONFIGURABLE(CommunicatorEnvironment)
REGISTER_CONFIGURABLE(ZeromqRequestReplyCommunicator)
REGISTER_CONFIGURABLE(ZeromqAgent)
REGISTER_CONFIGURABLE(ZeromqAgentDRL)


void ZeromqCommunicator::request(ConfigurationRequest *config)
{
  config->push_back(CRP("pattern", "Pattern of the zeromq implementation (Pub/Sub, Request/Reply)", "", CRP::Configuration, {"NONE", "ZMQ_SUB", "ZMQ_PUB","ZMQ_REQ","ZMQ_REP"}));
  config->push_back(CRP("sync", "Syncronization ip address", sync_));
}

void ZeromqCommunicator::configure(Configuration &config)
{
  std::string type = config["pattern"].str();
  sync_ = config["sync"].str();

  if (type == "ZMQ_SUB")
    type_ = ZMQ_SUB;
  if (type == "ZMQ_PUB")
    type_ = ZMQ_PUB;
  if (type == "ZMQ_REQ")
    type_ = ZMQ_REQ;
  if (type == "ZMQ_REP")
    type_ = ZMQ_REP;
}

void ZeromqCommunicator::send(const Vector v) const
{
  zmq_messenger_.send(reinterpret_cast<const void*>(v.data()), v.cols()*sizeof(double));
}

bool ZeromqCommunicator::recv(Vector *v) const
{
  Vector v_rc;
  v_rc.resize(v->size());
  CRAWL(v_rc.cols());
  bool rc = zmq_messenger_.recv(reinterpret_cast<void*>(v_rc.data()), v_rc.cols()*sizeof(double), 0);//, ZMQ_DONTWAIT);
  if (rc)
  {
    *v = v_rc; // modify content only if data was received
    //std::cout << std::fixed << std::setprecision(2) << std::right << std::setw(7) << v << std::endl << std::endl;
  }
  return rc;
}

/////////////////////////////////////////////////////////

void ZeromqPubSubCommunicator::request(ConfigurationRequest *config)
{
  ZeromqCommunicator::request(config);
  config->push_back(CRP("pub", "Publisher address", pub_));
  config->push_back(CRP("sub", "subscriber address", sub_));
}

void ZeromqPubSubCommunicator::configure(Configuration &config)
{
  ZeromqCommunicator::configure(config);

  // possible addresses
  // zmq_.init("tcp://*:5561", "tcp://192.168.2.210:5562", "tcp://192.168.2.210:5560", ZMQ_SYNC_SUB); // wifi
  // zmq_.init("tcp://*:5561",  "tcp://192.168.1.10:5562",  "tcp://192.168.1.10:5560", ZMQ_SYNC_SUB); // ethernet

  pub_ = config["pub"].str();
  sub_ = config["sub"].str();

  // initialize zmq
  zmq_messenger_.start(type_, pub_.c_str(), sub_.c_str(), sync_.c_str());
}
////////////////////////////////////////////////////////

void ZeromqRequestReplyCommunicator::request(ConfigurationRequest *config)
{
  ZeromqCommunicator::request(config);
  config->push_back(CRP("addr", "Address", addr_));
}

void ZeromqRequestReplyCommunicator::configure(Configuration &config)
{
  ZeromqCommunicator::configure(config);

  addr_ = config["addr"].str();

  // initialize zmq
  zmq_messenger_.start(type_, addr_.c_str(), NULL, sync_.c_str());
}

//////////////////////////////////////////////////////////
void CommunicatorEnvironment::request(ConfigurationRequest *config)
{
  config->push_back(CRP("converter", "converter", "Convert states and actions if needed", converter_, true));
  config->push_back(CRP("communicator", "communicator", "Comunicator which exchanges messages with an actual/virtual environment", communicator_));
  config->push_back(CRP("target_obs_dims", "Observation dimension of a target", target_obs_dims_, CRP::System));
  config->push_back(CRP("target_action_dims", "Action dimension of a target", target_action_dims_, CRP::System));
}

void CommunicatorEnvironment::configure(Configuration &config)
{
  converter_ = (StateActionConverter*)config["converter"].ptr();
  communicator_ = (Communicator*)config["communicator"].ptr();
  target_obs_dims_ = config["target_obs_dims"];
  target_action_dims_ = config["target_action_dims"];

  if (converter_)
  {
    if (converter_->get_state_in_size() != target_obs_dims_)
      throw bad_param("environment/communicator:target_obs_dims");

    if (converter_->get_action_out_size() != target_action_dims_)
      throw bad_param("environment/communicator:target_action_dims_");
  }

  obs_conv_.resize(target_obs_dims_);
  action_conv_.resize(target_action_dims_);
  computation_stat_.setBufferLength(500);
}

void CommunicatorEnvironment::reconfigure(const Configuration &config)
{
}

void CommunicatorEnvironment::start(int test, Observation *obs)
{
  communicator_->recv(&obs_conv_);
  clock_gettime(CLOCK_MONOTONIC, &computation_begin_);
  if (converter_)
    converter_->convert_state(obs_conv_, obs->v);
  else
    *obs = obs_conv_;
  obs->absorbing = false;
}

double CommunicatorEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  if (converter_)
    converter_->convert_action(action, action_conv_);
  else
    action_conv_ = action;

  if (measure_stat_)
  {
    timespec computation_end;
    clock_gettime(CLOCK_MONOTONIC, &computation_end);
    double computation_delay = (computation_end.tv_sec - computation_begin_.tv_sec)*1.0e6 + (static_cast<double>(computation_end.tv_nsec - computation_begin_.tv_nsec))/1.0e3;
    computation_stat_.addValue(computation_delay);
    std::cout << "Computation delay: " << computation_stat_.toStr("us") << std::endl;
  }

  communicator_->send(action_conv_);
  communicator_->recv(&obs_conv_);

  timespec computation_begin_prev = computation_begin_;
  clock_gettime(CLOCK_MONOTONIC, &computation_begin_);

  if (converter_)
    converter_->convert_state(obs_conv_, obs->v);
  else
    *obs = obs_conv_;
  obs->absorbing = false;

  double tau = (computation_begin_.tv_sec - computation_begin_prev.tv_sec) + (static_cast<double>(computation_begin_.tv_nsec - computation_begin_prev.tv_nsec))/1.0e9;
  //std::cout << "stg time: " << tau << std::endl;
  return tau;
}

//////////////////////////////////////////////////////////
void ZeromqAgent::request(ConfigurationRequest *config)
{
  config->push_back(CRP("communicator", "communicator", "Comunicator which exchanges messages with an actual/virtual environment", communicator_));
  config->push_back(CRP("observation_dims", "int.observation_dims", "Number of observation dimensions", observation_dims_, CRP::System));
  config->push_back(CRP("action_dims", "int.action_dims", "Number of action dimensions", action_dims_, CRP::System));
  config->push_back(CRP("action_min", "vector.action_min", "Lower limit of action", action_min_, CRP::System));
  config->push_back(CRP("action_max", "vector.action_max", "Upper limit of action", action_max_, CRP::System));
  config->push_back(CRP("test", "int.test", "Selection of learning/testing agent", test_, CRP::System));
}

void ZeromqAgent::configure(Configuration &config)
{
  // Read configuration
  action_dims_ = config["action_dims"];
  observation_dims_ = config["observation_dims"];
  action_min_ = config["action_min"].v();
  action_max_ = config["action_max"].v();
  communicator_ = (Communicator*)config["communicator"].ptr();
  test_ = config["test"];
}

void ZeromqAgent::reconfigure(const Configuration &config)
{
}

void ZeromqAgent::start(const Observation &obs, Action *action)
{
  action->v.resize(action_dims_);
  action->type = atUndefined;

  Vector a(obs.v.cols()+1);
  a << test_, obs.v;
  communicator_->send(a);
  communicator_->recv(&(action->v));
  //NOTICE(action->v);
}

void ZeromqAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  action->v.resize(action_dims_);
  action->type = atUndefined;
  
  Vector a(obs.v.cols()+3);
  a << test_, obs.v, reward, 0;
  communicator_->send(a);
  communicator_->recv(&(action->v));
}

void ZeromqAgent::end(double tau, const Observation &obs, double reward)
{
  Vector temp(action_dims_);

  Vector a(obs.v.cols()+3);
  a << test_, obs.v, reward, 2;
  communicator_->send(a);
  communicator_->recv(&temp);
}

//////////////////////////////////////////////////////////
void ZeromqAgentDRL::request(ConfigurationRequest *config)
{
  config->push_back(CRP("communicator", "communicator", "Comunicator which exchanges messages with an actual/virtual environment", communicator_));
  config->push_back(CRP("observation_dims", "int.observation_dims", "Number of observation dimensions", observation_dims_, CRP::System));
  config->push_back(CRP("action_dims", "int.action_dims", "Number of action dimensions", action_dims_, CRP::System));
  config->push_back(CRP("action_min", "vector.action_min", "Lower limit of action", action_min_, CRP::System));
  config->push_back(CRP("action_max", "vector.action_max", "Upper limit of action", action_max_, CRP::System));
  config->push_back(CRP("test", "int.test", "Selection of learning/testing agent", test_, CRP::System));
  config->push_back(CRP("pub_state_drl","signal/vector","State received from python",pub_state_drl_,true));
}

void ZeromqAgentDRL::configure(Configuration &config)
{
  // Read configuration
  action_dims_ = config["action_dims"];
  observation_dims_ = config["observation_dims"];
  action_min_ = config["action_min"].v();
  action_max_ = config["action_max"].v();
  communicator_ = (Communicator*)config["communicator"].ptr();
  test_ = config["test"];
  pub_state_drl_ = (VectorSignal*)config["pub_state_drl"].ptr();
}

void ZeromqAgentDRL::reconfigure(const Configuration &config)
{
}

void ZeromqAgentDRL::start(const Observation &obs, Action *action)
{
  action->v.resize(action_dims_);
  action->type = atUndefined;

  Vector a(obs.v.cols()+1);
  Vector b(action->v.cols()+obs.v.cols());
  Vector c(obs.v.cols());
  a << test_, obs.v;
  communicator_->send(a);

  communicator_->recv(&b);
  action->v << b.head((action->v.cols()));
  c << b.tail(obs.v.cols());
  pub_state_drl_->set(c);
}

void ZeromqAgentDRL::step(double tau, const Observation &obs, double reward, Action *action)
{
  action->v.resize(action_dims_);
  action->type = atUndefined;

  Vector a(obs.v.cols()+3);
  Vector b(action->v.cols()+obs.v.cols());
  Vector c(obs.v.cols());
  a << test_, obs.v, reward, 0;
  communicator_->send(a);

  communicator_->recv(&b);
  action->v << b.head(action->v.cols());
  c << b.tail(obs.v.cols());
  pub_state_drl_->set(c);
  //NOTICE(pub_state_drl_);
}

void ZeromqAgentDRL::end(double tau, const Observation &obs, double reward)
{
  Vector temp(action_dims_+obs.v.cols());

  Vector a(obs.v.cols()+3);
  a << test_, obs.v, reward, 2;
  communicator_->send(a);
  communicator_->recv(&temp);
}
