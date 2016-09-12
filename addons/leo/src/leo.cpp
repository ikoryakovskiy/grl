#include <sys/stat.h>
#include <XMLConfiguration.h>
#include <grl/environments/leo/leo.h>

using namespace grl;

void CLeoBhBase::resetState()
{
  mIsObserving          = false;
  mLastRewardedFoot     = lpFootLeft;
  mLastStancelegWasLeft = -1;
  mFootstepLength       = 0.0;
  mLastFootstepLength   = 0.0;
  mNumFootsteps         = 0;
  mWalkedDistance       = 0.0;
  mTrialEnergy          = 0.0;

  // Reset velocity filters to zero velocity (this is the result of robot->setIC)
  for (int iJoint=0; iJoint<ljNumJoints; iJoint++)
    mJointSpeedFilter[iJoint].clear();

  for (int i=0; i<ljNumDynamixels; i++)
    mJointSpeedFilter[i].init(1.0/mTotalStepTime, 10.0);
  mJointSpeedFilter[ljTorso].init(mTotalStepTime, 25.0);	// 25Hz because? : 1) this encoder has 8x the resolution of a dynamixel 2) torso angles/velocities are more important
}

void CLeoBhBase::fillLeoState(const Vector &obs, const Vector &action, CLeoState &leoState)
{
  // '-' required to match with Erik's code, but does not matter for learning.
  // Erik used a rotation matrix which was rotating a unit vector. For torso it seems
  // the positive direction was not same as for other joints, internally defined in ODE.
  leoState.mJointAngles[ljTorso]      = -obs[svTorsoAngle];
  leoState.mJointSpeeds[ljTorso]      = -mJointSpeedFilter[ljTorso].filter(obs[svTorsoAngleRate]);
  leoState.mJointAngles[ljShoulder]   = obs[svLeftArmAngle];
  leoState.mJointSpeeds[ljShoulder]   = mJointSpeedFilter[ljShoulder].filter(obs[svLeftArmAngleRate]);
  leoState.mJointAngles[ljHipRight]   = obs[svRightHipAngle];
  leoState.mJointSpeeds[ljHipRight]   = mJointSpeedFilter[ljHipRight].filter(obs[svRightHipAngleRate]);
  leoState.mJointAngles[ljHipLeft]    = obs[svLeftHipAngle];
  leoState.mJointSpeeds[ljHipLeft]    = mJointSpeedFilter[ljHipLeft].filter(obs[svLeftHipAngleRate]);
  leoState.mJointAngles[ljKneeRight]  = obs[svRightKneeAngle];
  leoState.mJointSpeeds[ljKneeRight]	= mJointSpeedFilter[ljKneeRight].filter(obs[svRightKneeAngleRate]);
  leoState.mJointAngles[ljKneeLeft]   = obs[svLeftKneeAngle];
  leoState.mJointSpeeds[ljKneeLeft]   = mJointSpeedFilter[ljKneeLeft].filter(obs[svLeftKneeAngleRate]);
  leoState.mJointAngles[ljAnkleRight] = obs[svRightAnkleAngle];
  leoState.mJointSpeeds[ljAnkleRight] = mJointSpeedFilter[ljAnkleRight].filter(obs[svRightAnkleAngleRate]);
  leoState.mJointAngles[ljAnkleLeft]  = obs[svLeftAnkleAngle];
  leoState.mJointSpeeds[ljAnkleLeft]	= mJointSpeedFilter[ljAnkleLeft].filter(obs[svLeftAnkleAngleRate]);

  leoState.mFootContacts  = obs[svRightToeContact]?LEO_FOOTSENSOR_RIGHT_TOE:0;
  leoState.mFootContacts |= obs[svRightHeelContact]?LEO_FOOTSENSOR_RIGHT_HEEL:0;
  leoState.mFootContacts |= obs[svLeftToeContact]?LEO_FOOTSENSOR_LEFT_TOE:0;
  leoState.mFootContacts |= obs[svLeftHeelContact]?LEO_FOOTSENSOR_LEFT_HEEL:0;

  // required for the correct energy calculation in the reward function
  if (action.size())
  {
    leoState.mActuationVoltages[ljShoulder]   = action[avLeftArmTorque];
    leoState.mActuationVoltages[ljHipRight]   = action[avRightHipTorque];
    leoState.mActuationVoltages[ljHipLeft]    = action[avLeftHipTorque];
    leoState.mActuationVoltages[ljKneeRight]  = action[avRightKneeTorque];
    leoState.mActuationVoltages[ljKneeLeft]   = action[avLeftKneeTorque];
    leoState.mActuationVoltages[ljAnkleRight] = action[avRightAnkleTorque];
    leoState.mActuationVoltages[ljAnkleLeft]  = action[avLeftAnkleTorque];
  }
}

void CLeoBhBase::parseLeoState(const CLeoState &leoState, Vector &obs)
{
  obs[owTorsoAngle]           = leoState.mJointAngles[ljTorso];
  obs[owTorsoAngleRate]       = leoState.mJointSpeeds[ljTorso];
  obs[owHipStanceAngle]       = leoState.mJointAngles[mHipStance];
  obs[owHipStanceAngleRate]   = leoState.mJointSpeeds[mHipStance];
  obs[owHipSwingAngle]        = leoState.mJointAngles[mHipSwing];
  obs[owHipSwingAngleRate]    = leoState.mJointSpeeds[mHipSwing];
  obs[owKneeStanceAngle]      = leoState.mJointAngles[mKneeStance];
  obs[owKneeStanceAngleRate]  = leoState.mJointSpeeds[mKneeStance];
  obs[owKneeSwingAngle]       = leoState.mJointAngles[mKneeSwing];
  obs[owKneeSwingAngleRate]   = leoState.mJointSpeeds[mKneeSwing];
}

void CLeoBhBase::setCurrentSTGState(CLeoState *leoState)
{
  mCurrentSTGState = leoState;
}

void CLeoBhBase::setPreviousSTGState(CLeoState *leoState)
{
  mPreviousSTGState = *leoState;
}

void CLeoBhBase::updateDerivedStateVars(CLeoState* currentSTGState)
{
  CLeoBhWalkSym::updateDerivedStateVars(currentSTGState);
}

int CLeoBhBase::jointNameToIndex(const std::string jointName) const
{
  if (jointName == "torso_boom")
    return ljTorso;
  else if (jointName == "shoulder")
    return ljShoulder;
  else if (jointName == "hipright")
    return ljHipRight;
  else if (jointName == "hipleft")
    return ljHipLeft;
  else if (jointName == "kneeright")
    return ljKneeRight;
  else if (jointName == "kneeleft")
    return ljKneeLeft;
  else if (jointName == "ankleright")
    return ljAnkleRight;
  else if (jointName == "ankleleft")
    return ljAnkleLeft;
  else
    return -1; // augmented state
}

std::string CLeoBhBase::jointIndexToName(int jointIndex) const
{
  switch(jointIndex)
  {
    case ljTorso      : return std::string("torso_boom"); break;
    case ljShoulder   : return std::string("shoulder");   break;
    case ljHipRight   : return std::string("hipright");   break;
    case ljHipLeft    : return std::string("hipleft");    break;
    case ljKneeRight  : return std::string("kneeright");  break;
    case ljKneeLeft   : return std::string("kneeleft");   break;
    case ljAnkleRight : return std::string("ankleright"); break;
    case ljAnkleLeft  : return std::string("ankleleft");  break;
    default:
      ERROR("Joint index out of bounds '" << jointIndex << "'");
      throw bad_param("leobase:jointIndex");
  }
}

/////////////////////////////////

LeoBaseEnvironment::LeoBaseEnvironment() :
  target_env_(NULL),
  observation_dims_(CLeoBhBase::svNumStates),
  time_test_(0),
  time_learn_(0),
  time0_(0),
  test_(0),
  exporter_(NULL),
  bh_(NULL)
{
}

void LeoBaseEnvironment::request(ConfigurationRequest *config)
{
  config->push_back(CRP("xml", "XML configuration filename", xml_));
  config->push_back(CRP("target_env", "environment", "Interaction environment", target_env_));
  config->push_back(CRP("observe", "string.observe", "Comma-separated list of state elements observed by an agent"));
  config->push_back(CRP("actuate", "string.actuate", "Comma-separated list of action elements provided by an agent"));
  config->push_back(CRP("exporter", "exporter", "Optional exporter for transition log (supports time, state, observation, action, reward, terminal)", exporter_, true));

  config->push_back(CRP("observation_dims", "int.observation_dims", "Number of observation dimensions", target_observation_dims_));
  config->push_back(CRP("action_dims", "int.action_dims", "Number of action dimensions", target_action_dims_));
}

void LeoBaseEnvironment::configure(Configuration &config)
{
  // Setup path to a configuration file
  xml_ = config["xml"].str();
  struct stat buffer;
  if (stat (xml_.c_str(), &buffer) != 0)
    xml_ = std::string(LEO_CONFIG_DIR) + "/" + config["xml"].str();
  std::cout << xml_ << std::endl;

  target_env_ = (Environment*)config["target_env"].ptr(); // here we can select an actual Leo enviromnent (simulation/real)

  target_observation_dims_ = config["observation_dims"];
  target_action_dims_ = config["action_dims"];

  exporter_ = (Exporter*) config["exporter"].ptr();
  if (exporter_)
    exporter_->init({"time", "state0", "state1", "action", "reward", "terminal"});

  // Process configuration of Leo
  CXMLConfiguration xmlConfig;
  if (!xmlConfig.loadFile(xml_))
  {
    ERROR("Couldn't load XML configuration file \"" << xml_ << "\"!\nPlease check that the file exists and that it is sound (error: " << xmlConfig.errorStr() << ").");
    return;
  }

  // Resolve expressions
  xmlConfig.resolveExpressions();
  
  // Read rewards and preprogrammed angles
  bh_->readConfig(xmlConfig.root());

  // Create ode object which resolves states and actions
  ODESTGEnvironment *ode = new ODESTGEnvironment();
  if (!ode->configure(config))
  {
    ERROR("Could not initialize STG ODE environment");
    return;
  }

  // Select states and actions that are delivered to an agent
  configParseObservations(config, ode->getSensors());
  configParseActions(config, ode->getActuators());

  delete ode;

  // reserve memory
  target_obs_.resize(target_observation_dims_);
  target_action_.resize(target_action_dims_);
}

void LeoBaseEnvironment::reconfigure(const Configuration &config)
{
  time_test_ = time_learn_ = time0_ = 0;
}

LeoBaseEnvironment *LeoBaseEnvironment::clone() const
{
  return NULL;
}

void LeoBaseEnvironment::start(int test)
{
  test_ = test;
  bh_->resetState();

  if (exporter_)
    exporter_->open((test_?"test":"learn"), (test_?time_test_:time_learn_) != 0.0);
  time0_ = test_?time_test_:time_learn_;
}

void LeoBaseEnvironment::step(double tau, double reward, int terminal)
{
  double &time = test_?time_test_:time_learn_;

  // Export & debug
  std::vector<double> s1(leoState_.mJointAngles, leoState_.mJointAngles + ljNumJoints);
  std::vector<double> v1(leoState_.mJointSpeeds, leoState_.mJointSpeeds + ljNumJoints);
  std::vector<double> a(leoState_.mActuationVoltages, leoState_.mActuationVoltages + ljNumDynamixels);

  if (exporter_)
  {
    std::vector<double> s0(bh_->getPreviousSTGState()->mJointAngles, bh_->getPreviousSTGState()->mJointAngles + ljNumJoints);
    std::vector<double> v0(bh_->getPreviousSTGState()->mJointSpeeds, bh_->getPreviousSTGState()->mJointSpeeds + ljNumJoints);
    s0.insert(s0.end(), v0.begin(), v0.end());
    s1.insert(s1.end(), v1.begin(), v1.end());

    Vector s0v, s1v, av;
    toVector(s0, s0v);
    toVector(s1, s1v);
    toVector(a, av);

    exporter_->write({grl::VectorConstructor(time), s0v,  s1v,
                      av, grl::VectorConstructor(reward), grl::VectorConstructor(terminal)
                     });
  }

  TRACE("State angles: " << s1);
  TRACE("State velocities: " << v1);
  TRACE("Contacts: " << (int)leoState_.mFootContacts);
  TRACE("Full action: " << a);
  TRACE("Reward: " << reward);

  time += tau;
}

void LeoBaseEnvironment::report(std::ostream &os) const
{
  double trialTime  = test_?time_test_:time_learn_ - time0_;
  os << bh_->getProgressReport(trialTime);
}

///////////////////////////////////////////
/// Helper functions
///
void LeoBaseEnvironment::configParseObservations(Configuration &config, const std::vector<CGenericStateVar> &sensors)
{
  const std::vector<std::string> observeList = cutLongStr( config["observe"].str() );
  std::vector<std::string> observe;
  fillObserve(sensors, observeList, observe);
  EnvironmentAgentInterface::ObserverInterface observer;
  fillObserver(observe, observer);
  observation_dims_ = observe.size();

  // mirror left and right legs
  std::vector<std::string> observe_sym = observe;
  for (int i = 0; i < observe.size(); i++)
  {
    std::size_t idx = observe[i].find("right");
    if (idx != std::string::npos)
      observe_sym[i].replace(idx, 5, "left");
    idx = observe[i].find("left");
    if (idx != std::string::npos)
      observe_sym[i].replace(idx, 4, "right");
  }
  EnvironmentAgentInterface::ObserverInterface observer_sym;
  fillObserver(observe_sym, observer_sym);

  // mask observation min/max vectors
  Vector ode_observation_min, ode_observation_max, observation_min, observation_max;
  config.get("observation_min", ode_observation_min);
  config.get("observation_max", ode_observation_max);
  observation_min.resize(observation_dims_);
  observation_max.resize(observation_dims_);

  int i, j, k;
  for (i = 0; i < observer.angles.size(); i++)
  {
    std::string name = "robot." + bh_->jointIndexToName(observer.angles[i]) + ".angle";
    int sensor_idx = findVarIdx(sensors, name);
    observation_min[i] = ode_observation_min[sensor_idx];
    observation_max[i] = ode_observation_max[sensor_idx];
  }
  for (j = 0; j < observer.angle_rates.size(); j++)
  {
    std::string name = "robot." + bh_->jointIndexToName(observer.angle_rates[j]) + ".anglerate";
    int sensor_idx = findVarIdx(sensors, name);
    observation_min[i+j] = ode_observation_min[sensor_idx];
    observation_max[i+j] = ode_observation_max[sensor_idx];
  }

  // Set parameters exported to an agent
  config.set("observation_dims", observation_dims_);
  config.set("observation_min", observation_min);
  config.set("observation_max", observation_max);

  // Prepare observer indexes for easy connection between states of the target environment and agent observations
  bh_->setObserverInterface(observer, observer_sym);
}

int LeoBaseEnvironment::findVarIdx(const std::vector<CGenericStateVar> &genericStates, std::string query) const
{
  std::vector<CGenericStateVar>::const_iterator gState = genericStates.begin();
  for (int i = 0; gState < genericStates.end(); gState++, i++)
    if (query == gState->name())
      return i;
  return -1;
}

void LeoBaseEnvironment::fillObserver(const std::vector<std::string> &observer_names, EnvironmentAgentInterface::ObserverInterface &observer_interface) const
{
  for (int i = 0; i < observer_names.size(); i++)
  {
    std::string name = observer_names[i];
    std::replace( name.begin(), name.end(), '.', ' ');
    std::vector<std::string> cuttedName = cutLongStr(name);
    if (cuttedName.size() == 1)
      observer_interface.augmented.push_back(name);
    else if (cuttedName[2] == "angle")
      observer_interface.angles.push_back(bh_->jointNameToIndex(cuttedName[1]));
    else if (cuttedName[2] == "anglerate")
      observer_interface.angle_rates.push_back(bh_->jointNameToIndex(cuttedName[1]));
    else if (cuttedName[2] == "contact")
      observer_interface.augmented.push_back(cuttedName[1]);
    else
    {
      ERROR("Unknown joint '" << cuttedName[2] << "'");
      throw bad_param("leobase:cuttedName[2]");
    }
  }
}

void LeoBaseEnvironment::configParseActions(Configuration &config, const std::vector<CGenericActionVar> &actuators)
{
  EnvironmentAgentInterface::ActuatorInterface actuator_interface;
  std::vector<std::string> actuateList = cutLongStr(config["actuate"].str());
  fillActuate(actuators, actuateList, actuator_interface);
  TRACE("Actuate '" << actuator_interface.actions << "'"); // array which maps target_environment action vector to an agent action vector
  if (actuator_interface.actions.size() != target_action_dims_)
    throw bad_param("leobase:actuate");
  action_dims_ = actuateList.size();

  // mask observation min/max vectors
  Vector target_action_min, target_action_max, action_min, action_max;
  config.get("action_min", target_action_min);
  config.get("action_max", target_action_max);
  action_min.resize(action_dims_);
  action_max.resize(action_dims_);

  for (int j = 0; j < action_dims_; j++)
  {
    // if we actuate a few joints by the same value, as "knee" from 'actuateList' will actuate both left and right knees
    double min = -std::numeric_limits<double>::max();
    double max = +std::numeric_limits<double>::max();
    for (int i = 0; i < actuator_interface.actions.size(); i++)
    {
      if (actuator_interface.actions[i] == j)
      {
        max = MIN(target_action_max[i], max);
        min = MAX(target_action_min[i], min);
      }
    }

    // actuate required joints but narrowest bound is enforced
    action_min[j] = min;
    action_max[j] = max;
  }

  // Set parameters exported to an agent
  config.set("action_dims", action_dims_);
  config.set("action_min", action_min);
  config.set("action_max", action_max);

  TRACE("Action min: " << action_min);
  TRACE("Action max: " << action_max);

  // Prepare actuator indexes for easy connection between actions of the target environment and agent actions
  bh_->setActuatorInterface(actuator_interface);
}

void LeoBaseEnvironment::fillObserve(const std::vector<CGenericStateVar> &genericStates,
                                      const std::vector<std::string> &observeList,
                                      std::vector<std::string> &observe) const
{
  std::vector<std::string>::const_iterator listMember = observeList.begin();
  std::vector<CGenericStateVar>::const_iterator gState;
  std::string::const_iterator it;
  for (; listMember < observeList.end(); listMember++)
  {
    bool found = false;
    gState = genericStates.begin();
    for (int i = 0; gState < genericStates.end(); gState++, i++)
    {
      const std::string &name = gState->name();
      it = std::search(name.begin(), name.end(), listMember->begin(), listMember->end());

      if (it != name.end())
      {
        it += listMember->size(); // point at the end of substring
        if (it == name.end() || *it == '.')
        {
          INFO("Adding to the observation vector (physical state): " << name);
          observe.push_back(name);
          found = true;
        }
      }
    }

    if (!found)
    {
      INFO("Adding to the observation vector (augmented state): " << *listMember);
      observe.push_back(*listMember);
    }
  }
}

void LeoBaseEnvironment::fillActuate(const std::vector<CGenericActionVar> &genericAction,
                                     const std::vector<std::string> &actuateList,
                                     EnvironmentAgentInterface::ActuatorInterface &out,
                                     const std::string *req,
                                     std::vector<int>  *reqIdx) const
{
  out.actions.resize(genericAction.size());
  for (int i = 0; i < out.actions.size(); i++) out.actions[i] = -1;
  std::vector<std::string>::const_iterator listMember = actuateList.begin();
  std::vector<CGenericActionVar>::const_iterator gAction;
  std::string::const_iterator it;

  for (int j = 0; listMember < actuateList.end(); listMember++, j++)
  {
    bool found = false;
    gAction = genericAction.begin();
    for (int i = 0; gAction < genericAction.end(); gAction++, i++)
    {
      std::string name = gAction->name();
      it = std::search(name.begin(), name.end(), listMember->begin(), listMember->end());

      if (it != name.end())
      {
        it += listMember->size(); // point at the end of substring
        //if (it == name.end() || *it == '.')
        {
          INFO("Adding to the actuation vector: " << name);
          out.actions[i] = j;
          if (req != NULL && reqIdx != NULL)
            if (std::search(listMember->begin(), listMember->end(), req->begin(), req->end()) != listMember->end())
              reqIdx->push_back(i);
          found = true;
        }
      }
    }

    if (!found)
    {
      ERROR("Unknown actuaton joint '" << *listMember << "'");
      throw bad_param("leobase:listMember");
    }
  }

  for (int i = 0; i < out.actions.size(); i++)
  {
    if (out.actions[i] == -1)
    {
      std::string name = genericAction[i].name();
      std::replace( name.begin(), name.end(), '.', ' ');
      std::vector<std::string> cuttedName = cutLongStr(name);
      out.autoActuated.push_back(cuttedName[1]);
      INFO("Adding auto-actuated joint '" << cuttedName[1] << "'");
    }
  }
}