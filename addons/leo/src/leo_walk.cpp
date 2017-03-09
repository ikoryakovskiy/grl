#include <XMLConfiguration.h>
#include <grl/environments/leo/leo_walk.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoBhWalkSym)
REGISTER_CONFIGURABLE(LeoBhWalk)
REGISTER_CONFIGURABLE(LeoWalkEnvironment)

double LeoBhWalk::calculateReward()
{
  // Original reward used by Erik
  return CLeoBhWalkSym::calculateReward();
}

void LeoBhWalk::parseStateFromTarget(const CLeoState &leoState, Vector &obs, const TargetInterface::ObserverInterface *observer) const
{
  int i, j;
  for (i = 0; i < observer->angles.size(); i++)
    obs[i] = leoState.mJointAngles[ observer->angles[i] ];
  for (j = 0; j < observer->angle_rates.size(); j++)
    obs[i+j] = leoState.mJointSpeeds[ observer->angle_rates[j] ];
  for (int k = 0; k < observer->augmented.size(); k++)
  {
    if (observer->augmented[k] == "heeltoe")
      obs[i+j+k] = (leoState.mFootContacts == 0?0:1); // any contact
    else if (observer->augmented[k] == "toeright")
      obs[i+j+k] = (leoState.mFootContacts & LEO_FOOTSENSOR_RIGHT_TOE);
    else if (observer->augmented[k] == "heelright")
      obs[i+j+k] = (leoState.mFootContacts & LEO_FOOTSENSOR_RIGHT_HEEL);
    else if (observer->augmented[k] == "toeleft")
      obs[i+j+k] = (leoState.mFootContacts & LEO_FOOTSENSOR_LEFT_TOE);
    else if (observer->augmented[k] == "heelleft")
      obs[i+j+k] = (leoState.mFootContacts & LEO_FOOTSENSOR_LEFT_HEEL);
    else
    {
      ERROR("Unknown augmented field '" << observer->augmented[i] << "'");
      throw bad_param("leo_walk:observer_idx_.augmented[i]");
    }
  }
}

void LeoBhWalk::parseActionForTarget(const Action &action, Action &target_action, const TargetInterface::ActuatorInterface *actuator)
{
  for (int i = 0; i < actuator->action.size(); i++)
  {
    if (actuator->action[i] != -1)
      target_action[i] = action[ actuator->action[i] ]; // this is 1-to-1 mapping or symmetrically inverted
  }

  for (int i = 0; i < actuator->autoActuated.size(); i++)
  {
    if (actuator->autoActuated[i] == "stanceknee")
    {
      // refers to an auto-actuation of the stance leg knee
      if (stanceLegLeft())
        target_action[ljKneeLeft] = grlAutoActuateKnee();
      else
        target_action[ljKneeRight] = grlAutoActuateKnee();
    }

    if (actuator->autoActuated[i] == "ankleleft")
      target_action[ljAnkleLeft] = grlAutoActuateLeftAnkle();

    if (actuator->autoActuated[i] == "ankleright")
      target_action[ljAnkleRight] = grlAutoActuateRightAnkle();

    if (actuator->autoActuated[i] == "shoulder")
      target_action[ljShoulder] = grlAutoActuateArm();
  }

  target_action.type = action.type;
}

std::string LeoBhWalk::getProgressReport(double trialTime)
{
  const int pw = 15;
  std::stringstream progressString;
  progressString << std::fixed << std::setprecision(3) << std::right;

  // Report from the base class
  progressString << CLeoBhWalkSym::getProgressReport(trialTime);

  // Number of footsteps in this trial
  progressString << std::setw(pw) << mNumFootsteps;

  // Walked distance
  progressString << std::setw(pw) << mWalkedDistance;

  // Average length of a step
  progressString << std::setw(pw) << mWalkedDistance/mNumFootsteps;

  // Speed
  progressString << std::setw(pw) << mWalkedDistance/trialTime;

  // Energy usage
  progressString << std::setw(pw) << mTrialEnergy;

  // Energy per traveled meter
  if (mWalkedDistance > 0.001)
    progressString << std::setw(pw) << mTrialEnergy/mWalkedDistance;
  else
    progressString << std::setw(pw) << 0.0;

  return progressString.str();
}

void LeoBhWalk::parseLeoState(const CLeoState &leoState, Vector &obs)
{
  parseStateFromTarget(leoState, obs, &interface_.observer);
}

void LeoBhWalk::parseLeoAction(const Action &action, Action &target_action)
{
  parseActionForTarget(action, target_action, &interface_.actuator);
}

/////////////////////////////////

void LeoBhWalkSym::parseLeoState(const CLeoState &leoState, Vector &obs)
{
  if (stanceLegLeft())
    parseStateFromTarget(leoState, obs, &interface_.observer);
  else
    parseStateFromTarget(leoState, obs, &interface_.observer_sym);
}

void LeoBhWalkSym::parseLeoAction(const Action &action, Action &target_action)
{
  if (stanceLegLeft())
    parseActionForTarget(action, target_action, &interface_.actuator);
  else
    parseActionForTarget(action, target_action, &interface_.actuator_sym);
}

std::string LeoBhWalkSym::getProgressReport(double trialTime)
{
  return LeoBhWalk::getProgressReport(trialTime);
}

/////////////////////////////////

LeoWalkEnvironment::LeoWalkEnvironment()
{
}

void LeoWalkEnvironment::request(ConfigurationRequest *config)
{
  LeoBaseEnvironment::request(config);
}

void LeoWalkEnvironment::configure(Configuration &config)
{
  LeoBaseEnvironment::configure(config);

  // Augmenting state with a direction indicator variable, e.g: sit down or stand up
  const TargetInterface &interface = bh_->getInterface();
  Vector obs_min = config["observation_min"].v();
  Vector obs_max = config["observation_max"].v();

  int offset = interface.observer.angles.size()+interface.observer.angle_rates.size();// + interface.observer.contacts.size();
  for (int i = 0; i < interface.observer.augmented.size(); i++)
  {
    if (interface.observer.augmented[i] == "direction")
    {
      obs_min[offset + i] = -1;
      obs_max[offset + i] = +1;
    }
    else if ( (interface.observer.augmented[i] == "heeltoe") || // any contact
              (interface.observer.augmented[i] == "toeright") || (interface.observer.augmented[i] == "heelright") ||
              (interface.observer.augmented[i] == "toeleft") || (interface.observer.augmented[i] == "heelleft") )
    {
      obs_min[offset + i] =  0;
      obs_max[offset + i] =  1;
    }
    else
    {
      ERROR("Unknown augmented field '" << interface.observer.augmented[i] << "'");
      throw bad_param("leo_squat:os.augmented[i]");
    }
  }

  config.set("observation_min", obs_min);
  config.set("observation_max", obs_max);

  TRACE("Observation min: " << obs_min);
  TRACE("Observation max: " << obs_max);
}

void LeoWalkEnvironment::start(int test, Observation *obs)
{
  CRAWL("Starting leo env");
  LeoBaseEnvironment::start(test);

  target_env_->start(test_, &target_obs_);
  CRAWL(target_obs_);

  add_measurement_noise(&target_obs_.v);
  CRAWL(target_obs_);

  // Parse obs into CLeoState (Start with left leg being the stance leg)
  bh_->fillLeoState(target_obs_, Vector(), leoState_);
  bh_->setCurrentSTGState(&leoState_);
  bh_->setPreviousSTGState(&leoState_);

  // update derived state variables
  bh_->updateDerivedStateVars(&leoState_); // swing-stance switching happens here

  // construct new obs from CLeoState
  obs->v.resize(observation_dims_);
  bh_->parseLeoState(leoState_, *obs);
  obs->absorbing = false;

  bh_->setCurrentSTGState(NULL);

  // set ground contact if it is established
  if (pub_ic_signal_)
  {
    int gc = leoState_.mFootContacts ? lstGroundContact : lstNone;
    int sl = bh_->stanceLegLeft() ? lstStanceLeft : lstStanceRight;
    pub_ic_signal_->set(VectorConstructor(gc | sl));
  }
}

double LeoWalkEnvironment::step(const Action &action, Observation *obs, double *reward, int *terminal)
{
  CRAWL("RL action: " << action);

  bh_->setCurrentSTGState(&leoState_);

  // Reconstruct a Leo action from a possibly reduced agent action
  bh_->parseLeoAction(action, target_action_);

  // ensure that action is always within bounds
  CRAWL(target_action_);
  ensure_bounds(&target_action_.v);

  // Execute action
  bh_->setPreviousSTGState(&leoState_);

  TRACE("Target action " << target_action_);
  double tau = target_env_->step(target_action_, &target_obs_, reward, terminal);
  CRAWL(target_obs_);

  add_measurement_noise(&target_obs_.v);
  CRAWL(target_obs_);

  // Filter joint speeds
  // Parse obs into CLeoState
  bh_->fillLeoState(target_obs_, target_action_, leoState_);
  bh_->setCurrentSTGState(&leoState_);

  // update derived state variables
  bh_->updateDerivedStateVars(&leoState_);

  // construct new obs from CLeoState
  bh_->parseLeoState(leoState_, *obs);
  obs->absorbing = false;

  // Determine reward
  *reward = bh_->calculateReward();

  // ... and termination
  if (*terminal == 1) // timeout
    *terminal = 1;
  else if (bh_->isDoomedToFall(&leoState_, true))
  {
    *terminal = 2;
    obs->absorbing = true;
  }
  else
    *terminal = 0;

  // signal contact (agent may use this signal to tackle discontinuities)
  if (pub_ic_signal_)
  {
    Vector signal;
    signal.resize(1+2*ljNumDynamixels);
    if (bh_->madeFootstep())
    {
      const TargetInterface ti = bh_->getInterface();
      Vector ti_actuator, ti_actuator_sym;
      toVector(ti.actuator.action, ti_actuator);
      toVector(ti.actuator_sym.action, ti_actuator_sym);
      if (bh_->stanceLegLeft())
        signal << VectorConstructor(0), ti_actuator, ti_actuator_sym;
      else
        signal << VectorConstructor(0), ti_actuator_sym, ti_actuator;
    }

    int td = bh_->madeFootstep()      ? lstSwlTouchDown  : lstNone;
    int gc = leoState_.mFootContacts  ? lstGroundContact : lstNone;
    int sl = bh_->stanceLegLeft()     ? lstStanceLeft    : lstStanceRight;
    signal[0] = td | gc | sl;
    pub_ic_signal_->set(signal);
  }

  LeoBaseEnvironment::step(tau, *reward, *terminal);

  return tau;
}

void LeoWalkEnvironment::report(std::ostream &os) const
{
  double trialTime  = (test_?time_test_:time_learn_) - time0_;
  os << bh_->getProgressReport(trialTime);
}

