#ifndef GRL_LEO_WALK_ENVIRONMENT_H_
#define GRL_LEO_WALK_ENVIRONMENT_H_

#include <leo.h>
#include <grl/environments/odesim/environment.h>
#include <grl/signal.h>
#include <LeoBhWalkSym.h>
#include <STGLeo.h>
#include <STGLeoSim.h>
#include <ThirdOrderButterworth.h>

namespace grl
{

class LeoBhWalk: public CLeoBhBase
{
  public:
    TYPEINFO("behavior/leo_walk", "Leo walking behavior without symmetrical switchers of observations")

    LeoBhWalk() {}

    virtual double calculateReward();
    virtual void parseLeoState(const CLeoState &leoState, Vector &obs);
    virtual void parseLeoAction(const Action &action, Action &target_action);
    virtual std::string getProgressReport(double trialTime);

  protected:
    void parseStateFromTarget(const CLeoState &leoState, Vector &obs, const TargetInterface::ObserverInterface *observer) const;
    void parseActionForTarget(const Action &action, Action &target_action, const TargetInterface::ActuatorInterface *actuator);

};

class LeoBhWalkSym: public LeoBhWalk
{
  public:
    TYPEINFO("behavior/leo_walk_sym", "Leo walking behavior with symmetrical switchers of observations")

    LeoBhWalkSym() {}

    virtual void parseLeoState(const CLeoState &leoState, Vector &obs);
    virtual void parseLeoAction(const Action &action, Action &target_action);
    virtual std::string getProgressReport(double trialTime);

};

/// Simulation of original Leo robot by Erik Schuitema.
class LeoWalkEnvironment: public LeoBaseEnvironment
{
  public:
    TYPEINFO("environment/leo_walk", "Leo walking environment")

  public:
    LeoWalkEnvironment();
    ~LeoWalkEnvironment() {}

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Environment
    virtual void start(int test, Observation *obs);
    virtual double step(const Action &action, Observation *obs, double *reward, int *terminal);

    virtual void report(std::ostream &os) const;
    
  protected:
    int requested_action_dims_;
};

}

#endif /* GRL_LEO_WALK_ENVIRONMENT_H_ */
