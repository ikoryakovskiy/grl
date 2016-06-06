#ifndef GRL_LEO_ENVIRONMENT_H_
#define GRL_LEO_ENVIRONMENT_H_

#include <grl/environments/odesim/environment.h>
#include <LeoBhWalkSym.h>
#include <STGLeo.h>
#include <STGLeoSim.h>
#include <ThirdOrderButterworth.h>

namespace grl
{

// Base classes for Leo
class CLeoBhBase: public CLeoBhWalkSym
{
  public:
    enum LeoStateVar
    {
      svTorsoAngle,
      svTorsoAngleRate,
      svLeftArmAngle,
      svLeftArmAngleRate,
      svRightHipAngle,
      svRightHipAngleRate,
      svLeftHipAngle,
      svLeftHipAngleRate,
      svRightKneeAngle,
      svRightKneeAngleRate,
      svLeftKneeAngle,
      svLeftKneeAngleRate,
      svRightAnkleAngle,
      svRightAnkleAngleRate,
      svLeftAnkleAngle,
      svLeftAnkleAngleRate,
      svRightToeContact,
      svRightHeelContact,
      svLeftToeContact,
      svLeftHeelContact,
      svNumStates
    };
    enum LeoActionVar
    {
      avLeftArmTorque,
      avRightHipTorque,
      avLeftHipTorque,
      avRightKneeTorque,
      avLeftKneeTorque,
      avRightAnkleTorque,
      avLeftAnkleTorque,
      svNumActions
    };

  public:
    CLeoBhBase(ISTGActuation *actuationInterface) : CLeoBhWalkSym(actuationInterface) {}

    int getHipStance()   {return mHipStance;}
    int getHipSwing()    {return mHipSwing;}
    int getKneeStance()  {return mKneeStance;}
    int getKneeSwing()   {return mKneeSwing;}
    int getAnkleStance() {return mAnkleStance;}
    int getAnkleSwing()  {return mAnkleSwing;}
    bool stanceLegLeft() {return mLastStancelegWasLeft;}

  public:
    void resetState();
    void fillLeoState(const Vector &obs, const Vector &action, CLeoState &leoState);
    void parseLeoState(const CLeoState &leoState, Vector &obs);
    void updateDerivedStateVars(CLeoState *currentSTGState);
    void setCurrentSTGState(CLeoState *leoState);
    void setPreviousSTGState(CLeoState *leoState);
    void grlAutoActuateAnkles(Vector &out)
    {
      CSTGLeoSim *leoSim = dynamic_cast<CSTGLeoSim*>(mActuationInterface);
      CLeoBhWalkSym::autoActuateAnkles_FixedPos(leoSim);
      out.resize(2);
      out << leoSim->getJointVoltage(ljAnkleRight), leoSim->getJointVoltage(ljAnkleLeft);
    }
    double grlAutoActuateArm()
    {
      CSTGLeoSim *leoSim = dynamic_cast<CSTGLeoSim*>(mActuationInterface);
      CLeoBhWalkSym::autoActuateArm(leoSim);
      return leoSim->getJointVoltage(ljShoulder);
    }
    double grlAutoActuateKnee()
    {
      CSTGLeoSim *leoSim = dynamic_cast<CSTGLeoSim*>(mActuationInterface);
      CLeoBhWalkSym::autoActuateKnees(leoSim);
      return stanceLegLeft() ? leoSim->getJointVoltage(ljKneeLeft) : leoSim->getJointVoltage(ljKneeRight);
    }

  protected:
    CButterworthFilter<1>	mJointSpeedFilter[ljNumJoints];
};

/// Base class for simulated and real Leo
class LeoBaseEnvironment: public Environment
{
  public:
    LeoBaseEnvironment();
    ~LeoBaseEnvironment() {}

  protected:
    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Environment
    virtual LeoBaseEnvironment *clone() const;
    virtual void start(int test);
    virtual void step(double tau, double reward, int terminal);
    virtual void report(std::ostream &os);

    // Own
    void set_bh(CLeoBhBase *bh) { bh_ = bh; }
    
  protected:
    CSTGLeoSim leoSim_;
    CLeoState leoState_;
    Environment *target_env_;
    std::string xml_;
    ODESTGEnvironment *ode_;

    int observation_dims_, action_dims_;
    int target_observation_dims_, target_action_dims_;
    Vector target_obs_, target_action_;
    Vector observe_, actuate_;

    // Exporter
    Exporter *exporter_;
    int test_;
    double time_test_, time_learn_, time0_;

  protected:
    void config_parse_observations(Configuration &config);
    void config_parse_actions(Configuration &config);

    void fillObserve(const std::vector<CGenericStateVar> &genericStates,
                     const std::vector<std::string> &observeList,
                     Vector &out) const;

    void fillActuate(const std::vector<CGenericActionVar> &genericAction,
                     const std::vector<std::string> &actuateList,
                     Vector &out,
                     const std::string *req = NULL,
                     std::vector<int>  *reqIdx = NULL) const;

  private:
    CLeoBhBase *bh_; // makes it invisible in derived classes
};

}

#endif /* GRL_LEO_ENVIRONMENT_H_ */
