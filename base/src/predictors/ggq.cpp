#include <grl/predictors/ggq.h>

using namespace grl;

REGISTER_CONFIGURABLE(GGQPredictor)

void GGQPredictor::request(ConfigurationRequest *config)
{
}

void GGQPredictor::configure(Configuration &config)
{
  projector_ = (Projector*)config["projector"].ptr();
  theta_ = (Representation*)config["theta"].ptr();
  w_ = (Representation*)config["w"].ptr();
  policy_ = (QPolicy*)config["policy"].ptr();
  
  alpha_ = config["alpha"];
  eta_ = config["eta"];
  gamma_ = config["gamma"];
  lambda_ = config["lambda"];
}

void GGQPredictor::reconfigure(const Configuration &config)
{
}

GGQPredictor *GGQPredictor::clone() const
{
  return NULL;
}

void GGQPredictor::update(const Transition &transition)
{
  Vector v;
  
  // phi (actual taken action)
  ProjectionPtr phi = projector_->project(transition.prev_obs, transition.prev_action);
  
  // phi_next for greedy target policy
  Vector action;
  policy_->act(transition.obs, &action);
  ProjectionPtr phi_next = projector_->project(transition.obs, action);

  // temporal difference error
  double delta = transition.reward + gamma_*theta_->read(phi_next, &v) - theta_->read(phi, &v);

  // w^Tphi
  double dotwphi = w_->read(phi, &v);

  // Update regular weights
  theta_->update(phi, VectorConstructor(alpha_*delta));
  theta_->update(phi_next, VectorConstructor(-alpha_*gamma_*(1-lambda_)*dotwphi));
  
  // Update extra weights
  w_->update(phi, VectorConstructor(alpha_*eta_*(delta - dotwphi)));
}

void GGQPredictor::finalize()
{
}