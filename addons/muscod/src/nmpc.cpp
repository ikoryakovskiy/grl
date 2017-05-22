// *****************************************************************************
// Includes
// *****************************************************************************

#include <iostream>
#include <dlfcn.h>
#include <sys/stat.h>
#include <iomanip>

// GRL
#include <grl/policies/nmpc.h>

// MUSCOD-II interface
#include <wrapper.hpp>

using namespace grl;

REGISTER_CONFIGURABLE(NMPCPolicy);

NMPCPolicy::~NMPCPolicy()
{
  // stop threads
  if (nmpc_)
    stop_thread (*nmpc_, &thread_);

  // safely delete instances
  safe_delete(&nmpc_);
  safe_delete(&muscod_nmpc_);
}

void NMPCPolicy::request(ConfigurationRequest *config)
{
  NMPCBase::request(config);
  config->push_back(CRP("feedback", "Choose between a non-treaded and a threaded feedback of NMPC", feedback_, CRP::Configuration, {"non-threaded", "threaded"}));
  config->push_back(CRP("n_iter", "Number of iteration", (int)n_iter_, CRP::System, 0, INT_MAX));
  config->push_back(CRP("pub_error_signal", "signal/vector", "Publsher of the model-plant mismatch error signal", pub_error_signal_, true));
  config->push_back(CRP("pub_sim_state", "signal/vector", "Publsher of the simulated state signal", pub_sim_state_, true));
}

void NMPCPolicy::configure(Configuration &config)
{
  NMPCBase::configure(config);
  feedback_ = config["feedback"].str();
  n_iter_ = config["n_iter"];
  pub_error_signal_ = (VectorSignal*)config["pub_error_signal"].ptr();
  pub_sim_state_ = (VectorSignal*)config["pub_sim_state"].ptr();

  INFO("Running " << feedback_ << " implementation of NMPC");

  // Setup path for the problem description library and lua, csv, dat files used by it
  std::string problem_path  = model_path_ + "/" + model_name_;

  //-------------------- Load Lua model which is used by muscod ------------------- //
  if (!config["lua_model"].str().empty())
  {
    lua_model_ = problem_path + "/" + config["lua_model"].str();

    struct stat buffer;
    if (stat(lua_model_.c_str(), &buffer) != 0) // check if lua file exists in the problem description folder
      lua_model_ = std::string(RBDL_LUA_CONFIG_DIR) + "/" + config["lua_model"].str(); // if not, then use it as a reference from dynamics
  }
  else
    lua_model_ = "";
  //----------------- Set path in the problem description library ----------------- //
  setup_model_path(problem_path, nmpc_model_name_, lua_model_);

  //------------------- Initialize NMPC ------------------- //
  muscod_nmpc_ = new MUSCOD();
  if (verbose_) {
    muscod_nmpc_->setLogLevelAndFile(-1, NULL, NULL);
  } else {
    muscod_nmpc_->setLogLevelTotal(-1);
  }
  nmpc_ = new NMPCProblem(problem_path.c_str(), nmpc_model_name_.c_str(), muscod_nmpc_);
  if (verbose_) {
    nmpc_->m_verbose = true;
  } else {
    nmpc_->m_verbose = false;
  }

  // provide condition variable and mutex to NMPC instance
  nmpc_->cond_iv_ready_ = &cond_iv_ready_;
  nmpc_->mutex_ = &mutex_;

  // start NMPC controller in own thread running signal controlled event loop
  initialize_thread(
    &thread_, muscod_run, static_cast<void*> (nmpc_),
    &cond_iv_ready_, &mutex_, true
  );

  // Allocate memory
  initial_sd_ = ConstantVector(nmpc_->NXD(), 0);
  initial_pf_ = ConstantVector(nmpc_->NP(), 0);
  initial_qc_ = ConstantVector(nmpc_->NU(), 0);
  final_sd_   = ConstantVector(nmpc_->NXD(), 0);
  initial_sd_prev_ = ConstantVector(nmpc_->NXD(), 0);
  initial_pf_prev_ = ConstantVector(nmpc_->NP(), 0);
  initial_qc_prev_ = ConstantVector(nmpc_->NU(), 0);

  grl_assert(nmpc_->NU() == action_max_.size() || nmpc_->NU() == action_max_.size());
  grl_assert(nmpc_->NU() == action_min_.size() || nmpc_->NU() == action_max_.size());

  // run single SQP iteration to be able to write a restart file
  nmpc_->feedback();
  nmpc_->transition();
  nmpc_->preparation();

  // Save MUSCOD state
  if (verbose_) {
    std::cout << "saving MUSCOD-II state to" << std::endl;
    std::cout << "  " << nmpc_->m_options->modelDirectory << restart_path_ << "/" << restart_name_ << ".bin" << std::endl;
  }
  nmpc_->m_muscod->writeRestartFile(
    restart_path_.c_str(), restart_name_.c_str()
  );

  if (verbose_)
    std::cout << "MUSCOD-II is ready!" << std::endl;
}

void NMPCPolicy::reconfigure(const Configuration &config)
{
}

void NMPCPolicy::muscod_reset(const Vector &initial_obs, const Vector &initial_pf, Vector &initial_qc)
{
  // wait for preparation phase
  if (true) { // TODO Add wait flag
    wait_for_iv_ready(nmpc_, verbose_);
    if (nmpc_->get_iv_ready() == true) {
    } else {
        std::cerr << "MAIN: bailing out ..." << std::endl;
        abort();
    }
  }

  // restore muscod state
  if (verbose_) {
    std::cout << "restoring MUSCOD-II state to" << std::endl;
    std::cout << "  " << nmpc_->m_options->modelDirectory << restart_path_ << "/" << restart_name_ << ".bin" << std::endl;
  }

  nmpc_->m_muscod->readRestartFile(restart_path_.c_str(), restart_name_.c_str());
  nmpc_->m_muscod->nmpcInitialize (
      4,  // 4 for restart
      restart_path_.c_str(), restart_name_.c_str()
  );

  // initialize NMPC
  for (int inmpc = 0; inmpc < 10; ++inmpc)
  {
    // 1) Feedback: Embed parameters and initial value from MHE
    if (initFeedback_) {
      nmpc_->feedback(initial_obs, initial_pf, &initial_qc);
    } else {
      nmpc_->feedback();
    }
    // 2) Transition
    nmpc_->transition();
    // 3) Preparation
    nmpc_->preparation();
  }

  // NOTE: both flags are set to true then iv is provided and
  //       qc is is computed
  // NOTE: due to waiting flag, main thread is on hold until
  //       computations are finished (<=2ms!)
  iv_provided_ = true;
  qc_retrieved_ = true;

  // wait for preparation phase
  if (true) { // TODO Add wait flag
    wait_for_iv_ready(nmpc_, verbose_);
    if (nmpc_->get_iv_ready() == true) {
    } else {
        std::cerr << "MAIN: bailing out ..." << std::endl;
        abort();
    }
  }

  sum_error_ = 0;
  sum_error_counter_ = 0;

  if (verbose_)
    std::cout << "MUSCOD is reseted!" << std::endl;
}

void NMPCPolicy::muscod_reset(const Vector &initial_obs, Vector &initial_qc)
{
  // wait for preparation phase
  if (true) { // TODO Add wait flag
    wait_for_iv_ready(nmpc_, verbose_);
    if (nmpc_->get_iv_ready() == true) {
    } else {
        std::cerr << "MAIN: bailing out ..." << std::endl;
        abort();
    }
  }

  // restore muscod state
  if (verbose_) {
    std::cout << "restoring MUSCOD-II state to" << std::endl;
    std::cout << "  " << nmpc_->m_options->modelDirectory << restart_path_ << "/" << restart_name_ << ".bin" << std::endl;
  }

  nmpc_->m_muscod->readRestartFile(restart_path_.c_str(), restart_name_.c_str());
  nmpc_->m_muscod->nmpcInitialize (
      4,  // 4 for restart
      restart_path_.c_str(), restart_name_.c_str()
  );

  // initialize NMPC
  for (int inmpc = 0; inmpc < 20; ++inmpc)
  {
    // 1) Feedback: Embed parameters and initial value from MHE
    if (initFeedback_) {
      nmpc_->feedback(initial_obs, &initial_qc);
    } else {
      nmpc_->feedback();
    }
    // 2) Transition
    nmpc_->transition();
    // 3) Preparation
    nmpc_->preparation();
  }

  // NOTE: both flags are set to true then iv is provided and
  //       qc is is computed
  // NOTE: due to waiting flag, main thread is on hold until
  //       computations are finished (<=2ms!)
  iv_provided_ = true;
  qc_retrieved_ = true;

  // wait for preparation phase
  if (true) { // TODO Add wait flag
    wait_for_iv_ready(nmpc_, verbose_);
    if (nmpc_->get_iv_ready() == true) {
    } else {
        std::cerr << "MAIN: bailing out ..." << std::endl;
        abort();
    }
  }

  sum_error_ = 0;
  sum_error_counter_ = 0;

  if (verbose_)
    std::cout << "MUSCOD is reseted!" << std::endl;
}

void NMPCPolicy::act(double time, const Observation &in, Action *out)
{
  grl_assert((in.v.size() == nmpc_->NXD() + nmpc_->NP()) || (in.v.size() == nmpc_->NXD()));

  // subdivide 'in' into state and setpoint
  if (in.v.size() == nmpc_->NXD() + nmpc_->NP())
  {
    initial_sd_ << in.v.block(0, 0, 1, nmpc_->NXD());
    initial_pf_ << in.v.block(0, nmpc_->NXD(), 1, nmpc_->NP());
  } else {
    initial_sd_ << in.v;
  }

  if (verbose_)
  {
    std::cout << "time: [ " << time << " ]; state: [ " << initial_sd_ << "]" << std::endl;
    std::cout << "                          param: [ " << initial_pf_ << "]" << std::endl;
  }

  if (time == 0.0)
    muscod_reset(initial_sd_, initial_pf_, initial_qc_);

  // simulate model over specified time interval using NMPC internal model
  if (pub_error_signal_ && time != 0)
  {
    double time_interval = 0.03; //nmpc_->getSamplingRate();
    nmpc_->simulate(
        initial_sd_prev_, // state from previous iteration
        initial_pf_prev_, // parameter from previous iteration
        initial_qc_prev_, // control from previous iteration
        time_interval,
        &final_sd_        // state from comparison with input
    );
    Vector x = final_sd_.block(0, 0, 1, nmpc_->NU()) - initial_sd_.block(0, 0, 1, nmpc_->NU());
    double error = sqrt(x.cwiseProduct(x).sum());
    sum_error_ += error;
    sum_error_counter_++;
    pub_error_signal_->set(ConstantVector(nmpc_->NU(), sum_error_/sum_error_counter_));
    //std::cout << "Model-plant error " << pub_error_signal_->get() << std::endl;
  }

  out->v.resize( nmpc_->NU() );
  if (feedback_ == "non-threaded")
  {
    for (int inmpc = 0; inmpc < n_iter_; ++inmpc) {
      //std::cout << "NON-THREADED VERSION!" << std::endl;
      // 1) Feedback: Embed parameters and initial value from MHE
      // NOTE the same initial values (sd, pf) are embedded several time,
      //      but this will result in the same solution as running a MUSCOD
      //      instance for several iterations
      nmpc_->feedback(initial_sd_, initial_pf_, &initial_qc_);
      // 2) Shifting
      // NOTE do that only once at last iteration
      // NOTE this has to be done before the transition phase
      if (n_iter_ > 0 && inmpc == n_iter_-1) {
        nmpc_->shifting(1);
      }
      // 3) Transition
      nmpc_->transition();
      // 4) Preparation
      nmpc_->preparation();
    }
    // } // END FOR NMPC ITERATIONS
  }

  if (feedback_ == "threaded")
  {
    for (int inmpc = 0; inmpc < n_iter_; ++inmpc) {
        // std::cout << "THREADED VERSION!" << std::endl;
        // 1) Feedback: Embed parameters and initial value from SIMULATION
        // establish IPC communication to NMPC thread


        // NOTE do that only once at last iteration
        // NOTE this has to be done before the transition phase
        if (n_iter_ > 0 && inmpc == n_iter_ - 1) {
          nmpc_->set_shift_mode (1);
        } else {
          nmpc_->set_shift_mode (-1);
        }

        // provide iv to thread, get time until thread was ready
        iv_provided_ = true;
        provide_iv (nmpc_, initial_sd_, initial_pf_, &iv_provided_, false, verbose_);

        // provide iv to thread, get time until thread was ready
        if (iv_provided_)
          retrieve_qc (nmpc_, &initial_qc_, &qc_retrieved_, true, verbose_);

        // wait for preparation phase
        if (false) { // TODO Add wait flag
          wait_for_iv_ready(nmpc_, verbose_);
          if (nmpc_->get_iv_ready() == true) {
          } else {
              std::cerr << "MAIN: bailing out ..." << std::endl;
              abort();
          }
        }
    } // END FOR NMPC ITERATIONS
  }

  // Here we can return the feedback control
  // NOTE feedback control is cut of at action limits 'action_min/max'
  for (int i = 0; i < action_min_.size(); i++)
  {
    out->v[i] = fmax( fmin(initial_qc_[i], action_max_[i]) , action_min_[i]);
    if (out->v[i] != initial_qc_[i])
      WARNING("NMPC action " << i << " was truncated");
    initial_qc_[i] = out->v[i];
  }

  out->type = atGreedy;

  if (verbose_)
    std::cout << "Feedback Control: [" << out->v << "]" << std::endl;

  // simulate model over specified time interval using NMPC internal model
  if (pub_sim_state_)
  {
    double time_interval = 0.03; //nmpc_->getSamplingRate();
    nmpc_->simulate(
        initial_sd_,    // current state
        initial_pf_,    // current parameter
        initial_qc_,    // applied control
        time_interval,
        &final_sd_      // next state
    );
    pub_sim_state_->set(final_sd_);
  }

  // record variables for simulation
  initial_sd_prev_ = initial_sd_;
  initial_pf_prev_ = initial_pf_;
  initial_qc_prev_ = initial_qc_;
}

