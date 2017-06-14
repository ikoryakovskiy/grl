// *****************************************************************************
// Includes
// *****************************************************************************

#include <iostream>
#include <dlfcn.h>
#include <sys/stat.h>
#include <iomanip>

// GRL
#include <grl/policies/nmpc_mlrti.h>

// MUSCOD-II interface
#include <wrapper.hpp>

using namespace grl;

REGISTER_CONFIGURABLE(NMPCPolicyMLRTI);

NMPCPolicyMLRTI::~NMPCPolicyMLRTI()
{
  // release pointer on instances
  cntl_ = NULL;
  idle_ = NULL;

  // stop threads
  if (nmpc_A_)
    stop_thread (*nmpc_A_, &thread_A_, true);
  if (nmpc_B_)
    stop_thread (*nmpc_B_, &thread_B_, true);

  // safely delete instances
  safe_delete(&nmpc_A_);
  safe_delete(&nmpc_B_);
}

void NMPCPolicyMLRTI::request(ConfigurationRequest *config)
{
  NMPCBase::request(config);
}

void NMPCPolicyMLRTI::configure(Configuration &config)
{
  NMPCBase::configure(config);

  // Setup path for the problem description library and lua, csv, dat files used by it
  std::string problem_path  = model_path_ + "/" + model_name_;

  //-------------------- Load Lua model which is used by muscod ------------------- //
  lua_model_ = problem_path + "/" + config["lua_model"].str();

  struct stat buffer;
  if (stat(lua_model_.c_str(), &buffer) != 0) // check if lua file exists in the problem description folder
    lua_model_ = std::string(RBDL_LUA_CONFIG_DIR) + "/" + config["lua_model"].str(); // if not, then use it as a reference from dynamics

  //----------------- Set path in the problem description library ----------------- //
  setup_model_path(problem_path, nmpc_model_name_, lua_model_);

  //------------------- Initialize NMPC thread A ------------------- //
  // start NMPC controller in own thread running signal controlled event loop
  initialize_thread(
    thread_A_, muscod_run, nmpc_A_,
    problem_path, nmpc_model_name_,
    thread_id_A,
    cond_iv_ready_A_, mutex_A_,
    verbose_, true
  );

  //------------------- Initialize NMPC thread B ------------------- //
  // start NMPC controller in own thread running signal controlled event loop
  initialize_thread(
    thread_B_, muscod_run, nmpc_B_,
    problem_path, nmpc_model_name_,
    thread_id_B,
    cond_iv_ready_B_, mutex_B_,
    verbose_, true
  );

  //------------------- Define state of MLRTI NMPC ------------------- //

  // use pointers to identify different controllers
  // NOTE cntl_ means LMPC at current set-point
  // NOTE idle_ means NMPC feedback before going into re-linearization
  cntl_ = nmpc_A_; // LMPC controller
  idle_ = nmpc_B_; // NMPC controller

  // set proper mode of controller
  // NOTE nmpc_mode 0 => provide feedback but then re-linearize controller
  // NOTE nmpc_mode 1 => provide linear feedback only, no re-linearization
  idle_->set_nmpc_mode(0);
  cntl_->set_nmpc_mode(1);

  // define current state
  current_state_ = idle_call;

  //------------------- Initialize NMPC data ------------------- //

  // Allocate memory
  //initial_sd_ = ConstantVector(nmpc_A_->NXD(), 0);
  initial_pf_ = ConstantVector(nmpc_A_->NP(), 0);
  initial_qc_ = ConstantVector(nmpc_A_->NU(), 0);
  final_sd_   = ConstantVector(nmpc_A_->NXD(), 0);

  // Save MUSCOD state
  if (verbose_) {
    std::cout << "saving MUSCOD-II state to" << std::endl;
    std::cout << "  " << nmpc_A_->get_model_directory() << restart_path_ << "/" << restart_name_ << ".bin" << std::endl;
  }
  // nmpc_A_->writeRestartFile(restart_path_, restart_name_);

  // Muscod params
  initFeedback_ = config["initFeedback"];

  if (verbose_) {
    std::cout << "MUSCOD is ready!" << std::endl;
  }
}

void NMPCPolicyMLRTI::reconfigure(const Configuration &config)
{
}


void NMPCPolicyMLRTI::muscod_reset(const Vector &initial_obs, double time)
{
  //-------------------- Stop MLRTI NMPC threads --------------------- //
  stop_thread(*nmpc_A_, &thread_A_, verbose_);
  stop_thread(*nmpc_B_, &thread_B_, verbose_);
  //-------------------- Start MLRTI NMPC threads -------------------- //

  nmpc_A_->m_quit = false;
  initialize_thread(
    thread_A_, muscod_run, nmpc_A_,
    nmpc_A_->m_problem_path, nmpc_A_->m_model_name,
    thread_id_A,
    cond_iv_ready_A_, mutex_A_,
    verbose_, true
  );

  nmpc_B_->m_quit = false;
  initialize_thread(
    thread_B_, muscod_run, nmpc_B_,
    nmpc_B_->m_problem_path, nmpc_B_->m_model_name,
    thread_id_B,
    cond_iv_ready_B_, mutex_B_,
    verbose_, true
  );

  //------------------- Define state of MLRTI NMPC ------------------- //

  // use pointers to identify different controllers
  // NOTE cntl_ means LMPC at current set-point
  // NOTE idle_ means NMPC feedback before going into re-linearization
  cntl_ = nmpc_A_; // LMPC controller
  idle_ = nmpc_B_; // NMPC controller

  // set proper mode of controller
  // NOTE nmpc_mode 0 => provide feedback but then re-linearize controller
  // NOTE nmpc_mode 1 => provide linear feedback only, no re-linearization
  idle_->set_nmpc_mode(0);
  cntl_->set_nmpc_mode(1);

  // define current state
  current_state_ = idle_call;

  sum_error_ = 0;
  sum_error_counter_ = 0;

  if (verbose_)
    std::cout << "MUSCOD is reseted!" << std::endl;
}

void NMPCPolicyMLRTI::act(double time, const Observation &in, Action *out)
{
  grl_assert(in.v.size() == nmpc_A_->NXD() + 1); // setpoint indicator

  // reference height
  initial_pf_ << in.v[in.v.size()-1];

  // remove indicator
  Vector initial_sd_ = in.v.block(0, 0, 1, in.v.size()-1);

  if (time == 0.0)
    muscod_reset(initial_sd_, time);

  if (verbose_)
  {
    std::cout << "time: [ " << time << " ]; state: [ " << initial_sd_ << "]" << std::endl;
    std::cout << "                          param: [ " << initial_pf_ << "]" << std::endl;
  }

  NMPCProblem* tmp_nmpc;

  // switch statement implementing the above mentioned finite state machine
  // 0: idle_call
  //    call idle_ at current state, retrieve feedback, start
  //    re-linearization
  //    state -> 1 (idle_ready)
  // 1: idle_ready:
  //    while waiting for the preparation phase of idle_, provide feedback
  //    from cntl_, if idle_ is ready state -> 2 (idle_switch)
  // 2: idle_switch
  //    switch controllers idle_ <-> cntl_
  //    state -> 0 (idle_call)
  switch (current_state_) {
    case idle_call:
      if (verbose_)
      {
        std::cout << "MAIN: STATE: IDLE_CALL " << std::endl;
      }

      // NOTE: both flags are set to true then iv is provided and
      //       qc is is computed
      // NOTE: due to waiting flag, main thread is on hold until
      //       computations are finished (<=2ms!)
      idle_iv_provided_ = true;
      idle_qc_retrieved_ = true;

      // establish IPC communication to NMPC thread
      get_feedback (
        idle_,
        initial_sd_,
        initial_pf_,
        &initial_qc_,
        &idle_iv_provided_,
        &idle_qc_retrieved_,
        // NOTE: we use wait flag here to guarantee separation of
        //       feedback phases
        true // wait flag
      );

      // handle return values from thread
      // NOTE iv_provided and qc_retrieved shall be true!
      if (idle_iv_provided_ == true) {
          if (verbose_)
          {
            std::cout << "MAIN: Provided initial values to thread!" << std::endl;
          }
      } else {
        std::cerr << "MAIN: Providing initial values to thread was not possible!" << std::endl;
        abort();
      }
      if (idle_qc_retrieved_ == true) {
          if (verbose_)
          {
            std::cout << "MAIN: Retrieved feedback controls!" << std::endl;
          }
      } else {
        std::cerr << "MAIN: Retrieving feedback controls was not possible!" << std::endl;
        abort();
      }

      if (verbose_)
      {
        std::cout << "MAIN: timing statistics:" << std::endl;
        std::cout << idle_->timing._timing << std::endl;
      }

      // store entries for later analysis and write to file
      timing_values_idle_.push_back(idle_->timing._timing);
      timing_values_.push_back(idle_->timing._timing);

      // copy idle_ timer states to ttimer
      ttimer_ = idle_->timing;

      // idle_ is successfully idled, feedback
      if (verbose_)
      {
        std::cout << "MAIN: idled!" << std::endl;
      }
      // change state:
      //   state -> 1 (idle_ready)
      current_state_ = idle_ready;

      // break statement of switch case
      break; //optional

  // idle_ (NMPC) controller is idled and while waiting linear feedback
  // is provided
  case idle_ready:
    if (verbose_)
    {
      std::cout << "MAIN: STATE: IDLE_READY " << std::endl;
    }

    // NOTE: both flags are set to true then iv is provided and
    //       qc is is computed
    // NOTE: due to waiting flag, main thread is on hold until
    //       computations are finished (<=2ms!)
    cntl_iv_provided_ = true;
    cntl_qc_retrieved_ = true;

    // establish IPC communication to NMPC thread
    get_feedback (
      cntl_, // LMPC controller
      initial_sd_,
      initial_pf_,
      &initial_qc_,
      &cntl_iv_provided_,
      &cntl_qc_retrieved_,
      // NOTE: we use wait flag here to guarantee separation of
      //       feedback phases
      true // wait flag
    );

    // handle return values from thread
    // NOTE iv_provided and qc_retrieved shall be true!
    // FIXME is this necessary?
    if (cntl_iv_provided_ == true) {
          if (verbose_)
          {
            std::cout << "MAIN: Provided initial values to thread!" << std::endl;
          }
    } else {
      std::cerr << "MAIN: Providing initial values to thread was not possible!" << std::endl;
      abort();
    }
    if (cntl_qc_retrieved_ == true) {
          if (verbose_)
          {
            std::cout << "MAIN: Retrieved feedback controls!" << std::endl;
          }
    } else {
      std::cerr << "MAIN: Retrieving feedback controls was not possible!" << std::endl;
      abort();
    }

    if (verbose_)
    {
      std::cout << "MAIN: timing statistics:" << std::endl;
      std::cout << cntl_->timing._timing << std::endl;
    }

    // store entries for later analysis and write to file
    timing_values_cntl_.push_back(cntl_->timing._timing);
    timing_values_.push_back(cntl_->timing._timing);

    // copy idle_ timer states to ttimer
    ttimer_ = cntl_->timing;

    if (idle_->get_iv_ready()) {
          if (verbose_)
          {
                  std::cout << std::endl;
                  std::cout << "MAIN: IDLE preparation finished!" << std::endl;
                  std::cout << std::endl;
          }

        // change state:
        //   state -> 2 (idle_switch)
        current_state_ = idle_switch;
    }
    // break statement of switch case
    break; //optional

  // when idle_ (NMPC) controller is finished switch controllers and
  // reset state machine
  case idle_switch:
    if (verbose_)
    {
      std::cout << "MAIN: STATE: IDLE_SWTICH" << std::endl;
    }

    // use pointers to identify different controllers
    // switch controllers cntl_ <-> idle_
    tmp_nmpc = cntl_;
    cntl_ = idle_;
    idle_ = tmp_nmpc;

    // set proper mode of controller
    // NOTE nmpc_mode 0 => provide feedback but then re-linearize controller
    // NOTE nmpc_mode 1 => provide linear feedback only, no re-linearization
    idle_->set_nmpc_mode(0);
    cntl_->set_nmpc_mode(1);

    if (verbose_)
    {
      std::cout << std::endl;
      std::cout << "MAIN: controllers switched!" << std::endl;
      std::cout << std::endl;
    }

    // change state:
    //   state -> 0 (idle_call)
    current_state_ = idle_call;
    this->act(time, in, out);
    break; //optional

  default : //Optional
    std::cerr << "MAIN: ERROR: WRONG STATE!" << std::endl;
  }

  // Here we can return the feedback control
  out->v.resize( initial_qc_.size() );
  for (int i = 0; i < action_min_.size(); i++)
  {
    out->v[i] = fmax( fmin(initial_qc_[i], action_max_[i]) , action_min_[i]);
    if (out->v[i] != initial_qc_[i])
      WARNING("NMPC MLRTI action " << i << " was truncated");
    initial_qc_[i] = out->v[i];
  }
  out->type = atGreedy;

  if (verbose_)
    std::cout << "Feedback Control: [" << out->v << "]" << std::endl;

  // record variables for simulation
  initial_sd_prev_ = initial_sd_;
  initial_pf_prev_ = initial_pf_;
  initial_qc_prev_ = initial_qc_;
}

