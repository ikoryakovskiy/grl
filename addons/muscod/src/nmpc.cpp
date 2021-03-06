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
}

static MUSCOD* muscod_muscod_ = NULL;
void NMPCPolicy::configure(Configuration &config)
{
  NMPCBase::configure(config);
  feedback_ = config["feedback"].str();
  n_iter_ = config["n_iter"];
  pub_error_signal_ = (VectorSignal*)config["pub_error_signal"].ptr();

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

  if (feedback_ == "threaded")
  {
    //------------------- Initialize NMPC ------------------- //
    // start NMPC controller in own thread running signal controlled event loop
    initialize_thread(
      thread_, muscod_run, nmpc_,
      problem_path, nmpc_model_name_,
      "",
      cond_iv_ready_, mutex_,
      verbose_, true, ninit_
    );

    //------------------ Initialize Controller ------------------ //
    // wait until iv_ready condition is fulfilled
    wait_for_iv_ready (nmpc_, verbose_);

    // NOTE we skip setting up controller here, because muscod_reset is called afterwards
    nmpc_->set_iv_ready(true);
    pthread_cond_signal(nmpc_->cond_iv_ready_); // Seems like the problem is here!

    // wait until iv_ready condition is fulfilled
    wait_for_iv_ready (nmpc_, verbose_);
    if (nmpc_->get_iv_ready() == true) {
    } else {
        std::cerr << "MAIN: bailing out ..." << std::endl;
        abort();
    }
  } else if (feedback_ == "non-threaded")
  {
    // initialize NMPCProblem instance
    if (nmpc_ == 0) {
        nmpc_ = new NMPCProblem(
            problem_path, nmpc_model_name_,
            // forward verbosity from grl
            verbose_
        );
    }

    if (nmpc_->m_muscod != 0) {
      if (verbose_) {
        std::cout << "MAIN: MUSCOD is already there!" << std::endl;
      }
      nmpc_->delete_MUSCOD ();
    }

    // initialize MUSCOD instance
    MUSCOD* muscod_ = NULL;

    if (nmpc_->thread_id.compare("") == 0) {
      std::cout << "MAIN: " << (void*) muscod_muscod_ << std::endl;
      if (muscod_muscod_ == 0) {
          if (verbose_) {
              std::cout << "MAIN: created MUSCOD instance!" << std::endl;
          }
          muscod_muscod_ = new MUSCOD(false);
          muscod_muscod_->setModelPathAndName(
            nmpc_->m_problem_path.c_str(),
            nmpc_->m_model_name.c_str()
          );
          muscod_muscod_->loadFromDatFile(NULL, NULL);
          muscod_muscod_->nmpcInitialize(0, NULL, NULL);
      } else {
          std::cout << "MAIN: MUSCOD is still there!" << std::endl;
      }
      muscod_ = muscod_muscod_;
    } else {
      std::cout << "MAIN: you are fucked! " << std::endl;
      abort();
    }

    // define MCData pointer
    MCData = &(muscod_->data);

    // forward verbosity from grl
    if (verbose_) {
        muscod_->setLogLevelAndFile(-1, NULL, NULL);
    } else {
        muscod_->setLogLevelTotal(-1);
    }

    // assign MUSCOD instance
    nmpc_->create_MUSCOD(muscod_);
  } else {
    ERROR (
      "I don't recognize feedback_ =" + feedback_ + ".\n"
      + "Possible modes are: non-threaded and threaded."
    );
  }

  // Allocate memory
  initial_sd_ = ConstantVector(nmpc_->NXD(), 0);
  initial_pf_ = ConstantVector(nmpc_->NP(), 0);
  initial_qc_ = ConstantVector(nmpc_->NU(), 0);
  final_sd_   = ConstantVector(nmpc_->NXD(), 0);
  initial_sd_prev_ = ConstantVector(nmpc_->NXD(), 0);
  initial_pf_prev_ = ConstantVector(nmpc_->NP(), 0);
  initial_qc_prev_ = ConstantVector(nmpc_->NU(), 0);

  grl_assert(nmpc_->NU() == action_max_.size());
  grl_assert(nmpc_->NU() == action_min_.size());

  // run single SQP iteration to be able to write a restart file
  // nmpc_->feedback();
  // nmpc_->transition();
  // nmpc_->preparation();

  // Save MUSCOD state
  // if (verbose_) {
  //   std::cout << "saving MUSCOD-II state to" << std::endl;
  //   std::cout << "  " << nmpc_->m_options->modelDirectory << restart_path_ << "/" << restart_name_ << ".bin" << std::endl;
  // }
  // nmpc_->m_muscod->writeRestartFile(
  //   restart_path_.c_str(), restart_name_.c_str()
  // );

  if (verbose_)
    std::cout << "MUSCOD-II is ready!" << std::endl;
}

void NMPCPolicy::reconfigure(const Configuration &config)
{
}

void NMPCPolicy::muscod_reset(const Vector &initial_obs, const Vector &initial_pf, Vector &initial_qc)
{
  if (feedback_ == "threaded")
  {
    //-------------------- Stop NMPC threads --------------------- //
    stop_thread(*nmpc_, &thread_, verbose_);

    //-------------------- Start NMPC threads -------------------- //

    nmpc_->m_quit = false;
    initialize_thread(
      thread_, muscod_run, nmpc_,
      nmpc_->m_problem_path, nmpc_->m_model_name,
      thread_id_,
      cond_iv_ready_, mutex_,
      verbose_, true, ninit_
    );

    //------------------ Initialize Controller ------------------ //

    // provide initial value and wait again
    // wait until iv_ready condition is fulfilled
    // nmpc_->set_iv_ready(false);
    iv_provided_ = true;
    provide_iv (
        nmpc_,
        initial_obs,
        initial_pf,
        &iv_provided_,
        true, // wait
        true
    );

    // wait until iv_ready condition is fulfilled
    wait_for_iv_ready (nmpc_, verbose_);
    if (nmpc_->get_iv_ready() == true) {
    } else {
        std::cerr << "MAIN: bailing out ..." << std::endl;
        abort();
    }
  } else if (feedback_ == "non-threaded")
  {
    if (nmpc_->m_muscod != 0) {
      if (verbose_) {
        std::cout << "MAIN: MUSCOD is already there!" << std::endl;
      }
      nmpc_->delete_MUSCOD ();
    }

    // initialize MUSCOD instance
    MUSCOD* muscod_ = NULL;

    if (nmpc_->thread_id.compare("") == 0) {
      std::cout << "MAIN: " << (void*) muscod_muscod_ << std::endl;
      if (muscod_muscod_ == 0) {
          if (verbose_) {
              std::cout << "MAIN: created MUSCOD instance!" << std::endl;
          }
          muscod_muscod_ = new MUSCOD(false);
          muscod_muscod_->setModelPathAndName(
            nmpc_->m_problem_path.c_str(),
            nmpc_->m_model_name.c_str()
          );
          muscod_muscod_->loadFromDatFile(NULL, NULL);
          muscod_muscod_->nmpcInitialize(0, NULL, NULL);
      } else {
          std::cout << "MAIN: MUSCOD is still there!" << std::endl;
      }
      muscod_ = muscod_muscod_;
    } else {
      std::cout << "MAIN: you are fucked! " << std::endl;
      abort();
    }

    // define MCData pointer
    MCData = &(muscod_->data);

    // forward verbosity from grl
    if (verbose_) {
        muscod_->setLogLevelAndFile(-1, NULL, NULL);
    } else {
        muscod_->setLogLevelTotal(-1);
    }

    // assign MUSCOD instance
    nmpc_->create_MUSCOD(muscod_);

    // initialize controller by performing some dry runs
    Vector sd = initial_obs;
    Vector pf = initial_pf;
    Vector qc = initial_qc;
    initialize_controller (*nmpc_, 10, sd, pf, &qc);
    initial_qc << qc;

  } else {
    ERROR (
      "I don't recognize feedback_ =" + feedback_ + ".\n"
      + "Possible modes are: non-threaded and threaded."
    );
  }

  sum_error_ = 0;
  sum_error_counter_ = 0;

  if (verbose_)
    std::cout << "MUSCOD is reseted!" << std::endl;
}

void NMPCPolicy::act(double time, const Observation &in, Action *out)
{
  grl_assert(in.v.size() >= nmpc_->NXD());

  // subdivide 'in' into state and setpoint, ignore temperature
  if (in.v.size() == nmpc_->NXD())
  {
    initial_sd_ << in.v;
  }
  else
  {
    if (in.v.size() == nmpc_->NXD() + 1)
    {
      initial_pf_ << in.v[in.v.size()-1]; // reference height
      initial_sd_ = in.v.block(0, 0, 1, in.v.size()-1); // remove indicator
    }
    else if (in.v.size() == nmpc_->NXD() + 2)
    {
      initial_pf_ << in.v[in.v.size()-1]; // reference height
      initial_sd_ = in.v.block(0, 0, 1, in.v.size()-2); // remove indicator
    }
  }

  if (verbose_)
  {
    std::cout << "time: [ " << time << " ]; state: [ " << initial_sd_ << "]" << std::endl;
    std::cout << "                          param: [ " << initial_pf_ << "]" << std::endl;
  }

  if (time == 0.0)
    muscod_reset(initial_sd_, initial_pf_, initial_qc_);

  // simulate model over specified time interval using NMPC internal model
  if ((pub_error_signal_) && (time != 0) && (feedback_ == "non-threaded"))
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
  //std::cout << "feedback = " << feedback_ << std::endl;
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
  } else if (feedback_ == "threaded")
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
  } else {
    ERROR (
      "I don't recognize feedback_ =" + feedback_ + ".\n"
      + "Possible modes are: non-threaded and threaded."
    );
  }

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

  // record variables for simulation
  initial_sd_prev_ = initial_sd_;
  initial_pf_prev_ = initial_pf_;
  initial_qc_prev_ = initial_qc_;
}

