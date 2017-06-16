/** \file nmpc_base.cpp
 * \brief Base class for NMPC.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2016-09-15
 *
 * \copyright \verbatim
 * Copyright (c) 2015, Wouter Caarls
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

#include <iostream>
#include <dlfcn.h>
#include <sys/stat.h>
#include <iomanip>
#include <unistd.h>

// GRL
#include <grl/policies/nmpc_mlrti.h>

// MUSCOD-II interface
#include <wrapper.hpp>

using namespace grl;


#include <grl/policies/nmpc_base.h>

//-------------------------------------- GRL -----------------------------------//
void NMPCBase::request(ConfigurationRequest *config)
{
  config->push_back(CRP("verbose", "Verbose mode", (int)verbose_, CRP::System, 0, 1));
  config->push_back(CRP("initFeedback", "Initialize feedback", (int)initFeedback_, CRP::System, 0, 1));
  config->push_back(CRP("action_min", "vector.action_min", "Lower limit on actions", action_min_, CRP::System));
  config->push_back(CRP("action_max", "vector.action_max", "Upper limit on actions", action_max_, CRP::System));
  config->push_back(CRP("lua_model", "Lua model used by MUSCOD", lua_model_));
  config->push_back(CRP("model_name", "Name of the model in grl", model_name_));
  config->push_back(CRP("nmpc_model_name", "Name of MUSCOD NMPC model library", nmpc_model_name_));
}

void NMPCBase::configure(Configuration &config)
{
  model_path_       = std::string(MUSCOD_CONFIG_DIR);
  nmpc_model_name_  = config["nmpc_model_name"].str();
  model_name_       = config["model_name"].str();
  verbose_          = config["verbose"];
  action_min_       = config["action_min"].v();
  action_max_       = config["action_max"].v();

  if (action_min_.size() != action_max_.size())
    throw bad_param("policy/nmpc:{action_min, action_max}");
}

void *NMPCBase::setup_model_path(const std::string path, const std::string model, const std::string lua_model)
{
  // get the library handle,
  std::string so_path  = path + "/" + "lib" + model + ".so";
  void *so_handle = dlopen(so_path.c_str(), RTLD_NOW|RTLD_GLOBAL);
  if (so_handle==NULL)
  {
    std::cout << "ERROR: Could not load MUSCOD-II shared model library: '" << so_path << "'" << std::endl;
    std::cout << "dlerror responce: " << dlerror() << std::endl;
    std::cout << "bailing out ..." << std::endl;
    exit(EXIT_FAILURE);
  }

  // get the function handle
  void (*so_set_path)(std::string, std::string);
  std::string so_set_path_fn = "set_path"; // name of a function which sets the path
  so_set_path = (void (*)(std::string, std::string)) dlsym(so_handle, so_set_path_fn.c_str());
  if (so_set_path==NULL)
  {
    std::cout << "ERROR: Could not symbol in shared library: '" << so_set_path_fn << "'" << std::endl;
    std::cout << "bailing out ..." << std::endl;
    std::exit(-1);
  }

  // ... and finally set the paths
  if (verbose_)
  {
    std::cout << "MUSCOD: setting new problem path to: '" << path << "'" <<std::endl;
    std::cout << "MUSCOD: setting new Lua model file to: '" << lua_model << "'" <<std::endl;
  }
  so_set_path(path, lua_model);

  return so_handle;
}

//-------------------------------------- EXPORT -----------------------------------//
/*
std::string path = "/home/ivan/work/Project/Software/grl/qt-build/leo_squat/mlrti_timing/";

void erase_file(const std::string& file_name)
{
  // erase contents if it exists
  std::ofstream ofs;
  ofs.open(file_name, std::ofstream::out | std::ofstream::trunc);
  if (!ofs) {
    std::cerr << "Error opening file " << file_name << std::endl;
    abort();
  }
  ofs.close();

  return;
}

void write_header_to_file(const std::string& data_file)
{
  std::ofstream data_stream;
  data_stream.open (data_file, std::ios_base::trunc);
  if (!data_stream) {
    std::cerr << "Error opening file " << data_file << std::endl;
    abort();
  }

  // configure data stream
  data_stream << std::setprecision(18) << std::scientific;

  // add header to file:
  // # feedback, transition, shift, preparation, simulation
  data_stream << "#" << " ";
  //data_stream << "time" << ", ";
  data_stream << "feedback" << ", ";
  data_stream << "transition" << ", ";
  data_stream << "shift" << ", ";
  data_stream << "prepare" << ", ";
  data_stream << "preparation" << ", ";
  data_stream << "total";
  data_stream << std::endl;

  // finally close file
  data_stream.close();
}


void append_timing_to_file(Vector d_values, const std::string& data_file)
{
  std::ofstream data_stream;
  data_stream.open (data_file, std::ios_base::app);
  if (!data_stream) {
    std::cerr << "Error opening file " << data_file << std::endl;
    abort();
  }

  // configure data stream
  data_stream << std::setprecision(18) << std::scientific;
  //data_stream << time << ", ";

  for (unsigned int j = 0; j < d_values.size(); j++) {
    data_stream << d_values[j];
    if (j < d_values.size() -1 ) {
      data_stream << ", ";
    }
  }
  data_stream << std::endl;

    // finally close file
    data_stream.close();
}
*/

std::string path = "";
void erase_file(const std::string& file_name){}
void write_header_to_file(const std::string& data_file) {}
void append_timing_to_file(Vector d_values, const std::string& data_file) {}

//-------------------------------------- MUSCOD -----------------------------------//
namespace grl
{

double stop_watch(){
    static struct timespec timer;
    if (clock_gettime (CLOCK_REALTIME, &timer) != 0) return -1;
    return (double) timer.tv_sec + 1.0E-9 * (double) timer.tv_nsec;
}

// clean-up MUSCOD-II main thread
void stop_thread (NMPCProblem& data,
    pthread_t* muscod_thread,
    bool verbose
) {
    bool wait = true;
    unsigned int cnt;
    if (wait) {
        cnt = 0;

        // debug message
        if (verbose) {
            std::cout << "MAIN: Waiting for thread to get ready!" << std::endl;
        }

        // wait until thread is ready, i.e. cond_iv_ready_ condition is established
        while (!data.get_iv_ready()) {
            if (cnt > 10000) {
                if (verbose) {
                    std::cerr << "MAIN: thread frozen!" << std::endl;
                }
                break;
            }
            usleep(100); // wait for 0.1 ms
            cnt += 1;
        }

        if (verbose) {
            std::cout << "MAIN: Waiting for " << (double) cnt * 0.1;
            std::cout << "ms for thread!" << std::endl;
        }

        // assure thread is ready
        if (data.get_iv_ready() == false) {
            std::cerr << "MAIN: bailing out ..." << std::endl;
            abort();
        }
    }

    //release the thread from waiting if needed
    pthread_mutex_lock(data.mutex_);
    data.m_quit = true; // send quit signal
    pthread_cond_signal(data.cond_iv_ready_); // send cond_iv_ready signal
    pthread_mutex_unlock(data.mutex_);

    // join thread, which closes the thread
    pthread_join(*muscod_thread, NULL);

    // destroy mutexes, because reinitialization without destroy results in undefined behaviour
    pthread_mutex_destroy(data.mutex_);
    pthread_cond_destroy (data.cond_iv_ready_);

    data.iv_ready_ = false;
    data.qc_ready_ = false;
}

// run MUSCOD-II NMPC for several iterations to initialize controller for
// current initial sd/pf, first qc is optional
void initialize_controller (
    MUSCODProblem& nmpc, const int nmpc_ninit,
    Vector& sd, Vector& pf,
    Vector* qc
) {
    // initialize NMPC by nmpc_init SQP iterations
    for (int inmpc = 0; inmpc < nmpc_ninit; ++inmpc) {
        // 1) Feedback: Embed parameters and initial value
        nmpc.feedback(sd, pf, qc);
        // 2) Transition
        nmpc.transition();
        // 3) Preparation
        nmpc.preparation();
    }
}

void initialize_thread(
    pthread_t& muscod_thread,
    void* (*function) (void*) ,
    NMPCProblem*& nmpc,
    std::string problem_path,
    std::string nmpc_model_name,
    const std::string thread_id,
    pthread_cond_t& cond_iv_ready,
    pthread_mutex_t& mutex,
    bool grl_verbose,
    bool verbose
) {
    // initialize NMPCProblem instance
    if (nmpc == 0) {
        nmpc = new NMPCProblem(
            problem_path, nmpc_model_name,
            // forward verbosity from grl
            grl_verbose
        );
    }
    // assign ID to thread
    nmpc->thread_id = thread_id;

    // provide condition variable and mutex to NMPC instance
    nmpc->cond_iv_ready_ = &cond_iv_ready;
    nmpc->mutex_ = &mutex;

    // initialize mutex and condition variable
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init (&cond_iv_ready, NULL);

    // start MUSCOD-II in a thread running a signal triggered execution loop
    pthread_mutex_lock(&mutex); // LOCK
    if (verbose) {
        std::cout << "In " << __func__ << ": creating MUSCOD-II thread" << std::endl;
    }

    // create thread running execution loop
    int rc = pthread_create (
        &muscod_thread, NULL, function, static_cast<void*> (nmpc)
    );

    // error message on error (rc > 0 if error happened!)
    if (rc) {
      std::cerr << "ERROR: pthread_create() failed with " << rc << std::endl;
      std::cerr << "bailing out..." << rc << std::endl;
      abort();
    }

    // wait for MUSCOD thread to initialize data structure
    if (verbose) {
        std::cout << "MUSCOD: Waiting for MUSCOD thread..." << std::endl;
    }
    pthread_cond_wait(&cond_iv_ready, &mutex); // WAIT FOR SIGNAL FROM THREAD
    pthread_mutex_unlock(&mutex);  // UNLOCK
}

static MUSCOD* muscod_muscod_ = NULL;
static MUSCOD* muscod_A_ = NULL;
static MUSCOD* muscod_B_ = NULL;

// MUSCOD-II main thread setup and execution loop
void *muscod_run (void *indata)
{
    // timing variables
    double tic, tac, ttac, ptac;

    // cast data back to object
    if (indata == NULL) {
        std::cerr << "ERROR: indata is NULL" << std::endl;
        std::cerr << "bailing out..." << std::endl;
        abort();
    }
    NMPCProblem& nmpc = *static_cast<NMPCProblem*> (indata);

    pthread_mutex_lock(nmpc.mutex_);

    // retrieve verbose flag from controller
    bool verbose = nmpc.m_verbose;

    // retrieve thread identifier from NMPC instance
    std::string thread_id = nmpc.thread_id;
    const std::string timing = path + "timing_" + thread_id + ".csv";
    erase_file(timing);
    write_header_to_file(timing);

    pthread_mutex_unlock(nmpc.mutex_);

    if (verbose) {
        if (!thread_id.empty()) {

            std::cout << "THREAD '" << thread_id << "': got thread id! " << std::endl;
        }
    }

    verbose = true;
    // initialize MUSCOD instance
    MUSCOD* muscod_ = NULL;

    std::cout << "THREAD '" << thread_id << "': muscod_muscod_ = " << (void*) muscod_muscod_ << std::endl;
    std::cout << "THREAD '" << thread_id << "': muscod_A_      = " << (void*) muscod_A_ << std::endl;
    std::cout << "THREAD '" << thread_id << "': muscod_B_      = " << (void*) muscod_B_ << std::endl;
    std::cout << "THREAD '" << thread_id << "': muscod_        = " << (void*) muscod_ << std::endl;

    if (thread_id.compare("") == 0) {
      std::cout << "THREAD '" << thread_id << "': " << (void*) muscod_muscod_ << std::endl;
      if (muscod_muscod_ == 0) {
          if (verbose) {
              std::cout << "THREAD '" << thread_id << "': created MUSCOD instance!" << std::endl;
          }
          muscod_muscod_ = new MUSCOD(false);
          muscod_muscod_->setModelPathAndName(nmpc.m_problem_path.c_str(), nmpc.m_model_name.c_str());
          muscod_muscod_->loadFromDatFile(NULL, NULL);
          muscod_muscod_->nmpcInitialize(0, NULL, NULL);
      } else {
          std::cout << "THREAD '" << thread_id << "': MUSCOD is still there!" << std::endl;
      }
      muscod_ = muscod_muscod_;
    } else if (thread_id.compare("A") == 0) {
      std::cout << "THREAD '" << thread_id << "': " << (void*) muscod_A_ << std::endl;
      if (muscod_A_ == 0) {
          if (verbose) {
              std::cout << "THREAD '" << thread_id << "': created MUSCOD instance!" << std::endl;
          }
          muscod_A_ = new MUSCOD(false);
          muscod_A_->setModelPathAndName(nmpc.m_problem_path.c_str(), nmpc.m_model_name.c_str());
          muscod_A_->loadFromDatFile(NULL, NULL);
          muscod_A_->nmpcInitialize(0, NULL, NULL);
      } else {
          std::cout << "THREAD '" << thread_id << "': MUSCOD is still there!" << std::endl;
      }
      muscod_ = muscod_A_;
    } else if (thread_id.compare("B") == 0) {
      std::cout << "THREAD '" << thread_id << "': " << (void*) muscod_B_ << std::endl;
      if (muscod_B_ == 0) {
          if (verbose) {
              std::cout << "THREAD '" << thread_id << "': created MUSCOD instance!" << std::endl;
          }
          muscod_B_ = new MUSCOD(false);
          muscod_B_->setModelPathAndName(nmpc.m_problem_path.c_str(), nmpc.m_model_name.c_str());
          muscod_B_->loadFromDatFile(NULL, NULL);
          muscod_B_->nmpcInitialize(0, NULL, NULL);
      } else {
          std::cout << "THREAD '" << thread_id << "': MUSCOD is still there!" << std::endl;
      }
      muscod_ = muscod_B_;
    } else {
      std::cout << "THREAD '" << thread_id << "': you are fucked! " << std::endl;
      abort();
    }
    verbose = false;

    std::cout << "THREAD '" << thread_id << "': muscod_muscod_ = " << (void*) muscod_muscod_ << std::endl;
    std::cout << "THREAD '" << thread_id << "': muscod_A_      = " << (void*) muscod_A_ << std::endl;
    std::cout << "THREAD '" << thread_id << "': muscod_B_      = " << (void*) muscod_B_ << std::endl;
    std::cout << "THREAD '" << thread_id << "': muscod_        = " << (void*) muscod_ << std::endl;

    // define MCData pointer
    MCData = &(muscod_->data);
    // if (muscod_ == 0) {
    //     if (verbose) {
    //         std::cout << "THREAD '" << thread_id << "': created MUSCOD instance!" << std::endl;
    //     }
    //     // muscod_ = new MUSCOD();
    // } else {
    //     std::cout << "THREAD '" << thread_id << "': MUSCOD is still there!" << std::endl;
    //     abort();
    // }

    // if (nmpc.m_muscod == 0) {
    //     if (verbose) {
    //         std::cout << "THREAD '" << thread_id << "': created MUSCOD instance!" << std::endl;
    //     }
    //     muscod_ = new MUSCOD();
    // } else {
    //     std::cout << "THREAD '" << thread_id << "': MUSCOD is still there!" << std::endl;
    // }
    // NOTE check if address local data structure is equivalent with global one
    muscod_->print_MCData_address();
    muscod_->print_data_address();

    // forward verbosity from grl
    if (verbose) {
        muscod_->setLogLevelAndFile(-1, NULL, NULL);
    } else {
        muscod_->setLogLevelTotal(-1);
    }

    // assign MUSCOD instance to NMPC problem
    pthread_mutex_lock(nmpc.mutex_);
    nmpc.create_MUSCOD(muscod_);
    pthread_mutex_unlock(nmpc.mutex_);

    // instantiate values of NMPC structure
    pthread_mutex_lock(nmpc.mutex_);
    //unsigned long NMSN = nmpc.NMSN();
    // unsigned long NXD = nmpc.NXD();
    // unsigned long NP = nmpc.NP();
    // unsigned long NU = nmpc.NU();

    // define initial value and placeholder for feedback
    // NOTE copy values and dimension from NMPC problem
    Vector sd = nmpc.m_sd;
    Vector pf = nmpc.m_pf;
    Vector qc = nmpc.m_qc;

    // same for exchange TODO: move this to backup_muscod_state
    // nmpc.m_sd = Vector::Zero (NXD);
    // nmpc.m_pf = Vector::Zero (NP);
    // nmpc.m_qc = Vector::Zero (NU);
    pthread_mutex_unlock(nmpc.mutex_);

    // run single SQP iteration to be able to write a restart file
    pthread_mutex_lock(nmpc.mutex_);
    // TODO run initialize controller
    // initialize_controller (nmpc, 10, nmpc.m_sd, nmpc.m_pf);
    pthread_mutex_unlock(nmpc.mutex_);

    // release setup of thread
    pthread_mutex_lock(nmpc.mutex_);
    nmpc.m_is_initialized = true;
    pthread_cond_signal(nmpc.cond_iv_ready_);
    pthread_mutex_unlock(nmpc.mutex_);

    if (verbose) {
        std::cout << "THREAD '" << thread_id << "': MUSCOD thread is ready!" << std::endl;
    }
    // INITIALIZE CONTROLLER
    pthread_mutex_lock(nmpc.mutex_); // --> Lock the mutex
    // LOCK THE MUTEX
    // NOTE: wait for cond_iv_ready_ signal and then lock the mutex again
    nmpc.iv_ready_ = true;
    nmpc.in_preparation_ = false;
    pthread_cond_wait(nmpc.cond_iv_ready_, nmpc.mutex_);

    // TODO provide nnmpc_init by NMPCProblem
    if (!nmpc.iv_ready_) {
        // GET INITIAL VALUES AND PARAMETERS FROM DATA
        sd << nmpc.m_sd;
        pf << nmpc.m_pf;

        if (verbose) {
            std::cout << "THREAD '" << thread_id << "': sd: " << sd << std::endl;
            std::cout << "THREAD '" << thread_id << "': pf: " << pf << std::endl;
            std::cout << "THREAD '" << thread_id << "': initialize controller ... ";
        }

        // initialize controller
        initialize_controller (nmpc, 10, sd, pf);

        // if (verbose) {
            std::cout << "finished!" << std::endl;
        // }
    }

    // UNLOCK THE MUTEX
    pthread_mutex_unlock(nmpc.mutex_);

    // EXECUTION LOOP
    while (true) {
        pthread_mutex_lock(nmpc.mutex_); // --> Lock the mutex
        // LOCK THE MUTEX
        // NOTE: wait for cond_iv_ready_ signal and then lock the mutex again
        nmpc.iv_ready_ = true;
        nmpc.in_preparation_ = false;
        pthread_cond_wait(nmpc.cond_iv_ready_, nmpc.mutex_);

        // LEAVE THREAD IF QUIT FLAG IS SET IN DATA
        if (nmpc.m_quit) {
            pthread_mutex_unlock(nmpc.mutex_);
            break; // exit when thread is blocked
        }

        // GET INITIAL VALUES AND PARAMETERS FROM DATA
        sd << nmpc.m_sd;
        pf << nmpc.m_pf;
        if (verbose) {
            std::cout << "THREAD '" << thread_id << "': sd = ";
            std::cout << sd << std::endl;
            std::cout << "THREAD '" << thread_id << "': pf = ";
            std::cout << pf << std::endl;
        }

        // UNLOCK THE MUTEX
        pthread_mutex_unlock(nmpc.mutex_);

        // NMPC loop
        tic = stop_watch(); // <- timing execution
        ptac = 0.0;
        ttac = 0.0;
        pthread_mutex_lock(nmpc.mutex_); // --> Lock the mutex
        nmpc.timing._timing[6] = nmpc.m_nmpc_mode;
        pthread_mutex_unlock(nmpc.mutex_); // --> Unlock the mutex

        // 1) Feedback: Embed parameters and initial value from SIMULATION
        nmpc.feedback(sd, pf, &qc);

        tac = stop_watch(); // <- timing execution
        ttac += tac - tic;
        pthread_mutex_lock(nmpc.mutex_); // --> Lock the mutex
        nmpc.timing.time_feedback(tac - tic);
        pthread_mutex_unlock(nmpc.mutex_); // --> Unlock the mutex

        if (verbose) {
            std::cout << "THREAD '" << thread_id << "': qc = " << qc << std::endl;
        }

        pthread_mutex_lock(nmpc.mutex_); // --> Lock the mutex
        // LEAVE THREAD IF QUIT FLAG IS SET IN DATA
        if (nmpc.m_quit) {
            pthread_mutex_unlock(nmpc.mutex_);
            break; // exit when thread is blocked
        }

        // PUT CONTROLS INTO NMPC STRUCTURE
        // NOTE: Let main thread know that controls are ready
        //       and thread requires new state!
        nmpc.m_qc << qc;

        // signal to MUSCOD thread to embed initial values and start the computation
        nmpc.qc_ready_ = true;
        if (verbose) {
            std::cout << "THREAD '" << thread_id << "': Signaling 'QC ready'!" << std::endl;
        }

        // signal back that preparation phase has started
        nmpc.in_preparation_ = false;

        // UNLOCK THE MUTEX
        pthread_mutex_unlock(nmpc.mutex_);

        // reset timing statistics
        append_timing_to_file(nmpc.timing._timing, timing);
        nmpc.timing.reset ();         // reset timing statistics

        // when in NMPC mode then continue re-linearization
        // LMPC skips this code
        if (nmpc.m_nmpc_mode == 0) {
            if (verbose) {
                std::cout << "THREAD '" << thread_id << "': MODE 0" << std::endl;
            }
            // 2) Transition
            int shift = nmpc.get_shift_mode();

            // 3) Shifting
            tic = stop_watch(); // <- time execution
            nmpc.shifting(shift);
            tac = stop_watch();
            ptac += tac - tic;
            ttac += tac - tic;
            pthread_mutex_lock(nmpc.mutex_); // --> Lock the mutex
            nmpc.timing.time_shift(tac - tic);
            pthread_mutex_unlock(nmpc.mutex_); // --> Unlock the mutex

            // 4) Transition
            tic = stop_watch();
            nmpc.transition();
            tac = stop_watch();
            ptac += tac - tic;
            ttac += tac - tic;
            pthread_mutex_lock(nmpc.mutex_); // --> Lock the mutex
            nmpc.timing.time_transition(tac - tic);
            pthread_mutex_unlock(nmpc.mutex_); // --> Unlock the mutex

            // 4) Preparation
            tic = stop_watch();
            nmpc.preparation();
            tac = stop_watch();
            ptac += tac - tic;
            ttac += tac - tic;
            pthread_mutex_lock(nmpc.mutex_); // --> Lock the mutex
            nmpc.timing.time_prepare(tac - tic);
            pthread_mutex_unlock(nmpc.mutex_); // --> Unlock the mutex
        } // END IF NMPC MODE

/*
        // time total evaluation time of NMPC iteration
        pthread_mutex_lock(nmpc.mutex_); // --> Lock the mutex
        nmpc.timing.time_preparation(ptac);
        nmpc.timing.time_total(ttac);
        pthread_mutex_unlock(nmpc.mutex_); // --> Unlock the mutex
*/
    } // END WHILE LOOP

    if (verbose) {
        std::cout << "THREAD '" << thread_id << "': exiting MUSCOD thread" << std::endl;
    }
    pthread_mutex_lock(nmpc.mutex_);
    nmpc.delete_MUSCOD ();
    // delete muscod_;
    pthread_mutex_unlock(nmpc.mutex_);
    pthread_exit(NULL);
} // END OF muscod_run

void wait_for_iv_ready (NMPCProblem* nmpc, bool verbose)
{
    unsigned long cnt = 0;
    double total_time = 0.0;

    if (verbose) {
        std::cout << "MAIN: Waiting for thread to get ready!" << std::endl;
    }

    // sleep loop until thread is ready
    while (!nmpc->get_iv_ready()) {
        if (cnt >= 10000) {
            if (verbose) {
                std::cerr << "MAIN: thread frozen!" << std::endl;
            }
            break;
        }
        usleep(100); // wait for 0.1 ms
        cnt += 1;
    }

    // compute total_time
    total_time = (double) cnt * 0.1; // in ms

    if (verbose) {
        std::cout << "MAIN: Waiting for " << total_time;
        std::cout << "ms for preparation!" << std::endl;
    }

    if (nmpc->get_iv_ready() == false) {
        std::cerr << "MAIN: bailing out ..." << std::endl;
        abort();
    }
}

void provide_iv (
    NMPCProblem* nmpc,
    const Vector& initial_sd,
    const Vector& initial_pf,
    bool* iv_provided,
    bool wait,
    bool verbose
) {
    // want to provide initial values
    if ( !(*iv_provided) ) {
        std::cerr << "MAIN: in " << __func__ << " ..." << std::endl;
        std::cerr << "MAIN: iv_provided = " << *iv_provided << std::endl;
        std::cerr << "MAIN: skipped!" << std::endl;
        return;
    }

    // wait for thread to finish computations
    if (wait) {
        wait_for_iv_ready(nmpc, verbose);
    } // endif wait

    // PROVIDE INITIAL VALUES WHEN CONTROLLER IS READY
    if (nmpc->get_iv_ready() == true) {
          if (verbose) {
              std::cout << "MAIN: iv_ready: " << (bool) nmpc->get_iv_ready();
              std::cout << std::endl;
          }

          // LOCK THREAD
          pthread_mutex_lock(nmpc->mutex_);

          // Provide state and parameters
          nmpc->m_sd << initial_sd;
          nmpc->m_pf << initial_pf;

          // signal to MUSCOD thread to embed initial values and start the
          // computation
          nmpc->iv_ready_ = false;
          pthread_cond_signal(nmpc->cond_iv_ready_); // Seems like the problem is here!

          // UNLOCK THREAD
          pthread_mutex_unlock(nmpc->mutex_);

          // signal back that initial values was provided
          *iv_provided = true;
          return;
    }

    // signal back that initial values was *NOT* provided
    *iv_provided = false;
    return;
}

void wait_for_qc_ready (NMPCProblem* nmpc, bool verbose)
{
    unsigned long cnt = 0;
    double total_time = 0.0;

    if (verbose) {
        std::cout << "MAIN: Waiting for thread to finish calculations!" << std::endl;
    }

    while (!nmpc->get_qc_ready()) {
        if (cnt >= 10000) {
            if (verbose) {
                std::cerr << "MAIN: thread frozen!" << std::endl;
            }
            break;
        }
        usleep(100); // wait for 0.1 ms
        cnt += 1;
    }

    // compute total_time
    total_time = (double) cnt * 0.1; // in ms

    if (verbose) {
        std::cout << "MAIN: Waiting for " << total_time;
        std::cout << "ms for feedback!" << std::endl;
    }

    if (nmpc->get_qc_ready() == false) {
        std::cerr << "MAIN: bailing out ..." << std::endl;
        abort();
    }
}

void retrieve_qc (
    NMPCProblem* nmpc,
    Vector* first_qc,
    bool* qc_retrieved,
    bool wait,
    bool verbose
) {
    //double total_time = 0.0;
    //unsigned long cnt;
    if (!qc_retrieved) {
      abort();
    }

    // want to retrieve feedback?
    if ( !(*qc_retrieved) ) {
        std::cerr << "MAIN: in " << __func__ << " ..." << std::endl;
        std::cerr << "MAIN: qc_retrieved = " << *qc_retrieved << std::endl;
        std::cerr << "MAIN: skipped!" << std::endl;
        return;
    }

    // wait for thread to finish computations
    if (wait) {
        wait_for_qc_ready(nmpc, verbose);
    } // endif wait

    // RETRIEVE FEEDBACK CONTROLS WHEN COMPUTATIONS ARE FINISHED
    if (nmpc->get_qc_ready() == true) {
        if (verbose) {
            std::cout << "MAIN: qc_ready: " << (bool) nmpc->get_qc_ready() << std::endl;
        }
        // LOCK THREAD
        pthread_mutex_lock(nmpc->mutex_);

        // Retrieve feedback control from controller
        *first_qc << nmpc->m_qc;

        // revoke ready flag in thread
        nmpc->qc_ready_ = false;

        // UNLOCK THREAD
        pthread_mutex_unlock(nmpc->mutex_);

        if (verbose) {
            std::cout << "MAIN: qc = " << first_qc << std::endl;
        }

        // signal back that control was provided
        *qc_retrieved = true;
        return;
    }

    // signal back that control was *NOT* provided
    *qc_retrieved = false;
    return;
}

// handle communication with MUSCOD thread
// INPUT: iv_provided = false => skip providing iv
// INPUT: qc_retrieved = false => skip retrieving qc
// INPUT: wait = true => wait for MUSCOD-II thread to be finished
// return value is total time

void get_feedback (
    NMPCProblem* nmpc,
    const Vector& initial_sd,
    const Vector& initial_pf,
    Vector* first_qc,
    bool* iv_provided,
    bool* qc_retrieved,
    bool wait,
    bool verbose
) {
    // provide iv to thread, get time until thread was ready
    provide_iv (nmpc, initial_sd, initial_pf, iv_provided, wait, verbose);

    // provide iv to thread, get time until thread was ready
    retrieve_qc (nmpc, first_qc, qc_retrieved, wait, verbose);

    return;
}

}

