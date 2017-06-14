/** \file nmpc_base.h
 * \brief Base class for NMPC.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2016-09-15
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

#ifndef GRL_NMPC_BASE_POLICY_H_
#define GRL_NMPC_BASE_POLICY_H_

#include <grl/policy.h>
#include <grl/policies/muscod_nmpc.h>

class MUSCOD;

namespace grl
{

/// NMPCBase policy
class NMPCBase : public Policy
{
  public:
    NMPCBase() : initFeedback_(0), verbose_(0) {}

  //-------------------------------------- GRL -----------------------------------//
  protected:
    int initFeedback_;
    int verbose_;
    std::string model_name_, lua_model_, nmpc_model_name_, model_path_;
    Vector action_min_, action_max_;

    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

  protected:
    void *setup_model_path(const std::string path, const std::string model, const std::string lua_model);
};

//-------------------------------------- MUSCOD -----------------------------------//
double stop_watch();

// clean-up MUSCOD-II main thread
void stop_thread (NMPCProblem& data,
    pthread_t* muscod_thread,
    bool verbose = false
);

// run MUSCOD-II NMPC for several iterations to initialize controller for
// current initial sd/pf, first qc is optional
void initialize_controller (
    MUSCODProblem& nmpc, const int nmpc_ninit,
    Vector& sd, Vector& pf,
    Vector* qc = NULL
);

// Initialize mutex and condition variable objects and the controller thread
void initialize_thread(pthread_t& muscod_thread,
    void* (*function) (void*) ,
    NMPCProblem*& data,
    std::string problem_path,
    std::string nmpc_model_name,
    const std::string thread_id,
    pthread_cond_t& cond_iv_ready,
    pthread_mutex_t& mutex,
    bool grl_verbose,
    bool verbose
);

// MUSCOD-II main thread setup and execution loop
void *muscod_run (void *indata);

void wait_for_iv_ready (NMPCProblem* nmpc, bool verbose = false);

void provide_iv (
    NMPCProblem* nmpc,
    const Vector& initial_sd,
    const Vector& initial_pf,
    bool* iv_provided,
    bool wait = true,
    bool verbose = false
);

void wait_for_qc_ready (NMPCProblem* nmpc, bool verbose = false);

void retrieve_qc (
    NMPCProblem* nmpc,
    Vector* first_qc,
    bool* qc_retrieved,
    bool wait = true,
    bool verbose = false
);

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
    bool wait = true,
    bool verbose = false
);

}

#endif /* GRL_NMPC_BASE_POLICY_H_ */
