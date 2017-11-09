/** \file leo_curriculum_learning.h
 * \brief Leo curriculum learning experiment header file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-11-09
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

#ifndef GRL_LEO_CURRICULUM_LEARNING_EXPERIMENT_H_
#define GRL_LEO_CURRICULUM_LEARNING_EXPERIMENT_H_

#include <grl/experiments/online_learning.h>
#include <grl/vector.h>

namespace grl
{

/// Standard Agent-Environment interaction experiment.
class LeoCurriculumLearningExperiment : public OnlineLearningExperiment
{
  public:
    TYPEINFO("experiment/leo/curriculum_learning", "Interactive curriculum learning experiment for Leo")

  protected:
    Vector rwForward_;
    int ssdiv_, ssdiv_stepup_, ttdiv_;

  public:
    LeoCurriculumLearningExperiment() : ssdiv_(0), ssdiv_stepup_(0), ttdiv_(0) { }

    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);

    // From Experiment
    virtual void run();

  protected:
    virtual void reconfigureLeo(double frac);
};

}

#endif /* GRL_LEO_CURRICULUM_LEARNING_EXPERIMENT_H_ */
