/** \file leo_online_learning.h
 * \brief Leo online learning experiment header file.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2017-07-31
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

#ifndef GRL_LEO_ONLINE_LEARNING_EXPERIMENT_H_
#define GRL_LEO_ONLINE_LEARNING_EXPERIMENT_H_

#include <grl/experiments/online_learning.h>

namespace grl
{

/// Standard Agent-Environment interaction experiment.
class LeoSquattingOnlineLearningExperiment : public OnlineLearningExperiment
{
  public:
    TYPEINFO("experiment/leo/online_learning/squatting", "Interactive learning experiment of squatting for LEO")

  public:
    LeoSquattingOnlineLearningExperiment()  { }

    // From Experiment
    virtual void run();
};

}

#endif /* GRL_LEO_ONLINE_LEARNING_EXPERIMENT_H_ */
