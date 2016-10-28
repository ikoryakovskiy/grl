/** \file discretizer.h
 * \brief Generic discretizer definition.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2015-01-22
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

#ifndef GRL_DISCRETIZER_H_
#define GRL_DISCRETIZER_H_

#include <grl/configurable.h>

namespace grl
{

/// Provides a list of discrete points spanning a continuous space.
class Discretizer : public Configurable
{
  public:
    virtual ~Discretizer() { }
    virtual Discretizer* clone() = 0;
    
    /// List of discrete points.
    virtual void options(std::vector<Vector> *out) const
    {
      out->reserve(size());
      for (iterator it=begin(); it != end(); ++it)
        out->push_back(*it);
    }
    
    /// Iterates over discrete points.
    class iterator
    {
      protected:
        const Discretizer *discretizer_;
        IndexVector idx_;

      public:
        iterator(const Discretizer *discretizer=NULL, IndexVector idx=IndexVector()) : discretizer_(discretizer), idx_(idx) { }

        bool operator==(const iterator &rhs) const { return idx_ == rhs.idx_; }
        bool operator!=(const iterator &rhs) const { return idx_ != rhs.idx_; }
        iterator &operator++() { discretizer_->inc(&idx_); return *this; }
        Vector operator*() { return discretizer_->get(idx_); }
        const IndexVector &idx() { return idx_; }
    };
    
    virtual size_t size() const = 0;
    virtual iterator begin() const = 0;
    virtual iterator end() const
    {
      return iterator(this);
    }
    
    virtual void   inc(IndexVector *idx) const = 0;
    virtual Vector get(const IndexVector &idx) const = 0;

    virtual Vector steps()  const = 0;

    /// Finds the most closest vector to 'vec' in L1 sense and satisfies discretization steps
    /// As an optiona parmater (2) returns index of the discretized vector
    virtual void discretize(Vector &vec, IndexVector *idx_v = NULL) const = 0;

    /// Converts indexed vector to an linear offset of pointing to an indexed representation of the same input vector, and back
    virtual size_t convert(const IndexVector &idx_v) const = 0;
    virtual IndexVector convert(const size_t idx) const = 0;

};

}

#endif /* GRL_DISCRETIZER_H_ */
