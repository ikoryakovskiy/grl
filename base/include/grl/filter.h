/*
 * Filter.h
 *
 *  Created on: Nov 24, 2008
 *      Author: Erik Schuitema
 */

#ifndef FILTER_H_
#define FILTER_H_

#include <string.h>

class CFilterBase
{
  public:
    virtual ~CFilterBase() {}
    virtual void init(double samplingFrequency, double cutoffFrequency) = 0;
    virtual double filter(double newSample) = 0;
    virtual void clear() = 0;
};

template<class FILTERTYPE, int FILTERLENGTH>
class CFilter : public CFilterBase
{
	protected:
		FILTERTYPE			mSampleBuffer[FILTERLENGTH];
		void				addSample(FILTERTYPE newSample)
		{
			for (int i = FILTERLENGTH-1; i>0; i--)	// Now let's hope that the compiler unrolls this loop
				mSampleBuffer[i] = mSampleBuffer[i-1];
			mSampleBuffer[0] = newSample;
		}
	public:
			CFilter() 		{clear();}
		virtual	~CFilter()		{}
		int	getFilterLength()	{return FILTERLENGTH;}
		virtual FILTERTYPE filter(FILTERTYPE newSample)	{return 0;}
		virtual void clear()		{memset(mSampleBuffer, 0, FILTERLENGTH*sizeof(FILTERTYPE));}
};

#endif /* FILTER_H_ */
