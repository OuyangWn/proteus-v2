package threshold

import (
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

func SampleGaussian(
	ringQ *ring.Ring,
	sigma float64,
) ring.Poly {

	prng, _ := sampling.NewPRNG()

	dist := ring.DiscreteGaussian{
		Sigma: sigma,
		Bound: 6 * sigma,
	}

	sampler, _ := ring.NewSampler(
		prng,
		ringQ,
		dist,
		false,
	)

	e := ringQ.NewPoly()

	sampler.Read(e)

	return e
}
