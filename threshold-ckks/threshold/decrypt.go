package threshold

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func PartialDecrypt(
	params ckks.Parameters,
	ct *rlwe.Ciphertext,
	sk *rlwe.SecretKey,
) ring.Poly {

	ringQ := params.RingQ()

	mu := ringQ.NewPoly()

	// μ_h = c1 * sk_h
	ringQ.MulCoeffsMontgomery(
		ct.Value[1],
		sk.Value.Q,
		mu,
	)

	return mu
}

func AggregateDecrypt(
	params ckks.Parameters,
	ct *rlwe.Ciphertext,
	mus []ring.Poly,
) ring.Poly {

	ringQ := params.RingQ()

	s := ringQ.NewPoly()

	// s = c0
	s.Copy(ct.Value[0])

	// s += Σ μ_h
	for _, mu := range mus {
		ringQ.Add(
			s,
			mu,
			s,
		)
	}

	return s
}
