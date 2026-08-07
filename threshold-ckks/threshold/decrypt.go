package threshold

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type PartialShare struct {
	From  int
	To    int
	Value ring.Poly
}

// MakePartialShare simulates custodian D_from sending mu_h to D_to.
func MakePartialShare(params ckks.Parameters, ct *rlwe.Ciphertext, sk *rlwe.SecretKey, from int, to int) PartialShare {
	return PartialShare{From: from, To: to, Value: PartialDecrypt(params, ct, sk)}
}

// PartialDecrypt computes one noisy share mu_h = c1*sk_h + e_h.
func PartialDecrypt(params ckks.Parameters, ct *rlwe.Ciphertext, sk *rlwe.SecretKey) ring.Poly {
	ringQ := params.RingQ().AtLevel(ct.Level())
	mu := ringQ.NewPoly()

	if ct.IsNTT {
		mu.Copy(ct.Value[1])
	} else {
		ringQ.NTTLazy(ct.Value[1], mu)
	}
	ringQ.MulCoeffsMontgomery(mu, sk.Value.Q, mu)

	e := SampleGaussian(ringQ, 3.2)
	ringQ.NTTLazy(e, e)
	ringQ.Add(mu, e, mu)
	ringQ.Reduce(mu, mu)
	return mu
}

// AggregateDecrypt reconstructs c0 + sum(mu_h).
func AggregateDecrypt(params ckks.Parameters, ct *rlwe.Ciphertext, mus []ring.Poly) ring.Poly {
	ringQ := params.RingQ().AtLevel(ct.Level())
	s := ringQ.NewPoly()

	if ct.IsNTT {
		s.Copy(ct.Value[0])
	} else {
		ringQ.NTTLazy(ct.Value[0], s)
	}
	for _, mu := range mus {
		ringQ.Add(s, mu, s)
	}
	ringQ.Reduce(s, s)
	if !ct.IsNTT {
		ringQ.INTT(s, s)
	}
	return s
}

// AggregateShares simulates the elected reconstructor collecting and aggregating shares.
func AggregateShares(params ckks.Parameters, ct *rlwe.Ciphertext, reconstructor int, shares []PartialShare) ring.Poly {
	if len(shares) == 0 {
		panic("no partial shares to aggregate")
	}

	seen := make(map[int]bool, len(shares))
	mus := make([]ring.Poly, len(shares))
	for i, share := range shares {
		if share.To != reconstructor {
			panic("partial share delivered to wrong reconstructor")
		}
		if seen[share.From] {
			panic("duplicate partial share sender")
		}
		seen[share.From], mus[i] = true, share.Value
	}
	return AggregateDecrypt(params, ct, mus)
}
