package threshold

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type CSO struct {
	Params ckks.Parameters
	Opts   CompareOptions
}

type Custodian struct {
	ID int
	SK *rlwe.SecretKey
}

type Reconstructor struct {
	ID int
	PK *rlwe.PublicKey
}

// MaskComparison simulates CSO constructing Enc(s) from Enc(x), Enc(y).
func (c CSO) MaskComparison(ctX, ctY *rlwe.Ciphertext) *rlwe.Ciphertext {
	ctZ := SubCiphertexts(c.Params, ctX, ctY)
	ctZ, err := ckks.NewEvaluator(c.Params, nil).SubNew(ctZ, c.Opts.StrictCompareBias)
	if err != nil {
		panic(err)
	}

	masks := GeneratePRFMask(c.Opts.MaskKey, c.Opts.MaskID, c.Opts.Length, c.Opts.MaskBaseBits, c.Opts.MaskRandomBits)
	ctMasked := MultiplyByMaskVector(c.Params, ctZ, masks)
	return AddFloodingNoise(c.Params, ctMasked, c.Opts.Length, c.Opts.FloodingSigma)
}

// BroadcastComparison simulates CSO broadcasting masked ciphertext to all custodians.
func (c CSO) BroadcastComparison(ctX, ctY *rlwe.Ciphertext) BroadcastMessage {
	return BroadcastMessage{
		CallID:     c.Opts.CallID,
		MaskID:     c.Opts.MaskID,
		Length:     c.Opts.Length,
		Ciphertext: c.MaskComparison(ctX, ctY),
	}
}

func NewCustodians(keys *ThresholdKeys) []Custodian {
	custodians := make([]Custodian, len(keys.PartySK))
	for i, sk := range keys.PartySK {
		custodians[i] = Custodian{ID: i, SK: sk}
	}
	return custodians
}

// MakeShare simulates D_h sending a partial decrypt share to D_k*.
func (d Custodian) MakeShare(params ckks.Parameters, ct *rlwe.Ciphertext, reconstructor int) PartialShare {
	return MakePartialShare(params, ct, d.SK, d.ID, reconstructor)
}

// HandleBroadcast simulates D_h receiving CSO's broadcast and sending one share upward.
func (d Custodian) HandleBroadcast(params ckks.Parameters, msg BroadcastMessage, publicSeed []byte, parties int) ShareMessage {
	to := SelectReconstructor(publicSeed, msg.CallID, parties)
	return ShareMessage{CallID: msg.CallID, Share: d.MakeShare(params, msg.Ciphertext, to)}
}

// Reconstruct simulates D_k* aggregating shares, extracting signs, and returning Enc(f).
func (r Reconstructor) Reconstruct(params ckks.Parameters, ct *rlwe.Ciphertext, shares []PartialShare, length int) (*rlwe.Ciphertext, []float64) {
	pt := ckks.NewPlaintext(params, ct.Level())
	*pt.MetaData = *ct.MetaData
	PolyToPlaintext(AggregateShares(params, ct, r.ID, shares), pt)

	values := make([]float64, length)
	if err := ckks.NewEncoder(params).Decode(pt, values); err != nil {
		panic(err)
	}

	f := ExtractPositiveIndicators(values, 0)
	return EncryptVector(params, r.PK, f), f
}

// HandleShares simulates D_k* receiving all share messages and returning Enc(f).
func (r Reconstructor) HandleShares(params ckks.Parameters, msg BroadcastMessage, shares []ShareMessage) ResultMessage {
	raw := make([]PartialShare, len(shares))
	for i, share := range shares {
		if share.CallID != msg.CallID {
			panic("share message call id mismatch")
		}
		raw[i] = share.Share
	}
	ctF, bits := r.Reconstruct(params, msg.Ciphertext, raw, msg.Length)
	return ResultMessage{CallID: msg.CallID, From: r.ID, Ciphertext: ctF, Bits: bits}
}
