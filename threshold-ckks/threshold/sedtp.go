package threshold

import (
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const demoSEDTPMaskBound = 1024.0

type SEDTPOptions struct {
	Length     int
	PublicSeed []byte
	CallID     string
	MaskBound  float64
}

type SEDTPResult struct {
	Ciphertext    *rlwe.Ciphertext
	Reconstructor int
	ShareCount    int
	Transcript    ProtocolTranscript
	Stats         ProtocolStats
}

func DefaultSEDTPOptions(length int, publicSeed []byte, callID string) SEDTPOptions {
	return SEDTPOptions{Length: length, PublicSeed: publicSeed, CallID: callID, MaskBound: demoSEDTPMaskBound}
}

// SEDTP transforms Enc_pkA(x) into Enc_pkB(x) through masked threshold reconstruction.
func SEDTP(params ckks.Parameters, sourceKeys *ThresholdKeys, targetPK *rlwe.PublicKey, ctX *rlwe.Ciphertext, opts SEDTPOptions) *rlwe.Ciphertext {
	return SEDTPWithTranscript(params, sourceKeys, targetPK, ctX, opts).Ciphertext
}

// SEDTPWithTranscript exposes the single-process protocol trace for measurement/debugging.
func SEDTPWithTranscript(params ckks.Parameters, sourceKeys *ThresholdKeys, targetPK *rlwe.PublicKey, ctX *rlwe.Ciphertext, opts SEDTPOptions) SEDTPResult {
	validateSEDTPInputs(params, sourceKeys, targetPK, ctX, opts)

	start := time.Now()
	stage := time.Now()
	mask := sampleAdditiveMask(opts.Length, opts.MaskBound)
	ctMaskA := EncryptVectorAtLevelScale(params, sourceKeys.PK, mask, ctX.Level(), ctX.Scale)
	ctMaskedA := AddCiphertexts(params, ctX, ctMaskA)
	broadcast := BroadcastMessage{CallID: opts.CallID, MaskID: "sedtp-additive-mask", Length: opts.Length, Ciphertext: ctMaskedA}
	csoTime := time.Since(stage)

	reconstructor := SelectReconstructor(opts.PublicSeed, opts.CallID, len(sourceKeys.PartySK))
	custodians := NewCustodians(sourceKeys)
	shares := make([]ShareMessage, len(custodians))

	stage = time.Now()
	for i, custodian := range custodians {
		shares[i] = custodian.HandleBroadcast(params, broadcast, opts.PublicSeed, len(custodians))
	}
	shareTime := time.Since(stage)

	stage = time.Now()
	ctMaskedB := reencryptMaskedPlaintext(params, broadcast, shares, reconstructor, targetPK, opts.Length)
	reconTime := time.Since(stage)

	resultMsg := ResultMessage{CallID: opts.CallID, From: reconstructor, Ciphertext: ctMaskedB}
	transcript := ProtocolTranscript{Broadcast: broadcast, Shares: shares, Result: resultMsg}

	stage = time.Now()
	ctMaskB := EncryptVectorAtLevelScale(params, targetPK, mask, ctMaskedB.Level(), ctMaskedB.Scale)
	ctOut := SubCiphertexts(params, ctMaskedB, ctMaskB)
	finalizeTime := time.Since(stage)
	stats := ProtocolStats{
		Durations:     ProtocolDurations{CSOBroadcast: csoTime, ShareGeneration: shareTime, Reconstruction: reconTime, CSOFinalize: finalizeTime, Total: time.Since(start)},
		Communication: EstimateCommunication(transcript, len(custodians)),
	}
	return SEDTPResult{Ciphertext: ctOut, Reconstructor: reconstructor, ShareCount: len(shares), Transcript: transcript, Stats: stats}
}

func reencryptMaskedPlaintext(params ckks.Parameters, msg BroadcastMessage, shares []ShareMessage, reconstructor int, targetPK *rlwe.PublicKey, length int) *rlwe.Ciphertext {
	raw := make([]PartialShare, len(shares))
	for i, share := range shares {
		if share.CallID != msg.CallID {
			panic("share message call id mismatch")
		}
		raw[i] = share.Share
	}

	pt := ckks.NewPlaintext(params, msg.Ciphertext.Level())
	*pt.MetaData = *msg.Ciphertext.MetaData
	PolyToPlaintext(AggregateShares(params, msg.Ciphertext, reconstructor, raw), pt)

	values := make([]float64, length)
	if err := ckks.NewEncoder(params).Decode(pt, values); err != nil {
		panic(err)
	}
	return EncryptVector(params, targetPK, values)
}

func sampleAdditiveMask(length int, bound float64) []float64 {
	mask := make([]float64, length)
	for i := range mask {
		mask[i] = (2*sampleUnitFloat64() - 1) * bound
	}
	return mask
}

func validateSEDTPInputs(params ckks.Parameters, sourceKeys *ThresholdKeys, targetPK *rlwe.PublicKey, ctX *rlwe.Ciphertext, opts SEDTPOptions) {
	if sourceKeys == nil || sourceKeys.PK == nil || len(sourceKeys.PartySK) == 0 {
		panic("source threshold keys must include public key and at least one party secret share")
	}
	if targetPK == nil {
		panic("target public key must not be nil")
	}
	if ctX == nil {
		panic("input ciphertext must not be nil")
	}
	if opts.Length <= 0 || opts.Length > params.MaxSlots() {
		panic("invalid SEDTP length")
	}
	if len(opts.PublicSeed) == 0 || opts.CallID == "" {
		panic("public seed and call id must not be empty")
	}
	if opts.MaskBound <= 0 {
		panic("mask bound must be positive")
	}
}
