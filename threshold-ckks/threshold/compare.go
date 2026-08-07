package threshold

import (
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const demoMaskBaseBits uint = 10
const demoMaskRandomBits uint = 4
const demoFloodingSigma = 4.0
const strictIntegerComparisonBias = 0.5

// CompareOptions configures one SPDCmp call; demo defaults use smaller masks than paper parameters.
type CompareOptions struct {
	Length            int
	MaskKey           []byte
	MaskID            string
	PublicSeed        []byte
	CallID            string
	MaskBaseBits      uint
	MaskRandomBits    uint
	FloodingSigma     float64
	StrictCompareBias float64
}

type SPDCmpOptions = CompareOptions

type CompareResult struct {
	Ciphertext    *rlwe.Ciphertext
	Bits          []float64
	Reconstructor int
	ShareCount    int
	Transcript    ProtocolTranscript
	Stats         ProtocolStats
}

func DefaultSPDCmpOptions(length int, maskKey []byte, maskID string, publicSeed []byte, callID string) SPDCmpOptions {
	return DefaultCompareOptions(length, maskKey, maskID, publicSeed, callID)
}

func DefaultCompareOptions(length int, maskKey []byte, maskID string, publicSeed []byte, callID string) CompareOptions {
	return CompareOptions{
		Length:            length,
		MaskKey:           maskKey,
		MaskID:            maskID,
		PublicSeed:        publicSeed,
		CallID:            callID,
		MaskBaseBits:      demoMaskBaseBits,
		MaskRandomBits:    demoMaskRandomBits,
		FloodingSigma:     demoFloodingSigma,
		StrictCompareBias: strictIntegerComparisonBias,
	}
}

// SPDCmp is the upper-layer API: input Enc(x), Enc(y), output Enc(f), f_i=1[x_i>y_i].
func SPDCmp(params ckks.Parameters, keys *ThresholdKeys, ctX, ctY *rlwe.Ciphertext, opts SPDCmpOptions) *rlwe.Ciphertext {
	return SPDCmpWithTranscript(params, keys, ctX, ctY, opts).Ciphertext
}

// SPDCmpWithTranscript runs the same protocol and exposes single-process debug data.
func SPDCmpWithTranscript(params ckks.Parameters, keys *ThresholdKeys, ctX, ctY *rlwe.Ciphertext, opts SPDCmpOptions) CompareResult {
	validateCompareInputs(params, keys, ctX, ctY, opts)
	validateCompareOptions(opts)

	start := time.Now()
	cso := CSO{Params: params, Opts: opts}

	stage := time.Now()
	broadcast := cso.BroadcastComparison(ctX, ctY)
	csoTime := time.Since(stage)

	reconstructor := SelectReconstructor(opts.PublicSeed, opts.CallID, len(keys.PartySK))

	custodians := NewCustodians(keys)
	shares := make([]ShareMessage, len(keys.PartySK))
	stage = time.Now()
	for i, custodian := range custodians {
		shares[i] = custodian.HandleBroadcast(params, broadcast, opts.PublicSeed, len(custodians))
	}
	shareTime := time.Since(stage)

	recon := Reconstructor{ID: reconstructor, PK: keys.PK}
	stage = time.Now()
	result := recon.HandleShares(params, broadcast, shares)
	reconTime := time.Since(stage)

	transcript := ProtocolTranscript{Broadcast: broadcast, Shares: shares, Result: result}
	stats := ProtocolStats{
		Durations:     ProtocolDurations{CSOBroadcast: csoTime, ShareGeneration: shareTime, Reconstruction: reconTime, Total: time.Since(start)},
		Communication: EstimateCommunication(transcript, len(custodians)),
	}
	return CompareResult{Ciphertext: result.Ciphertext, Bits: result.Bits, Reconstructor: result.From, ShareCount: len(shares), Transcript: transcript, Stats: stats}
}

// CompareEncryptedVectors keeps the previous name available while callers migrate to SPDCmp.
func CompareEncryptedVectors(params ckks.Parameters, keys *ThresholdKeys, ctX, ctY *rlwe.Ciphertext, opts CompareOptions) CompareResult {
	return SPDCmpWithTranscript(params, keys, ctX, ctY, opts)
}

func validateCompareInputs(params ckks.Parameters, keys *ThresholdKeys, ctX, ctY *rlwe.Ciphertext, opts CompareOptions) {
	if keys == nil || keys.PK == nil || len(keys.PartySK) == 0 {
		panic("threshold keys must include public key and at least one party secret share")
	}
	if ctX == nil || ctY == nil {
		panic("input ciphertexts must not be nil")
	}
	if opts.Length > params.MaxSlots() {
		panic("length exceeds CKKS max slots")
	}
}

func validateCompareOptions(opts CompareOptions) {
	if opts.Length <= 0 {
		panic("length must be positive")
	}
	if len(opts.MaskKey) == 0 {
		panic("mask key must not be empty")
	}
	if len(opts.PublicSeed) == 0 {
		panic("public seed must not be empty")
	}
	if opts.MaskID == "" || opts.CallID == "" {
		panic("mask id and call id must not be empty")
	}
}
