package protocols

import (
	"fmt"
	"time"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type SMAAOptions struct {
	Length       int
	MaskKey      []byte
	PublicSeed   []byte
	CallIDPrefix string
	MaskIDPrefix string
}

type SMAAResult struct {
	Max          *rlwe.Ciphertext
	Argmax       []*rlwe.Ciphertext
	Index        *rlwe.Ciphertext
	Comparisons  int
	CompareStats []threshold.ProtocolStats
	Stats        ProtocolStats
}

type smaaCandidate struct {
	Value  *rlwe.Ciphertext
	Argmax []*rlwe.Ciphertext
	Index  *rlwe.Ciphertext
}

func DefaultSMAAOptions(length int, maskKey []byte, publicSeed []byte, callIDPrefix string) SMAAOptions {
	return SMAAOptions{
		Length:       length,
		MaskKey:      maskKey,
		PublicSeed:   publicSeed,
		CallIDPrefix: callIDPrefix,
		MaskIDPrefix: callIDPrefix + ":mask",
	}
}

// SMAA returns the encrypted max value plus an encrypted argmax one-hot vector.
func SMAA(ctx HomContext, keys *threshold.ThresholdKeys, values []*rlwe.Ciphertext, opts SMAAOptions) SMAAResult {
	start := time.Now()
	validateSMAAInputs(ctx, keys, values, opts)

	candidates := initialSMAACandidates(ctx, keys.PK, values, opts.Length)
	compareStats := make([]threshold.ProtocolStats, 0, len(values)-1)
	childStats := make([]ProtocolStats, 0, len(values)-1)
	comparison := 0
	for len(candidates) > 1 {
		next := make([]smaaCandidate, 0, (len(candidates)+1)/2)
		for i := 0; i+1 < len(candidates); i += 2 {
			comparison++
			left, right := candidates[i], candidates[i+1]
			cmpOpts := threshold.DefaultSPDCmpOptions(
				opts.Length,
				opts.MaskKey,
				fmt.Sprintf("%s:%d", opts.MaskIDPrefix, comparison),
				opts.PublicSeed,
				fmt.Sprintf("%s:cmp:%d", opts.CallIDPrefix, comparison),
			)
			cmp := threshold.SPDCmpWithTranscript(ctx.Params, keys, left.Value, right.Value, cmpOpts)
			next = append(next, selectSMAACandidate(ctx, keys.PK, cmp.Ciphertext, left, right, opts.Length))
			compareStats = append(compareStats, cmp.Stats)
			childStats = append(childStats, spdCmpStats(cmp.Stats))
		}
		if len(candidates)%2 == 1 {
			next = append(next, candidates[len(candidates)-1])
		}
		candidates = next
	}

	winner := candidates[0]
	return SMAAResult{Max: winner.Value, Argmax: winner.Argmax, Index: winner.Index, Comparisons: comparison, CompareStats: compareStats, Stats: finishStats(start, childStats...)}
}

func selectSMAACandidate(ctx HomContext, pk *rlwe.PublicKey, bit *rlwe.Ciphertext, left smaaCandidate, right smaaCandidate, length int) smaaCandidate {
	argmax := make([]*rlwe.Ciphertext, len(left.Argmax))
	for i := range argmax {
		argmax[i] = SelectByEncryptedBit(ctx, pk, bit, left.Argmax[i], right.Argmax[i], length)
	}
	return smaaCandidate{
		Value:  SelectByEncryptedBit(ctx, pk, bit, left.Value, right.Value, length),
		Argmax: argmax,
		Index:  SelectByEncryptedBit(ctx, pk, bit, left.Index, right.Index, length),
	}
}

func initialSMAACandidates(ctx HomContext, pk *rlwe.PublicKey, values []*rlwe.Ciphertext, length int) []smaaCandidate {
	candidates := make([]smaaCandidate, len(values))
	for action := range values {
		argmax := make([]*rlwe.Ciphertext, len(values))
		for bit := range argmax {
			value := 0.0
			if bit == action {
				value = 1
			}
			argmax[bit] = encryptRepeated(ctx, pk, value, length, values[action])
		}
		candidates[action] = smaaCandidate{
			Value:  values[action],
			Argmax: argmax,
			Index:  encryptRepeated(ctx, pk, float64(action+1), length, values[action]),
		}
	}
	return candidates
}

func validateSMAAInputs(ctx HomContext, keys *threshold.ThresholdKeys, values []*rlwe.Ciphertext, opts SMAAOptions) {
	if ctx.Evaluator == nil {
		panic("homomorphic evaluator must not be nil")
	}
	if keys == nil || keys.PK == nil || len(keys.PartySK) == 0 {
		panic("threshold keys must include public key and party shares")
	}
	if len(values) == 0 {
		panic("SMAA requires at least one value")
	}
	for _, value := range values {
		if value == nil {
			panic("nil SMAA value ciphertext")
		}
	}
	if opts.Length <= 0 || opts.Length > ctx.Params.MaxSlots() {
		panic("invalid SMAA length")
	}
	if len(opts.MaskKey) == 0 || len(opts.PublicSeed) == 0 || opts.CallIDPrefix == "" || opts.MaskIDPrefix == "" {
		panic("SMAA mask key, public seed, and id prefixes must not be empty")
	}
}
