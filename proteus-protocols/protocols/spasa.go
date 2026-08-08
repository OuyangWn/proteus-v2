package protocols

import (
	"time"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type SPASAOptions struct {
	SMAA SMAAOptions
}

type SPASAResult struct {
	Action []*rlwe.Ciphertext
	Row    []*rlwe.Ciphertext
	SMAA   SMAAResult
	Stats  ProtocolStats
}

func DefaultSPASAOptions(length int, maskKey []byte, publicSeed []byte, callIDPrefix string) SPASAOptions {
	return SPASAOptions{SMAA: DefaultSMAAOptions(length, maskKey, publicSeed, callIDPrefix+":smaa")}
}

// SPASA mixes encrypted argmax and encrypted random one-hot actions by encrypted exploit bit.
func SPASA(ctx HomContext, keys *threshold.ThresholdKeys, state []*rlwe.Ciphertext, qTable [][]*rlwe.Ciphertext, exploitBit *rlwe.Ciphertext, randomAction []*rlwe.Ciphertext, opts SPASAOptions) SPASAResult {
	start := time.Now()
	validateSPASAInputs(ctx, keys, exploitBit, randomAction, opts)

	row := SAQSAWithStats(ctx, state, qTable)
	smaa := SMAA(ctx, keys, row.Row, opts.SMAA)
	if len(randomAction) != len(smaa.Argmax) {
		panic("random action and argmax dimensions mismatch")
	}

	action := make([]*rlwe.Ciphertext, len(smaa.Argmax))
	for i := range action {
		action[i] = SelectByEncryptedBit(ctx, keys.PK, exploitBit, smaa.Argmax[i], randomAction[i], opts.SMAA.Length)
	}
	return SPASAResult{Action: action, Row: row.Row, SMAA: smaa, Stats: finishStats(start, row.Stats, smaa.Stats)}
}

func validateSPASAInputs(ctx HomContext, keys *threshold.ThresholdKeys, exploitBit *rlwe.Ciphertext, randomAction []*rlwe.Ciphertext, opts SPASAOptions) {
	if ctx.Evaluator == nil {
		panic("homomorphic evaluator must not be nil")
	}
	if keys == nil || keys.PK == nil {
		panic("threshold keys must include public key")
	}
	if exploitBit == nil || len(randomAction) == 0 {
		panic("exploit bit and random action must not be empty")
	}
	for _, action := range randomAction {
		if action == nil {
			panic("nil random action ciphertext")
		}
	}
	if opts.SMAA.Length <= 0 || opts.SMAA.Length > ctx.Params.MaxSlots() {
		panic("invalid SPASA length")
	}
}
