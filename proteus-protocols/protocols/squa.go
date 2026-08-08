package protocols

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type SQUAResult struct {
	QTable [][]*rlwe.Ciphertext
	Stats  ProtocolStats
}

// SQUA applies Enc(Q_i) <- Enc(Q_i) + Enc(alpha)*Enc(delta)*Enc(u_i)*Enc(v_i).
func SQUA(ctx HomContext, qTable [][]*rlwe.Ciphertext, delta *rlwe.Ciphertext, state []*rlwe.Ciphertext, action []*rlwe.Ciphertext, alpha *rlwe.Ciphertext) [][]*rlwe.Ciphertext {
	return SQUAWithStats(ctx, qTable, delta, state, action, alpha).QTable
}

func SQUAWithStats(ctx HomContext, qTable [][]*rlwe.Ciphertext, delta *rlwe.Ciphertext, state []*rlwe.Ciphertext, action []*rlwe.Ciphertext, alpha *rlwe.Ciphertext) SQUAResult {
	start := time.Now()
	states, actions := validateSQUAInputs(ctx, qTable, delta, state, action, alpha)
	alphaDelta := mulRelinRescale(ctx, alpha, delta, "SQUA alpha*delta")

	updated := make([][]*rlwe.Ciphertext, states)
	for s := 0; s < states; s++ {
		updated[s] = make([]*rlwe.Ciphertext, actions)
		for a := 0; a < actions; a++ {
			cellMask := mulRelinRescale(ctx, state[s], action[a], fmt.Sprintf("SQUA mask state %d action %d", s, a))
			deltaQ := mulRelinRescale(ctx, alphaDelta, cellMask, fmt.Sprintf("SQUA delta state %d action %d", s, a))
			cell, err := ctx.Evaluator.AddNew(qTable[s][a], deltaQ)
			if err != nil {
				panic(fmt.Errorf("SQUA update state %d action %d: %w", s, a, err))
			}
			updated[s][a] = cell
		}
	}
	return SQUAResult{QTable: updated, Stats: finishStats(start)}
}

func mulRelinRescale(ctx HomContext, left, right *rlwe.Ciphertext, label string) *rlwe.Ciphertext {
	out, err := ctx.Evaluator.MulRelinNew(left, right)
	if err != nil {
		panic(fmt.Errorf("%s multiply: %w", label, err))
	}
	if err = ctx.Evaluator.Rescale(out, out); err != nil {
		panic(fmt.Errorf("%s rescale: %w", label, err))
	}
	return out
}

func validateSQUAInputs(ctx HomContext, qTable [][]*rlwe.Ciphertext, delta *rlwe.Ciphertext, state []*rlwe.Ciphertext, action []*rlwe.Ciphertext, alpha *rlwe.Ciphertext) (states int, actions int) {
	states, actions = validateSOHLInputs(ctx, state, qTable)
	if len(action) != actions {
		panic("action one-hot and Q table action dimension mismatch")
	}
	if alpha == nil || delta == nil {
		panic("alpha and delta ciphertexts must not be nil")
	}
	for _, bit := range action {
		if bit == nil {
			panic("nil action ciphertext")
		}
	}
	return states, actions
}
