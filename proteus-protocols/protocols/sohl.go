package protocols

import (
	"fmt"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type SOHLResult struct {
	Row   []*rlwe.Ciphertext
	Stats ProtocolStats
}

// SOHL returns the encrypted Q row selected by encrypted one-hot state bits.
func SOHL(ctx HomContext, state []*rlwe.Ciphertext, qTable [][]*rlwe.Ciphertext) []*rlwe.Ciphertext {
	return SOHLWithStats(ctx, state, qTable).Row
}

func SOHLWithStats(ctx HomContext, state []*rlwe.Ciphertext, qTable [][]*rlwe.Ciphertext) SOHLResult {
	start := time.Now()
	states, actions := validateSOHLInputs(ctx, state, qTable)
	row := make([]*rlwe.Ciphertext, actions)
	for a := 0; a < actions; a++ {
		var acc *rlwe.Ciphertext
		for s := 0; s < states; s++ {
			term, err := ctx.Evaluator.MulRelinNew(state[s], qTable[s][a])
			if err != nil {
				panic(fmt.Errorf("SOHL multiply state %d action %d: %w", s, a, err))
			}
			if err = ctx.Evaluator.Rescale(term, term); err != nil {
				panic(fmt.Errorf("SOHL rescale state %d action %d: %w", s, a, err))
			}
			if acc == nil {
				acc = term
				continue
			}
			if err = ctx.Evaluator.Add(acc, term, acc); err != nil {
				panic(fmt.Errorf("SOHL accumulate state %d action %d: %w", s, a, err))
			}
		}
		row[a] = acc
	}
	return SOHLResult{Row: row, Stats: finishStats(start)}
}

// SAQSA is the pseudocode wrapper: fetch the full encrypted action row.
func SAQSA(ctx HomContext, state []*rlwe.Ciphertext, qTable [][]*rlwe.Ciphertext) []*rlwe.Ciphertext {
	return SAQSAWithStats(ctx, state, qTable).Row
}

func SAQSAWithStats(ctx HomContext, state []*rlwe.Ciphertext, qTable [][]*rlwe.Ciphertext) SOHLResult {
	start := time.Now()
	result := SOHLWithStats(ctx, state, qTable)
	return SOHLResult{Row: result.Row, Stats: finishStats(start, result.Stats)}
}

func validateSOHLInputs(ctx HomContext, state []*rlwe.Ciphertext, qTable [][]*rlwe.Ciphertext) (states int, actions int) {
	if ctx.Evaluator == nil {
		panic("homomorphic evaluator must not be nil")
	}
	if len(state) == 0 || len(qTable) != len(state) {
		panic("state vector and Q table state dimension mismatch")
	}
	actions = len(qTable[0])
	if actions == 0 {
		panic("Q table must contain at least one action")
	}
	for s := range state {
		if state[s] == nil || len(qTable[s]) != actions {
			panic("nil state ciphertext or ragged Q table")
		}
		for a := range qTable[s] {
			if qTable[s][a] == nil {
				panic("nil Q-table ciphertext")
			}
		}
	}
	return len(state), actions
}
