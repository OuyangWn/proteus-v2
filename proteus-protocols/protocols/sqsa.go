package protocols

import (
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type SQSAResult struct {
	Value *rlwe.Ciphertext
	Stats ProtocolStats
}

// SQSA returns the encrypted Q value selected by encrypted state and action one-hots.
func SQSA(ctx HomContext, state []*rlwe.Ciphertext, action []*rlwe.Ciphertext, qTable [][]*rlwe.Ciphertext) *rlwe.Ciphertext {
	return SQSAWithStats(ctx, state, action, qTable).Value
}

func SQSAWithStats(ctx HomContext, state []*rlwe.Ciphertext, action []*rlwe.Ciphertext, qTable [][]*rlwe.Ciphertext) SQSAResult {
	start := time.Now()
	row := SOHLWithStats(ctx, state, qTable)
	value := SelectByEncryptedOneHot(ctx, action, row.Row)
	return SQSAResult{Value: value, Stats: finishStats(start, row.Stats)}
}
