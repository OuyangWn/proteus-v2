package protocols

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type HomContext struct {
	Params    ckks.Parameters
	Evaluator *ckks.Evaluator
}

// NewHomContext carries the public evaluation keys needed by upper-level protocols.
func NewHomContext(params ckks.Parameters, evk rlwe.EvaluationKeySet) HomContext {
	return HomContext{Params: params, Evaluator: ckks.NewEvaluator(params, evk)}
}
