package threshold

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

// PolyToPlaintext swaps a reconstructed polynomial into an existing plaintext template.
func PolyToPlaintext(poly ring.Poly, template *rlwe.Plaintext) *rlwe.Plaintext {
	template.Value = poly
	return template
}
