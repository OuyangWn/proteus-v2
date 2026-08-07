package threshold

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
)

func PolyToPlaintext(
	poly ring.Poly,
	template *rlwe.Plaintext,
) *rlwe.Plaintext {

	pt := template.CopyNew()

	pt.Value = poly

	return pt
}
