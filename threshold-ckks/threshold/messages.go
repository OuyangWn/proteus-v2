package threshold

import "github.com/tuneinsight/lattigo/v6/core/rlwe"

type BroadcastMessage struct {
	CallID     string
	MaskID     string
	Length     int
	Ciphertext *rlwe.Ciphertext
}

type ShareMessage struct {
	CallID string
	Share  PartialShare
}

type ResultMessage struct {
	CallID     string
	From       int
	Ciphertext *rlwe.Ciphertext
	Bits       []float64
}

type ProtocolTranscript struct {
	Broadcast BroadcastMessage
	Shares    []ShareMessage
	Result    ResultMessage
}
