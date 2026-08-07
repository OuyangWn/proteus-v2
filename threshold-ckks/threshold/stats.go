package threshold

import (
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type ProtocolDurations struct {
	CSOBroadcast    time.Duration
	ShareGeneration time.Duration
	Reconstruction  time.Duration
	Total           time.Duration
}

type CommunicationStats struct {
	BroadcastBytes          int
	BroadcastDeliveredBytes int
	ShareBytes              int
	ResultBytes             int
	TotalBytes              int
}

type ProtocolStats struct {
	Durations     ProtocolDurations
	Communication CommunicationStats
}

// EstimateCommunication counts serialized cryptographic payload bytes in the simulated network.
func EstimateCommunication(transcript ProtocolTranscript, parties int) CommunicationStats {
	broadcast := ciphertextBytes(transcript.Broadcast.Ciphertext)
	shareBytes := 0
	for _, share := range transcript.Shares {
		shareBytes += share.Share.Value.BinarySize()
	}

	delivered := broadcast * parties
	result := ciphertextBytes(transcript.Result.Ciphertext)
	return CommunicationStats{
		BroadcastBytes:          broadcast,
		BroadcastDeliveredBytes: delivered,
		ShareBytes:              shareBytes,
		ResultBytes:             result,
		TotalBytes:              delivered + shareBytes + result,
	}
}

func ciphertextBytes(ct *rlwe.Ciphertext) int {
	if ct == nil {
		return 0
	}
	return ct.BinarySize()
}
