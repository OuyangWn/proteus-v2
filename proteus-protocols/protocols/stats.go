package protocols

import (
	"time"

	"threshold-ckks/threshold"
)

type ProtocolStats struct {
	Duration      time.Duration
	Communication threshold.CommunicationStats
	SPDCmpCalls   int
	SEDTPCalls    int
}

func finishStats(start time.Time, children ...ProtocolStats) ProtocolStats {
	stats := ProtocolStats{Duration: time.Since(start)}
	for _, child := range children {
		stats.Communication = addCommunication(stats.Communication, child.Communication)
		stats.SPDCmpCalls += child.SPDCmpCalls
		stats.SEDTPCalls += child.SEDTPCalls
	}
	return stats
}

func spdCmpStats(stats threshold.ProtocolStats) ProtocolStats {
	return ProtocolStats{Communication: stats.Communication, SPDCmpCalls: 1}
}

func sedtpStats(stats threshold.ProtocolStats) ProtocolStats {
	return ProtocolStats{Communication: stats.Communication, SEDTPCalls: 1}
}

func addCommunication(a, b threshold.CommunicationStats) threshold.CommunicationStats {
	return threshold.CommunicationStats{
		BroadcastBytes:          a.BroadcastBytes + b.BroadcastBytes,
		BroadcastDeliveredBytes: a.BroadcastDeliveredBytes + b.BroadcastDeliveredBytes,
		ShareBytes:              a.ShareBytes + b.ShareBytes,
		ResultBytes:             a.ResultBytes + b.ResultBytes,
		TotalBytes:              a.TotalBytes + b.TotalBytes,
	}
}
