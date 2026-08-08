package protocols

import (
	"fmt"
	"time"

	"threshold-ckks/threshold"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type SMARLAAgentInput struct {
	QTable         [][]*rlwe.Ciphertext
	State          []*rlwe.Ciphertext
	NextState      []*rlwe.Ciphertext
	ExploitBit     *rlwe.Ciphertext
	RandomAction   []*rlwe.Ciphertext
	ActionOverride []*rlwe.Ciphertext
}

type SMARLAOptions struct {
	Length          int
	MaskKey         []byte
	PublicSeed      []byte
	CallIDPrefix    string
	RefreshActions  bool
	RefreshNextMax  bool
	RefreshUpdatedQ bool
}

type SMARLAStepResult struct {
	UpdatedQTables  [][][]*rlwe.Ciphertext
	Actions         [][]*rlwe.Ciphertext
	CurrentRows     [][]*rlwe.Ciphertext
	CurrentQTotal   *rlwe.Ciphertext
	NextMaxQTotal   *rlwe.Ciphertext
	Delta           *rlwe.Ciphertext
	SPASA           []SPASAResult
	NextMax         []SMAAResult
	Refresh         *threshold.SEDTPResult
	ActionRefreshes [][]*threshold.SEDTPResult
	QRefreshes      [][][]*threshold.SEDTPResult
	Stats           ProtocolStats
}

func DefaultSMARLAOptions(length int, maskKey []byte, publicSeed []byte, callIDPrefix string) SMARLAOptions {
	return SMARLAOptions{Length: length, MaskKey: maskKey, PublicSeed: publicSeed, CallIDPrefix: callIDPrefix, RefreshActions: true, RefreshNextMax: true, RefreshUpdatedQ: true}
}

// SMARLAStep executes one encrypted VDN training step after observations/rewards are already encrypted.
func SMARLAStep(ctx HomContext, keys *threshold.ThresholdKeys, agents []SMARLAAgentInput, reward, alpha, gamma *rlwe.Ciphertext, opts SMARLAOptions) SMARLAStepResult {
	start := time.Now()
	validateSMARLAInputs(ctx, keys, agents, reward, alpha, gamma, opts)

	childStats := make([]ProtocolStats, 0, len(agents)*5+1)
	spasa := make([]SPASAResult, len(agents))
	actions := make([][]*rlwe.Ciphertext, len(agents))
	var actionRefreshes [][]*threshold.SEDTPResult
	if opts.RefreshActions {
		actionRefreshes = make([][]*threshold.SEDTPResult, len(agents))
	}
	rows := make([][]*rlwe.Ciphertext, len(agents))
	currentValues := make([]*rlwe.Ciphertext, len(agents))
	for i, agent := range agents {
		if len(agent.ActionOverride) > 0 {
			row := SAQSAWithStats(ctx, agent.State, agent.QTable)
			childStats = append(childStats, row.Stats)
			actions[i], rows[i] = agent.ActionOverride, row.Row
		} else {
			spasaOpts := DefaultSPASAOptions(opts.Length, opts.MaskKey, opts.PublicSeed, fmt.Sprintf("%s:agent:%d:spasa", opts.CallIDPrefix, i))
			spasa[i] = SPASA(ctx, keys, agent.State, agent.QTable, agent.ExploitBit, agent.RandomAction, spasaOpts)
			childStats = append(childStats, spasa[i].Stats)
			actions[i], rows[i] = spasa[i].Action, spasa[i].Row
		}
		if opts.RefreshActions {
			actionRefreshes[i] = refreshCiphertextVector(ctx, keys, actions[i], opts, fmt.Sprintf("agent:%d:action", i), &childStats)
		}
		current := SQSAWithStats(ctx, agent.State, actions[i], agent.QTable)
		childStats = append(childStats, current.Stats)
		currentValues[i] = current.Value
	}
	currentQTotal := addMany(ctx, currentValues, "SMARLA current Q-total")

	nextMax := make([]SMAAResult, len(agents))
	nextMaxValues := make([]*rlwe.Ciphertext, len(agents))
	for i, agent := range agents {
		nextRow := SAQSAWithStats(ctx, agent.NextState, agent.QTable)
		childStats = append(childStats, nextRow.Stats)
		nextOpts := DefaultSMAAOptions(opts.Length, opts.MaskKey, opts.PublicSeed, fmt.Sprintf("%s:agent:%d:next", opts.CallIDPrefix, i))
		nextMax[i] = SMAA(ctx, keys, nextRow.Row, nextOpts)
		childStats = append(childStats, nextMax[i].Stats)
		nextMaxValues[i] = nextMax[i].Max
	}
	nextMaxQTotal := addMany(ctx, nextMaxValues, "SMARLA next max Q-total")

	var refresh *threshold.SEDTPResult
	if opts.RefreshNextMax {
		refreshOpts := threshold.DefaultSEDTPOptions(opts.Length, opts.PublicSeed, opts.CallIDPrefix+":refresh-next-max")
		refreshed := threshold.SEDTPWithTranscript(ctx.Params, keys, keys.PK, nextMaxQTotal, refreshOpts)
		nextMaxQTotal, refresh = refreshed.Ciphertext, &refreshed
		childStats = append(childStats, sedtpStats(refreshed.Stats))
	}

	delta := computeTDDelta(ctx, reward, gamma, nextMaxQTotal, currentQTotal)
	updated := make([][][]*rlwe.Ciphertext, len(agents))
	var qRefreshes [][][]*threshold.SEDTPResult
	if opts.RefreshUpdatedQ {
		qRefreshes = make([][][]*threshold.SEDTPResult, len(agents))
	}
	for i, agent := range agents {
		squa := SQUAWithStats(ctx, agent.QTable, delta, agent.State, actions[i], alpha)
		childStats = append(childStats, squa.Stats)
		updated[i] = squa.QTable
		if opts.RefreshUpdatedQ {
			qRefreshes[i] = refreshUpdatedQTable(ctx, keys, updated[i], opts, i, &childStats)
		}
	}

	return SMARLAStepResult{
		UpdatedQTables:  updated,
		Actions:         actions,
		CurrentRows:     rows,
		CurrentQTotal:   currentQTotal,
		NextMaxQTotal:   nextMaxQTotal,
		Delta:           delta,
		SPASA:           spasa,
		NextMax:         nextMax,
		Refresh:         refresh,
		ActionRefreshes: actionRefreshes,
		QRefreshes:      qRefreshes,
		Stats:           finishStats(start, childStats...),
	}
}

func refreshCiphertextVector(ctx HomContext, keys *threshold.ThresholdKeys, values []*rlwe.Ciphertext, opts SMARLAOptions, label string, childStats *[]ProtocolStats) []*threshold.SEDTPResult {
	refreshes := make([]*threshold.SEDTPResult, len(values))
	for i := range values {
		callID := fmt.Sprintf("%s:%s:%d", opts.CallIDPrefix, label, i)
		refreshes[i] = new(threshold.SEDTPResult)
		*refreshes[i] = threshold.SEDTPWithTranscript(ctx.Params, keys, keys.PK, values[i], threshold.DefaultSEDTPOptions(opts.Length, opts.PublicSeed, callID))
		values[i] = refreshes[i].Ciphertext
		*childStats = append(*childStats, sedtpStats(refreshes[i].Stats))
	}
	return refreshes
}

func refreshUpdatedQTable(ctx HomContext, keys *threshold.ThresholdKeys, qTable [][]*rlwe.Ciphertext, opts SMARLAOptions, agent int, childStats *[]ProtocolStats) [][]*threshold.SEDTPResult {
	refreshes := make([][]*threshold.SEDTPResult, len(qTable))
	for s := range qTable {
		refreshes[s] = make([]*threshold.SEDTPResult, len(qTable[s]))
		for a := range qTable[s] {
			callID := fmt.Sprintf("%s:agent:%d:q:%d:%d", opts.CallIDPrefix, agent, s, a)
			refreshes[s][a] = new(threshold.SEDTPResult)
			*refreshes[s][a] = threshold.SEDTPWithTranscript(ctx.Params, keys, keys.PK, qTable[s][a], threshold.DefaultSEDTPOptions(opts.Length, opts.PublicSeed, callID))
			qTable[s][a] = refreshes[s][a].Ciphertext
			*childStats = append(*childStats, sedtpStats(refreshes[s][a].Stats))
		}
	}
	return refreshes
}

func computeTDDelta(ctx HomContext, reward, gamma, nextMaxQTotal, currentQTotal *rlwe.Ciphertext) *rlwe.Ciphertext {
	discounted := mulRelinRescale(ctx, gamma, nextMaxQTotal, "SMARLA gamma*nextMax")
	target, err := ctx.Evaluator.AddNew(reward, discounted)
	if err != nil {
		panic(fmt.Errorf("SMARLA reward+discounted: %w", err))
	}
	delta, err := ctx.Evaluator.SubNew(target, currentQTotal)
	if err != nil {
		panic(fmt.Errorf("SMARLA delta: %w", err))
	}
	return delta
}

func addMany(ctx HomContext, values []*rlwe.Ciphertext, label string) *rlwe.Ciphertext {
	if len(values) == 0 {
		panic(label + ": no values")
	}
	acc := values[0].CopyNew()
	for i := 1; i < len(values); i++ {
		if err := ctx.Evaluator.Add(acc, values[i], acc); err != nil {
			panic(fmt.Errorf("%s add index %d: %w", label, i, err))
		}
	}
	return acc
}

func validateSMARLAInputs(ctx HomContext, keys *threshold.ThresholdKeys, agents []SMARLAAgentInput, reward, alpha, gamma *rlwe.Ciphertext, opts SMARLAOptions) {
	if ctx.Evaluator == nil {
		panic("homomorphic evaluator must not be nil")
	}
	if keys == nil || keys.PK == nil || len(keys.PartySK) == 0 {
		panic("threshold keys must include public key and party shares")
	}
	if len(agents) == 0 {
		panic("SMARLA requires at least one agent")
	}
	if reward == nil || alpha == nil || gamma == nil {
		panic("reward, alpha, and gamma ciphertexts must not be nil")
	}
	if opts.Length <= 0 || opts.Length > ctx.Params.MaxSlots() {
		panic("invalid SMARLA length")
	}
	if len(opts.MaskKey) == 0 || len(opts.PublicSeed) == 0 || opts.CallIDPrefix == "" {
		panic("SMARLA mask key, public seed, and call id prefix must not be empty")
	}
	for i, agent := range agents {
		validateSOHLInputs(ctx, agent.State, agent.QTable)
		validateSOHLInputs(ctx, agent.NextState, agent.QTable)
		_, actions := validateSOHLInputs(ctx, agent.State, agent.QTable)
		if len(agent.ActionOverride) > 0 {
			if len(agent.ActionOverride) != actions {
				panic(fmt.Sprintf("agent %d action override dimension mismatch", i))
			}
			for _, action := range agent.ActionOverride {
				if action == nil {
					panic(fmt.Sprintf("agent %d nil action override", i))
				}
			}
			continue
		}
		if agent.ExploitBit == nil {
			panic(fmt.Sprintf("agent %d exploit bit must not be nil", i))
		}
		if len(agent.RandomAction) == 0 {
			panic(fmt.Sprintf("agent %d random action must not be empty", i))
		}
	}
}
