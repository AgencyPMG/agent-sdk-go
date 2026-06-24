package agent

import (
	"context"
	"testing"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
)

// capturingLLM records the generate options it was called with, so tests can
// assert that agent-level options were threaded through to the LLM.
type capturingLLM struct {
	captured []interfaces.GenerateOption
}

func (m *capturingLLM) Name() string            { return "capturingLLM" }
func (m *capturingLLM) SupportsStreaming() bool { return false }

func (m *capturingLLM) Generate(ctx context.Context, prompt string, options ...interfaces.GenerateOption) (string, error) {
	m.captured = options
	return "ok", nil
}

func (m *capturingLLM) GenerateWithTools(ctx context.Context, prompt string, tools []interfaces.Tool, options ...interfaces.GenerateOption) (string, error) {
	m.captured = options
	return "ok", nil
}

func (m *capturingLLM) GenerateDetailed(ctx context.Context, prompt string, options ...interfaces.GenerateOption) (*interfaces.LLMResponse, error) {
	m.captured = options
	return &interfaces.LLMResponse{Content: "ok", Model: "capturing"}, nil
}

func (m *capturingLLM) GenerateWithToolsDetailed(ctx context.Context, prompt string, tools []interfaces.Tool, options ...interfaces.GenerateOption) (*interfaces.LLMResponse, error) {
	m.captured = options
	return &interfaces.LLMResponse{Content: "ok", Model: "capturing"}, nil
}

func (m *capturingLLM) resolved() *interfaces.GenerateOptions {
	opts := &interfaces.GenerateOptions{}
	for _, o := range m.captured {
		o(opts)
	}
	return opts
}

func TestWithGenerateOptionsThreadsThroughToLLM(t *testing.T) {
	llm := &capturingLLM{}
	a, err := NewAgent(
		WithLLM(llm),
		WithGenerateOptions(
			interfaces.WithFileID("file_123"),
			interfaces.WithCodeExecution(),
		),
	)
	if err != nil {
		t.Fatalf("NewAgent failed: %v", err)
	}

	if _, err := a.Run(context.Background(), "analyze this spreadsheet"); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	got := llm.resolved()
	if !got.EnableCodeExecution {
		t.Fatalf("expected CodeExecution to be threaded through to the LLM")
	}
	if len(got.FileInputs) != 1 || got.FileInputs[0].FileID != "file_123" {
		t.Fatalf("expected the file input threaded through, got %+v", got.FileInputs)
	}
}
