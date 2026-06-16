package mcp

import (
	"context"
	"net/http"
	"testing"

	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func intPtr(i int) *int { return &i }

func TestNewHTTPTransport_StreamableFields(t *testing.T) {
	ctx := context.Background()
	logger := logging.New()
	client := &http.Client{}

	tr := newHTTPTransport(ctx, HTTPServerConfig{
		BaseURL:              "https://example.com/mcp",
		ProtocolType:         StreamableHTTP,
		HTTPClient:           client,
		DisableStandaloneSSE: true,
		MaxRetries:           intPtr(-1),
	}, logger)

	st, ok := tr.(*mcpsdk.StreamableClientTransport)
	if !ok {
		t.Fatalf("expected *StreamableClientTransport, got %T", tr)
	}
	if st.Endpoint != "https://example.com/mcp" {
		t.Errorf("Endpoint = %q, want https://example.com/mcp", st.Endpoint)
	}
	if !st.DisableStandaloneSSE {
		t.Errorf("DisableStandaloneSSE = false, want true")
	}
	if st.MaxRetries != -1 {
		t.Errorf("MaxRetries = %d, want -1", st.MaxRetries)
	}
	if st.HTTPClient != client {
		t.Errorf("HTTPClient not the supplied override")
	}
}

func TestNewHTTPTransport_MaxRetriesNilKeepsDefault(t *testing.T) {
	tr := newHTTPTransport(context.Background(), HTTPServerConfig{
		BaseURL:      "https://example.com/mcp",
		ProtocolType: StreamableHTTP,
	}, logging.New())

	st := tr.(*mcpsdk.StreamableClientTransport)
	// nil MaxRetries leaves the zero value; go-sdk maps 0 to its default (5).
	if st.MaxRetries != 0 {
		t.Errorf("MaxRetries = %d, want 0 (go-sdk default sentinel)", st.MaxRetries)
	}
}

func TestNewHTTPTransport_TokenBearerClient(t *testing.T) {
	tr := newHTTPTransport(context.Background(), HTTPServerConfig{
		BaseURL:      "https://example.com/mcp",
		ProtocolType: StreamableHTTP,
		Token:        "abc",
	}, logging.New())

	st := tr.(*mcpsdk.StreamableClientTransport)
	if st.HTTPClient == nil || st.HTTPClient == http.DefaultClient {
		t.Errorf("Token should produce a non-default bearer client")
	}
}

func TestNewHTTPTransport_SSEAndDefault(t *testing.T) {
	for _, pt := range []ServerProtocolType{SSE, ""} {
		tr := newHTTPTransport(context.Background(), HTTPServerConfig{
			BaseURL:      "https://example.com/mcp",
			ProtocolType: pt,
		}, logging.New())
		if _, ok := tr.(*mcpsdk.SSEClientTransport); !ok {
			t.Errorf("protocol %q: expected *SSEClientTransport, got %T", pt, tr)
		}
	}
}
