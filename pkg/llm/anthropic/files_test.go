package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
)

const messagesOKResponse = `{"content":[{"type":"text","text":"ok"}],"model":"claude-opus-4-8","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`

// captureRequest spins up a test server that records the request body and headers
// for the first /v1/messages call and replies with a minimal successful response.
type captured struct {
	body        map[string]interface{}
	betaHeader  string
	uploadCalls int
}

func newMessagesServer(t *testing.T, cap *captured) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("expected /v1/messages, got %s", r.URL.Path)
		}
		cap.betaHeader = r.Header.Get("anthropic-beta")
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &cap.body); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(messagesOKResponse))
	}))
}

func TestUploadUserData(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/files" {
			t.Fatalf("expected /v1/files endpoint, got %s", r.URL.Path)
		}
		if got := r.Header.Get("anthropic-beta"); got != betaFilesAPI {
			t.Fatalf("expected anthropic-beta %q, got %q", betaFilesAPI, got)
		}
		if err := r.ParseMultipartForm(1024); err != nil {
			t.Fatalf("failed to parse multipart form: %v", err)
		}
		_, header, err := r.FormFile("file")
		if err != nil {
			t.Fatalf("expected file form field: %v", err)
		}
		if header.Filename != "data.csv" {
			t.Fatalf("expected filename data.csv, got %s", header.Filename)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"file_abc123","type":"file","filename":"data.csv"}`))
	}))
	defer server.Close()

	client := NewClient("test-key", WithModel(ClaudeOpus45), WithBaseURL(server.URL))

	fileID, err := client.UploadUserData(context.Background(), strings.NewReader("a,b\n1,2\n"), "data.csv", "")
	if err != nil {
		t.Fatalf("UploadUserData failed: %v", err)
	}
	if fileID != "file_abc123" {
		t.Fatalf("expected file_abc123, got %s", fileID)
	}
}

func TestUploadUserDataImplementsInterface(t *testing.T) {
	var _ interfaces.FileUploader = NewClient("test-key")
}

func TestGenerateAttachesFileDocumentBlock(t *testing.T) {
	cap := &captured{}
	server := newMessagesServer(t, cap)
	defer server.Close()

	client := NewClient("test-key", WithModel(ClaudeOpus45), WithBaseURL(server.URL))

	if _, err := client.Generate(context.Background(), "summarize", WithFileID("file_abc123")); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if cap.betaHeader != betaFilesAPI {
		t.Fatalf("expected beta header %q, got %q", betaFilesAPI, cap.betaHeader)
	}

	block := lastUserFileBlock(t, cap.body, 1)
	if block["type"] != "document" {
		t.Fatalf("expected document block, got %v", block["type"])
	}
	source, _ := block["source"].(map[string]interface{})
	if source["type"] != "file" || source["file_id"] != "file_abc123" {
		t.Fatalf("expected file source with file_abc123, got %v", source)
	}
}

func TestGenerateCodeExecutionAttachesContainerUploadAndTool(t *testing.T) {
	cap := &captured{}
	server := newMessagesServer(t, cap)
	defer server.Close()

	client := NewClient("test-key", WithModel(ClaudeOpus45), WithBaseURL(server.URL))

	_, err := client.Generate(context.Background(), "analyze the spreadsheet",
		WithFileID("file_abc123"),
		WithCodeExecution(),
	)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if !strings.Contains(cap.betaHeader, betaFilesAPI) || !strings.Contains(cap.betaHeader, betaCodeExecution) {
		t.Fatalf("expected beta header to include files + code execution, got %q", cap.betaHeader)
	}

	block := lastUserFileBlock(t, cap.body, 1)
	if block["type"] != "container_upload" || block["file_id"] != "file_abc123" {
		t.Fatalf("expected container_upload block with file_abc123, got %v", block)
	}

	tools, ok := cap.body["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Fatalf("expected one tool, got %v", cap.body["tools"])
	}
	tool, _ := tools[0].(map[string]interface{})
	if tool["type"] != codeExecutionToolType || tool["name"] != codeExecutionToolName {
		t.Fatalf("expected code execution tool, got %v", tool)
	}
}

func TestCodeExecutionRequiresUploadedFileID(t *testing.T) {
	client := NewClient("test-key", WithModel(ClaudeOpus45))

	_, err := client.Generate(context.Background(), "analyze",
		WithFileData("data.csv", "text/csv", []byte("a,b\n1,2\n")),
		WithCodeExecution(),
	)
	if err == nil || !strings.Contains(err.Error(), "must be an uploaded FileID") {
		t.Fatalf("expected uploaded FileID error, got %v", err)
	}
}

func TestFileInputRequiresExactlyOneSource(t *testing.T) {
	client := NewClient("test-key", WithModel(ClaudeOpus45))

	_, err := client.Generate(context.Background(), "summarize",
		func(o *interfaces.GenerateOptions) {
			o.FileInputs = append(o.FileInputs, interfaces.FileInput{FileID: "file_abc123", FileURL: "https://example.com/x.pdf"})
		},
	)
	if err == nil || !strings.Contains(err.Error(), "exactly one of FileID, FileURL, or FileData") {
		t.Fatalf("expected exactly-one-source error, got %v", err)
	}
}

func TestFileInputsRejectedOnBedrockAndCacheCombo(t *testing.T) {
	client := NewClient("test-key", WithModel(ClaudeOpus45))

	_, err := client.Generate(context.Background(), "summarize",
		WithFileID("file_abc123"),
		WithCacheSystemMessage(),
		interfaces.WithSystemMessage("system"),
	)
	if err == nil || !strings.Contains(err.Error(), "cannot be combined with file inputs") {
		t.Fatalf("expected cache+file combination error, got %v", err)
	}
}

func TestFileBlocksInlineData(t *testing.T) {
	builder := &fileRequestBuilder{files: []interfaces.FileInput{
		{Filename: "logo.png", FileData: "data:image/png;base64,aGVsbG8="},
		{Filename: "doc.pdf", FileData: "data:application/pdf;base64,aGVsbG8="},
	}}

	blocks, err := builder.fileBlocks()
	if err != nil {
		t.Fatalf("fileBlocks failed: %v", err)
	}
	if blocks[0].Type != "image" || blocks[0].Source.MediaType != "image/png" || blocks[0].Source.Data != "aGVsbG8=" {
		t.Fatalf("expected image block with bare base64, got %+v", blocks[0])
	}
	if blocks[1].Type != "document" || blocks[1].Source.Type != "base64" || blocks[1].Source.MediaType != "application/pdf" {
		t.Fatalf("expected document base64 block, got %+v", blocks[1])
	}
}

// lastUserFileBlock returns the content block at index `idx` of the last user
// message in the captured request body (index 0 is the prompt text block).
func lastUserFileBlock(t *testing.T, body map[string]interface{}, idx int) map[string]interface{} {
	t.Helper()
	messages, _ := body["messages"].([]interface{})
	if len(messages) == 0 {
		t.Fatalf("no messages in request body")
	}
	last, _ := messages[len(messages)-1].(map[string]interface{})
	content, _ := last["content"].([]interface{})
	if len(content) <= idx {
		t.Fatalf("expected at least %d content blocks, got %v", idx+1, content)
	}
	block, _ := content[idx].(map[string]interface{})
	return block
}

// fileTestTool is a minimal interfaces.Tool for exercising the tool-calling path.
type fileTestTool struct{ name string }

func (t *fileTestTool) Name() string                                             { return t.name }
func (t *fileTestTool) Description() string                                      { return "test tool" }
func (t *fileTestTool) Run(ctx context.Context, input string) (string, error)    { return "42", nil }
func (t *fileTestTool) Execute(ctx context.Context, args string) (string, error) { return "42", nil }
func (t *fileTestTool) Parameters() map[string]interfaces.ParameterSpec {
	return map[string]interfaces.ParameterSpec{
		"q": {Type: "string", Description: "query", Required: true},
	}
}

// hasBlockType reports whether any content block of the message is of the given type.
func hasBlockType(msg map[string]interface{}, blockType string) bool {
	content, ok := msg["content"].([]interface{})
	if !ok {
		return false
	}
	for _, b := range content {
		if block, ok := b.(map[string]interface{}); ok && block["type"] == blockType {
			return true
		}
	}
	return false
}

func TestGenerateWithToolsAttachesFileAndMergesTools(t *testing.T) {
	cap := &captured{}
	server := newMessagesServer(t, cap)
	defer server.Close()

	client := NewClient("test-key", WithModel(ClaudeOpus45), WithBaseURL(server.URL))

	_, err := client.GenerateWithTools(context.Background(), "analyze the spreadsheet",
		[]interfaces.Tool{&fileTestTool{name: "lookup"}},
		WithFileID("file_abc123"),
		WithCodeExecution(),
	)
	if err != nil {
		t.Fatalf("GenerateWithTools failed: %v", err)
	}

	if !strings.Contains(cap.betaHeader, betaFilesAPI) || !strings.Contains(cap.betaHeader, betaCodeExecution) {
		t.Fatalf("expected beta header to include files + code execution, got %q", cap.betaHeader)
	}

	block := lastUserFileBlock(t, cap.body, 1)
	if block["type"] != "container_upload" || block["file_id"] != "file_abc123" {
		t.Fatalf("expected container_upload block with file_abc123, got %v", block)
	}

	// The client function tool and the hosted code-execution tool must both survive.
	tools, ok := cap.body["tools"].([]interface{})
	if !ok || len(tools) != 2 {
		t.Fatalf("expected client tool + code execution tool, got %v", cap.body["tools"])
	}
	var sawClientTool, sawCodeExec bool
	for _, tv := range tools {
		tool, _ := tv.(map[string]interface{})
		if tool["name"] == "lookup" {
			sawClientTool = true
		}
		if tool["type"] == codeExecutionToolType && tool["name"] == codeExecutionToolName {
			sawCodeExec = true
		}
	}
	if !sawClientTool || !sawCodeExec {
		t.Fatalf("expected both client tool and code execution tool, got %v", tools)
	}
	if cap.body["tool_choice"] == nil {
		t.Fatalf("expected tool_choice to be carried through, got nil")
	}
}

func TestGenerateWithToolsPinsFileToOriginalPrompt(t *testing.T) {
	var bodies []map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]interface{}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "application/json")
		if len(bodies) == 1 {
			// First turn: ask to call the client tool, forcing a second iteration.
			_, _ = w.Write([]byte(`{"content":[{"type":"tool_use","id":"t1","name":"lookup","input":{"q":"x"}}],"model":"claude-opus-4-8","stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`))
			return
		}
		_, _ = w.Write([]byte(messagesOKResponse))
	}))
	defer server.Close()

	client := NewClient("test-key", WithModel(ClaudeOpus45), WithBaseURL(server.URL))

	_, err := client.GenerateWithTools(context.Background(), "analyze the spreadsheet",
		[]interfaces.Tool{&fileTestTool{name: "lookup"}},
		WithFileID("file_abc123"),
		WithCodeExecution(),
	)
	if err != nil {
		t.Fatalf("GenerateWithTools failed: %v", err)
	}
	if len(bodies) < 2 {
		t.Fatalf("expected at least 2 iterations, got %d", len(bodies))
	}

	// On the second turn the message list has grown (a user tool-result message
	// is appended). The container_upload must remain on the original prompt and
	// must NOT have migrated onto the newly-appended tool-result message — which
	// is what a naive "attach to the last user message" rule would have done.
	second := bodies[1]
	messages, _ := second["messages"].([]interface{})
	if len(messages) < 2 {
		t.Fatalf("expected original prompt + tool_result, got %d messages", len(messages))
	}
	first, _ := messages[0].(map[string]interface{})
	if !hasBlockType(first, "container_upload") {
		t.Fatalf("expected container_upload on the original prompt, got %v", first)
	}
	last, _ := messages[len(messages)-1].(map[string]interface{})
	if hasBlockType(last, "container_upload") {
		t.Fatalf("container_upload leaked onto the tool-result message: %v", last)
	}
}

// TestStreamingRejectsFileInputsAndCodeExecution verifies the streaming paths
// fail fast rather than silently dropping file inputs / code execution.
func TestStreamingRejectsFileInputsAndCodeExecution(t *testing.T) {
	client := NewClient("test-key", WithModel(ClaudeOpus45))

	cases := []struct {
		name string
		opt  interfaces.GenerateOption
	}{
		{"file input", WithFileID("file_abc123")},
		{"code execution", WithCodeExecution()},
	}
	for _, tc := range cases {
		t.Run(tc.name+"/GenerateStream", func(t *testing.T) {
			_, err := client.GenerateStream(context.Background(), "hi", tc.opt)
			if err == nil || !strings.Contains(err.Error(), "not supported with streaming") {
				t.Fatalf("expected streaming-unsupported error, got %v", err)
			}
		})
		t.Run(tc.name+"/GenerateWithToolsStream", func(t *testing.T) {
			_, err := client.GenerateWithToolsStream(context.Background(), "hi", nil, tc.opt)
			if err == nil || !strings.Contains(err.Error(), "not supported with streaming") {
				t.Fatalf("expected streaming-unsupported error, got %v", err)
			}
		})
	}
}
