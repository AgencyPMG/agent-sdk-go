package anthropic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"os"
	"path/filepath"
	"strings"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
)

// Beta headers and tool identifiers for file input + code execution. These are
// kept as constants so they are easy to bump when Anthropic ships newer versions.
const (
	// betaFilesAPI is required whenever a request references an uploaded file_id
	// (document/image/container_upload blocks) and for uploading via the Files API.
	betaFilesAPI = "files-api-2025-04-14"
	// betaCodeExecution is required to use the hosted code execution tool.
	betaCodeExecution = "code-execution-2025-08-25"
	// codeExecutionToolType is the hosted code execution tool version. The
	// 20250825 build is available on every code-execution-capable model.
	codeExecutionToolType = "code_execution_20250825"
	codeExecutionToolName = "code_execution"
)

// AnthropicClient implements the file-upload capability.
var _ interfaces.FileUploader = (*AnthropicClient)(nil)

// WithFileID attaches an already-uploaded Anthropic file (by its file_id) to the
// model input. It delegates to interfaces.WithFileID so the option surface is
// identical across providers.
func WithFileID(fileID string) interfaces.GenerateOption {
	return interfaces.WithFileID(fileID)
}

// WithFileURL attaches an externally reachable file URL to the model input.
// Anthropic supports URL sources for document content blocks.
func WithFileURL(fileURL string) interfaces.GenerateOption {
	return interfaces.WithFileURL(fileURL)
}

// WithFileData attaches inline file bytes to the model input as a base64 data URL.
func WithFileData(filename, mimeType string, data []byte) interfaces.GenerateOption {
	return interfaces.WithFileData(filename, mimeType, data)
}

// WithCodeExecution enables Anthropic's hosted code execution tool so the model
// can run code (e.g. pandas over an uploaded CSV/XLSX) to answer the prompt.
// Files attached via WithFileID are mounted into the sandbox container as
// container_upload inputs.
func WithCodeExecution() interfaces.GenerateOption {
	return interfaces.WithCodeExecution()
}

// fileUploadResponse is the relevant part of the Files API upload response.
type fileUploadResponse struct {
	ID string `json:"id"`
}

// UploadUserDataFile uploads a local file via the Anthropic Files API and returns
// its file_id, which can be passed to WithFileID.
func (c *AnthropicClient) UploadUserDataFile(ctx context.Context, path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer func() { _ = file.Close() }()

	name := filepath.Base(path)
	return c.UploadUserData(ctx, file, name, interfaces.ContentTypeForFilename(name))
}

// UploadUserData uploads file content via the Anthropic Files API and returns its
// file_id, which can be passed to WithFileID. The returned handle is only valid
// with the Anthropic provider that produced it.
func (c *AnthropicClient) UploadUserData(ctx context.Context, reader io.Reader, filename, contentType string) (string, error) {
	if c.VertexConfig != nil && c.VertexConfig.Enabled {
		return "", fmt.Errorf("anthropic file upload is not supported on Vertex AI")
	}
	if c.BedrockConfig != nil && c.BedrockConfig.Enabled {
		return "", fmt.Errorf("anthropic file upload is not supported on Bedrock")
	}
	if contentType == "" {
		contentType = interfaces.ContentTypeForFilename(filename)
	}
	if contentType == "" {
		contentType = "application/octet-stream"
	}

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename=%q`, filename))
	header.Set("Content-Type", contentType)
	part, err := writer.CreatePart(header)
	if err != nil {
		return "", fmt.Errorf("failed to create multipart file part: %w", err)
	}
	if _, err := io.Copy(part, reader); err != nil {
		return "", fmt.Errorf("failed to write file content: %w", err)
	}
	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("failed to finalize multipart body: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.BaseURL+"/v1/files", &body)
	if err != nil {
		return "", fmt.Errorf("failed to create upload request: %w", err)
	}
	httpReq.Header.Set("Content-Type", writer.FormDataContentType())
	httpReq.Header.Set("X-API-Key", c.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")
	httpReq.Header.Set("anthropic-beta", betaFilesAPI)

	httpResp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("failed to upload file: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read upload response: %w", err)
	}
	if httpResp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("anthropic Files API error (status %d): %s", httpResp.StatusCode, string(respBody))
	}

	var parsed fileUploadResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return "", fmt.Errorf("failed to parse upload response: %w", err)
	}
	if parsed.ID == "" {
		return "", fmt.Errorf("anthropic Files API returned an empty file id")
	}
	return parsed.ID, nil
}

// fileSourceParam is the source of a document/image content block.
type fileSourceParam struct {
	Type      string `json:"type"`                 // "file", "url", or "base64"
	FileID    string `json:"file_id,omitempty"`    // for type "file"
	URL       string `json:"url,omitempty"`        // for type "url"
	MediaType string `json:"media_type,omitempty"` // for type "base64"
	Data      string `json:"data,omitempty"`       // for type "base64"
}

// fileContentBlock is a single content block in a user message that carries files.
type fileContentBlock struct {
	Type   string           `json:"type"`              // "text", "document", "image", "container_upload"
	Text   string           `json:"text,omitempty"`    // for type "text"
	Source *fileSourceParam `json:"source,omitempty"`  // for "document"/"image"
	FileID string           `json:"file_id,omitempty"` // for "container_upload"
}

// fileMessage is a message whose content is an array of blocks (rather than a
// plain string), used to carry file inputs on the final user turn.
type fileMessage struct {
	Role    string             `json:"role"`
	Content []fileContentBlock `json:"content"`
}

// fileCompletionRequest mirrors CompletionRequest but uses json.RawMessage for the
// fields that need content-block / typed-tool shapes when files or code execution
// are in play.
type fileCompletionRequest struct {
	Model         string          `json:"model,omitempty"`
	Messages      json.RawMessage `json:"messages"`
	MaxTokens     int             `json:"max_tokens,omitempty"`
	Temperature   float64         `json:"temperature,omitempty"`
	TopP          float64         `json:"top_p,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	System        string          `json:"system,omitempty"`
	Tools         json.RawMessage `json:"tools,omitempty"`
	Thinking      *ReasoningSpec  `json:"thinking,omitempty"`
}

// fileRequestBuilder transforms a standard request into a file/code-execution
// request when those options are enabled.
type fileRequestBuilder struct {
	files         []interfaces.FileInput
	codeExecution bool
}

// newFileRequestBuilder creates a builder from the generation options.
func newFileRequestBuilder(params *interfaces.GenerateOptions) *fileRequestBuilder {
	if params == nil {
		return &fileRequestBuilder{}
	}
	return &fileRequestBuilder{files: params.FileInputs, codeExecution: params.CodeExecution}
}

// active reports whether any file-input / code-execution handling is required.
func (b *fileRequestBuilder) active() bool {
	return len(b.files) > 0 || b.codeExecution
}

// betaHeader returns the comma-separated anthropic-beta header value required for
// the configured options, or "" if none.
func (b *fileRequestBuilder) betaHeader() string {
	betas := []string{}
	if len(b.files) > 0 {
		betas = append(betas, betaFilesAPI)
	}
	if b.codeExecution {
		betas = append(betas, betaCodeExecution)
	}
	return strings.Join(betas, ",")
}

// validate checks that each file input is well-formed for the chosen mode.
func (b *fileRequestBuilder) validate() error {
	for i, f := range b.files {
		sources := 0
		if f.FileID != "" {
			sources++
		}
		if f.FileURL != "" {
			sources++
		}
		if f.FileData != "" {
			sources++
		}
		if sources != 1 {
			return fmt.Errorf("anthropic file input %d must set exactly one of FileID, FileURL, or FileData", i)
		}
		if b.codeExecution && f.FileID == "" {
			return fmt.Errorf("anthropic code execution file input %d must be an uploaded FileID; URL and inline data are not supported by the code execution container", i)
		}
	}
	return nil
}

// build produces the marshaled request body for the standard Anthropic API.
func (b *fileRequestBuilder) build(req *CompletionRequest) ([]byte, error) {
	if err := b.validate(); err != nil {
		return nil, err
	}

	fileBlocks, err := b.fileBlocks()
	if err != nil {
		return nil, err
	}

	messages, err := buildFileMessages(req.Messages, fileBlocks)
	if err != nil {
		return nil, err
	}

	out := &fileCompletionRequest{
		Model:         req.Model,
		Messages:      messages,
		MaxTokens:     req.MaxTokens,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		StopSequences: req.StopSequences,
		System:        req.System,
		Thinking:      req.Thinking,
	}

	if b.codeExecution {
		tools, err := json.Marshal([]map[string]string{{
			"type": codeExecutionToolType,
			"name": codeExecutionToolName,
		}})
		if err != nil {
			return nil, fmt.Errorf("failed to marshal code execution tool: %w", err)
		}
		out.Tools = tools
	}

	return json.Marshal(out)
}

// fileBlocks converts the file inputs into content blocks. In code-execution mode
// uploaded files become container_upload blocks; otherwise they become document /
// image blocks based on their source.
func (b *fileRequestBuilder) fileBlocks() ([]fileContentBlock, error) {
	blocks := make([]fileContentBlock, 0, len(b.files))
	for i, f := range b.files {
		if b.codeExecution {
			blocks = append(blocks, fileContentBlock{Type: "container_upload", FileID: f.FileID})
			continue
		}

		switch {
		case f.FileID != "":
			blocks = append(blocks, fileContentBlock{
				Type:   "document",
				Source: &fileSourceParam{Type: "file", FileID: f.FileID},
			})
		case f.FileURL != "":
			blocks = append(blocks, fileContentBlock{
				Type:   "document",
				Source: &fileSourceParam{Type: "url", URL: f.FileURL},
			})
		case f.FileData != "":
			mediaType, data, err := parseDataURL(f.FileData)
			if err != nil {
				return nil, fmt.Errorf("anthropic file input %d: %w", i, err)
			}
			blockType := "document"
			if strings.HasPrefix(mediaType, "image/") {
				blockType = "image"
			}
			blocks = append(blocks, fileContentBlock{
				Type:   blockType,
				Source: &fileSourceParam{Type: "base64", MediaType: mediaType, Data: data},
			})
		default:
			return nil, fmt.Errorf("anthropic file input %d: no file source set", i)
		}
	}
	return blocks, nil
}

// buildFileMessages serializes the message list, attaching the file content blocks
// to the last user message. If there is no user message, a new one is appended.
func buildFileMessages(messages []Message, fileBlocks []fileContentBlock) (json.RawMessage, error) {
	lastUser := -1
	for i := range messages {
		if messages[i].Role == "user" {
			lastUser = i
		}
	}

	result := make([]interface{}, 0, len(messages)+1)
	for i := range messages {
		if i == lastUser {
			content := []fileContentBlock{{Type: "text", Text: messages[i].Content}}
			content = append(content, fileBlocks...)
			result = append(result, fileMessage{Role: "user", Content: content})
			continue
		}
		result = append(result, messages[i])
	}

	if lastUser == -1 {
		result = append(result, fileMessage{Role: "user", Content: fileBlocks})
	}

	return json.Marshal(result)
}

// parseDataURL splits a "data:<mime>;base64,<data>" URL into its MIME type and the
// raw base64 payload (Anthropic expects the bare base64 string, not the data URL).
func parseDataURL(s string) (mediaType, data string, err error) {
	if !strings.HasPrefix(s, "data:") {
		return "", "", fmt.Errorf("invalid data URL")
	}
	rest := strings.TrimPrefix(s, "data:")
	comma := strings.IndexByte(rest, ',')
	if comma < 0 {
		return "", "", fmt.Errorf("invalid data URL: missing comma separator")
	}
	meta, encoded := rest[:comma], rest[comma+1:]
	mediaType = meta
	if semi := strings.IndexByte(meta, ';'); semi >= 0 {
		mediaType = meta[:semi]
	}
	if _, decodeErr := base64.StdEncoding.DecodeString(encoded); decodeErr != nil {
		return "", "", fmt.Errorf("invalid base64 data: %w", decodeErr)
	}
	return mediaType, encoded, nil
}
