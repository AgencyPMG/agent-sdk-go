package gemini

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"google.golang.org/genai"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
)

// WithFileID attaches an already-uploaded Gemini file (by its file URI) to the
// model input. It delegates to interfaces.WithFileID so the option surface is
// identical across providers.
func WithFileID(fileID string) interfaces.GenerateOption {
	return interfaces.WithFileID(fileID)
}

// WithFileURL attaches an externally reachable file URL to the model input.
// Note: Gemini does not support external file URLs in this SDK path yet (see
// buildGeminiFileParts); prefer uploading and using WithFileID.
func WithFileURL(fileURL string) interfaces.GenerateOption {
	return interfaces.WithFileURL(fileURL)
}

// WithFileData attaches inline file bytes to the model input.
func WithFileData(filename, mimeType string, data []byte) interfaces.GenerateOption {
	return interfaces.WithFileData(filename, mimeType, data)
}

// WithCodeExecution enables Gemini's hosted code-execution tool so the model can
// run code (e.g. pandas over an uploaded CSV/XLSX) to answer the prompt.
func WithCodeExecution() interfaces.GenerateOption {
	return interfaces.WithCodeExecution()
}

// UploadUserDataFile uploads a local file via the Gemini Files API and returns its
// file URI, which can be passed to WithFileID.
func (c *GeminiClient) UploadUserDataFile(ctx context.Context, path string) (string, error) {
	file, err := c.genaiClient.Files.UploadFromPath(ctx, path, &genai.UploadFileConfig{
		DisplayName: filepath.Base(path),
		MIMEType:    interfaces.ContentTypeForFilename(path),
	})
	if err != nil {
		return "", fmt.Errorf("failed to upload file: %w", err)
	}
	return file.URI, nil
}

// UploadUserData uploads file content via the Gemini Files API and returns its file
// URI, which can be passed to WithFileID.
func (c *GeminiClient) UploadUserData(ctx context.Context, reader io.Reader, filename, contentType string) (string, error) {
	if contentType == "" {
		contentType = interfaces.ContentTypeForFilename(filename)
	}
	file, err := c.genaiClient.Files.Upload(ctx, reader, &genai.UploadFileConfig{
		DisplayName: filename,
		MIMEType:    contentType,
	})
	if err != nil {
		return "", fmt.Errorf("failed to upload file: %w", err)
	}
	return file.URI, nil
}

// buildGeminiFileParts converts the request's file inputs into genai content parts.
// Uploaded files (FileID = file URI) become FileData parts; inline data becomes an
// InlineData blob. External file URLs are not supported yet and return an error
// rather than being silently dropped.
func buildGeminiFileParts(params *interfaces.GenerateOptions) ([]*genai.Part, error) {
	if params == nil || len(params.FileInputs) == 0 {
		return nil, nil
	}
	parts := make([]*genai.Part, 0, len(params.FileInputs))
	for i, f := range params.FileInputs {
		switch {
		case f.FileURL != "":
			return nil, fmt.Errorf("gemini file input %d: external file URLs are not supported yet; upload the file and use WithFileID", i)
		case f.FileID != "":
			parts = append(parts, &genai.Part{FileData: &genai.FileData{FileURI: f.FileID}})
		case f.FileData != "":
			mimeType, data, err := parseDataURL(f.FileData)
			if err != nil {
				return nil, fmt.Errorf("gemini file input %d: %w", i, err)
			}
			parts = append(parts, &genai.Part{InlineData: &genai.Blob{MIMEType: mimeType, Data: data}})
		default:
			return nil, fmt.Errorf("gemini file input %d: no file source set", i)
		}
	}
	return parts, nil
}

// parseDataURL splits a "data:<mime>;base64,<data>" URL into its MIME type and
// decoded bytes.
func parseDataURL(s string) (string, []byte, error) {
	if !strings.HasPrefix(s, "data:") {
		return "", nil, fmt.Errorf("invalid data URL")
	}
	rest := strings.TrimPrefix(s, "data:")
	comma := strings.IndexByte(rest, ',')
	if comma < 0 {
		return "", nil, fmt.Errorf("invalid data URL: missing comma separator")
	}
	meta, encoded := rest[:comma], rest[comma+1:]
	mimeType := meta
	if semi := strings.IndexByte(meta, ';'); semi >= 0 {
		mimeType = meta[:semi]
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return "", nil, fmt.Errorf("invalid base64 data: %w", err)
	}
	return mimeType, data, nil
}
