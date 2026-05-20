package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	baseURL    string
	httpClient *http.Client
}

type Request struct {
	Query          string   `json:"query"`
	Limit          int      `json:"limit"`
	DenseWeight    *float64 `json:"dense_weight,omitempty"`
	OCRWeight      *float64 `json:"ocr_weight,omitempty"`
	FlorenceWeight *float64 `json:"florence_weight,omitempty"`
	Diagnostics    bool     `json:"include_diagnostics"`
}

type Response struct {
	Query   string             `json:"query"`
	Limit   int                `json:"limit"`
	Weights map[string]float64 `json:"weights"`
	Results []Result           `json:"results"`
}

type Result struct {
	ImageID        string          `json:"image_id"`
	Score          float64         `json:"score"`
	Path           string          `json:"path"`
	Text           string          `json:"text"`
	OCRText        string          `json:"ocr_text"`
	Caption        string          `json:"caption"`
	ScoreBreakdown *ScoreBreakdown `json:"score_breakdown"`
}

type ScoreBreakdown struct {
	DenseScore    float64 `json:"dense_score"`
	OCRScore      float64 `json:"ocr_score"`
	FlorenceScore float64 `json:"florence_score"`
	DenseRank     *int    `json:"dense_rank"`
	OCRRank       *int    `json:"ocr_rank"`
	FlorenceRank  *int    `json:"florence_rank"`
}

func New(baseURL string, timeout time.Duration) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func NewWithHTTPClient(baseURL string, httpClient *http.Client) *Client {
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: httpClient,
	}
}

func (c *Client) Search(ctx context.Context, request Request) (Response, error) {
	body, err := json.Marshal(request)
	if err != nil {
		return Response{}, err
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/search", bytes.NewReader(body))
	if err != nil {
		return Response{}, err
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpResponse, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return Response{}, err
	}
	defer httpResponse.Body.Close()
	if httpResponse.StatusCode != http.StatusOK {
		return Response{}, fmt.Errorf("gateway search failed: %s", httpResponse.Status)
	}
	var response Response
	if err := json.NewDecoder(httpResponse.Body).Decode(&response); err != nil {
		return Response{}, err
	}
	return response, nil
}
