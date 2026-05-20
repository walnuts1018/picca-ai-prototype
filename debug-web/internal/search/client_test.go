package search

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestClientSearch(t *testing.T) {
	httpClient := &http.Client{
		Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if r.URL.Path != "/search" {
				t.Fatalf("path = %s", r.URL.Path)
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: io.NopCloser(strings.NewReader(
					`{"query":"torii","limit":5,"weights":{"dense":2},"results":[{"image_id":"debug/a.jpg","score":1.23,"path":"s3://images/debug/a.jpg","text":"torii","caption":"red gate","score_breakdown":{"dense_score":2,"ocr_score":0,"florence_score":0}}]}`,
				)),
			}, nil
		}),
	}

	client := NewWithHTTPClient("http://gateway", httpClient)
	response, err := client.Search(context.Background(), Request{Query: "torii", Limit: 5, Diagnostics: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Results) != 1 {
		t.Fatalf("results = %d", len(response.Results))
	}
	if response.Results[0].ImageID != "debug/a.jpg" {
		t.Fatalf("image_id = %q", response.Results[0].ImageID)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}
