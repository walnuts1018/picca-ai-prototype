package queue

import "testing"

func TestParseResultMessage(t *testing.T) {
	body := []byte(`{"image_id":"debug/a.jpg","status":"failed","occurred_at":"2026-05-21T01:02:03Z","error_message":"boom"}`)
	msg, err := ParseResultMessage(body)
	if err != nil {
		t.Fatal(err)
	}
	if msg.Status != "failed" {
		t.Fatalf("status = %q", msg.Status)
	}
	if msg.ErrorMessage != "boom" {
		t.Fatalf("error_message = %q", msg.ErrorMessage)
	}
}

func TestParseResultMessageRejectsBadStatus(t *testing.T) {
	if _, err := ParseResultMessage([]byte(`{"image_id":"debug/a.jpg","status":"queued","occurred_at":"2026-05-21T01:02:03Z"}`)); err == nil {
		t.Fatal("expected error")
	}
}
