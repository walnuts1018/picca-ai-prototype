package queue

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

type JobMessage struct {
	ImageID string `json:"image_id"`
}

func (m JobMessage) Marshal() ([]byte, error) {
	if strings.TrimSpace(m.ImageID) == "" {
		return nil, fmt.Errorf("image_id must not be blank")
	}
	return json.Marshal(m)
}

type ResultMessage struct {
	ImageID      string `json:"image_id"`
	Status       string `json:"status"`
	OccurredAt   string `json:"occurred_at"`
	ErrorMessage string `json:"error_message,omitempty"`
}

func ParseResultMessage(body []byte) (ResultMessage, error) {
	var msg ResultMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		return ResultMessage{}, err
	}
	msg.ImageID = strings.TrimSpace(msg.ImageID)
	msg.Status = strings.TrimSpace(msg.Status)
	msg.OccurredAt = strings.TrimSpace(msg.OccurredAt)
	msg.ErrorMessage = strings.TrimSpace(msg.ErrorMessage)
	if msg.ImageID == "" {
		return ResultMessage{}, fmt.Errorf("image_id must not be blank")
	}
	switch msg.Status {
	case "processing", "indexed", "failed":
	default:
		return ResultMessage{}, fmt.Errorf("unsupported status: %s", msg.Status)
	}
	if msg.OccurredAt == "" {
		return ResultMessage{}, fmt.Errorf("occurred_at must not be blank")
	}
	if _, err := time.Parse(time.RFC3339, msg.OccurredAt); err != nil {
		return ResultMessage{}, fmt.Errorf("occurred_at: %w", err)
	}
	return msg, nil
}
