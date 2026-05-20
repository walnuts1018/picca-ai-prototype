package queue

import (
	"context"
	"fmt"
	"time"

	"github.com/rabbitmq/amqp091-go"
)

type Client struct {
	conn *amqp091.Connection
	ch   *amqp091.Channel
}

func New(url string, heartbeat time.Duration) (*Client, error) {
	conn, err := amqp091.DialConfig(url, amqp091.Config{
		Heartbeat: heartbeat,
	})
	if err != nil {
		return nil, err
	}
	ch, err := conn.Channel()
	if err != nil {
		_ = conn.Close()
		return nil, err
	}
	return &Client{conn: conn, ch: ch}, nil
}

func (c *Client) Close() error {
	if c == nil {
		return nil
	}
	if c.ch != nil {
		_ = c.ch.Close()
	}
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

func (c *Client) PublishJob(queueName string, message JobMessage) error {
	body, err := message.Marshal()
	if err != nil {
		return err
	}
	if _, err := c.ch.QueueDeclare(queueName, true, false, false, false, nil); err != nil {
		return err
	}
	return c.ch.PublishWithContext(context.Background(), "", queueName, false, false, amqp091.Publishing{
		DeliveryMode: amqp091.Persistent,
		ContentType:  "application/json",
		Body:         body,
	})
}

func (c *Client) ConsumeResults(ctx context.Context, queueName string, handler func(ResultMessage) error) error {
	if _, err := c.ch.QueueDeclare(queueName, true, false, false, false, nil); err != nil {
		return err
	}
	if err := c.ch.Qos(20, 0, false); err != nil {
		return err
	}
	deliveries, err := c.ch.Consume(queueName, "", false, false, false, false, nil)
	if err != nil {
		return err
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case delivery, ok := <-deliveries:
			if !ok {
				return fmt.Errorf("result consumer closed")
			}
			msg, err := ParseResultMessage(delivery.Body)
			if err != nil {
				_ = delivery.Nack(false, false)
				continue
			}
			if err := handler(msg); err != nil {
				_ = delivery.Nack(false, true)
				continue
			}
			_ = delivery.Ack(false)
		}
	}
}
