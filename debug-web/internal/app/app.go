package app

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/config"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/queue"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/s3client"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/search"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/storage"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/web"
)

type App struct {
	cfg            config.Config
	repo           *storage.Repository
	publisher      *queue.Client
	resultConsumer *queue.Client
	server         *http.Server
}

func New(cfg config.Config) (*App, error) {
	repo, err := storage.Open(cfg.SQLitePath)
	if err != nil {
		return nil, err
	}
	publisher, err := queue.New(cfg.RabbitMQURL, cfg.RabbitMQHeartbeat)
	if err != nil {
		_ = repo.Close()
		return nil, err
	}
	resultConsumer, err := queue.New(cfg.RabbitMQURL, cfg.RabbitMQHeartbeat)
	if err != nil {
		_ = publisher.Close()
		_ = repo.Close()
		return nil, err
	}
	s3, err := s3client.New(context.Background(), cfg)
	if err != nil {
		_ = resultConsumer.Close()
		_ = publisher.Close()
		_ = repo.Close()
		return nil, err
	}
	handler, err := web.New(cfg, repo, publisher, search.New(cfg.GatewayBaseURL, cfg.SearchTimeout), s3)
	if err != nil {
		_ = resultConsumer.Close()
		_ = publisher.Close()
		_ = repo.Close()
		return nil, err
	}
	return &App{
		cfg:            cfg,
		repo:           repo,
		publisher:      publisher,
		resultConsumer: resultConsumer,
		server: &http.Server{
			Addr:              cfg.Address(),
			Handler:           handler.Routes(),
			ReadHeaderTimeout: 10 * time.Second,
		},
	}, nil
}

func (a *App) Run(ctx context.Context) error {
	consumerCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		err := a.resultConsumer.ConsumeResults(consumerCtx, a.cfg.RabbitMQResultQueue, func(message queue.ResultMessage) error {
			return web.ConsumeResultEvent(context.Background(), a.repo, message)
		})
		if err != nil && !errors.Is(err, context.Canceled) {
			log.Printf("result consumer stopped: %v", err)
		}
	}()
	log.Printf("debug web listening on %s", a.cfg.Address())
	if err := a.server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("listen: %w", err)
	}
	return nil
}

func (a *App) Close() error {
	var firstErr error
	if a.server != nil {
		if err := a.server.Close(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			firstErr = err
		}
	}
	if a.resultConsumer != nil {
		if err := a.resultConsumer.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if a.publisher != nil {
		if err := a.publisher.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if a.repo != nil {
		if err := a.repo.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}
