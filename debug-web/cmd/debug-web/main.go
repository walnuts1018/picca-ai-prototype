package main

import (
	"context"
	"log"

	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/app"
	"github.com/walnuts1018/picca-ai-prototype/debug-web/internal/config"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatal(err)
	}

	application, err := app.New(cfg)
	if err != nil {
		log.Fatal(err)
	}
	defer application.Close()

	if err := application.Run(context.Background()); err != nil {
		log.Fatal(err)
	}
}
