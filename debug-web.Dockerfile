FROM golang:1.26.3 AS build
WORKDIR /src
COPY go.work ./
COPY debug-web/go.mod debug-web/go.sum* ./debug-web/
WORKDIR /src/debug-web
RUN go mod download
COPY debug-web/ ./
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /out/debug-web ./cmd/debug-web

FROM gcr.io/distroless/base-debian12
WORKDIR /app
COPY --from=build /out/debug-web /app/debug-web
COPY debug-web/internal/web/templates /app/internal/web/templates
VOLUME ["/data"]
EXPOSE 8080
ENTRYPOINT ["/app/debug-web"]
