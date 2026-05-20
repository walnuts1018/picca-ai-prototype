# syntax=docker/dockerfile:1
FROM golang:1.26.3-trixie AS builder

ENV ROOT=/build
ARG BUILD_TAGS=""
ARG TARGETOS
ARG TARGETARCH
RUN mkdir ${ROOT}
WORKDIR ${ROOT}

COPY debug-web/go.mod debug-web/go.sum* ./

RUN --mount=type=cache,target=/go/pkg/mod/ \
    --mount=type=cache,target=/root/.cache/go-build,sharing=locked \
    go mod download -x

COPY debug-web/ .
RUN --mount=type=cache,target=/go/pkg/mod/ \
    --mount=type=cache,target=/root/.cache/go-build,sharing=locked \
    GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH} go build -trimpath -o /out/debug-web ./cmd/debug-web && \
    chmod +x /out/debug-web

FROM gcr.io/distroless/cc-debian13:nonroot
WORKDIR /app

COPY --from=builder /out/debug-web /app/debug-web
COPY debug-web/internal/web/templates /app/internal/web/templates

EXPOSE 8080

ENTRYPOINT ["/app/debug-web"]
