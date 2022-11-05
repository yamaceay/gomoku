FROM golang:latest
WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY *.go ./
RUN go build -o /docker-gomoku
EXPOSE 8080
CMD ["/docker-gomoku"]