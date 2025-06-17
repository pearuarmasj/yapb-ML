target "bot" {
  dockerfile = "Dockerfile"
  context = "."
  tags = ["ac-bot:latest"]
  cache-from = ["type=local,src=.buildx-cache"]
  cache-to = ["type=local,dest=.buildx-cache,mode=max"]
}
