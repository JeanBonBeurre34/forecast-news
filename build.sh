docker build -t global-trends . && mkdir -p data && docker run --rm -v "$(pwd)/data":/app/data global-trends
