#!/bin/bash
# Download the latest market_data.db from GitHub Releases.
# Usage: cd data && bash download_latest.sh

REPO="desai0820/tunnel-of-sorrow"
TAG="latest-data"
FILE="market_data.db"

echo "Downloading latest $FILE from $REPO..."
curl -sL "https://github.com/$REPO/releases/download/$TAG/$FILE" -o "$FILE"
echo "Done. $(wc -c < "$FILE" | tr -d ' ') bytes written to $FILE"
