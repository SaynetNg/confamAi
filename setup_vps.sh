#!/bin/bash
# ========================================================
# Confam AI - WhatsApp Bot 24/7 VPS Deployment Script
# Tested on Ubuntu 20.04 / 22.04 / 24.04 (Free Cloud VPS)
# ========================================================

set -e

echo ">>> Updating packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

echo ">>> Installing Node.js 20 LTS..."
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs git

echo ">>> Installing Google Chrome & Chromium dependencies..."
sudo apt-get install -y wget gnupg ca-certificates
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt-get update -y
sudo apt-get install -y google-chrome-stable fonts-ipafont-gothic fonts-wqy-zenhei fonts-thai-tlwg fonts-kacst fonts-freefont-ttf libxss1 --no-install-recommends

echo ">>> Installing PM2 process manager..."
sudo npm install -g pm2

echo ">>> Installing Bot dependencies..."
npm install

echo ">>> Starting Confam AI with PM2..."
pm2 start wwebjs_bot.js --name "confamai-bot"
pm2 save
pm2 startup | tail -n 1 | bash || true

echo "=========================================================="
echo " Confam AI Bot is now RUNNING 24/7!"
echo " Visit: http://YOUR_SERVER_IP:3000 to scan the QR code"
echo " To monitor logs: pm2 logs confamai-bot"
echo "=========================================================="
