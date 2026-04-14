const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = 3000;
const PYTHON_BACKEND = 'http://127.0.0.1:8080';

// 1. API Proxy - 将所有 /api 请求转发到原来的 Python server.py
app.use('/api', createProxyMiddleware({
  target: PYTHON_BACKEND,
  changeOrigin: true,
  logLevel: 'debug'
}));

// 2. Static Files - 托管前端构建后的产物
app.use(express.static(path.join(__dirname, 'public')));

// 3. Fallback to index.html (SPA support)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`\n🚀 Alpasim Webviz Debugger V2 is running!`);
  console.log(`🔗 URL: http://localhost:${PORT}`);
  console.log(`📡 Proxying API to: ${PYTHON_BACKEND}\n`);
});
