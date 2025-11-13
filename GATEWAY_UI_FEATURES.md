# 🚀 Gateway UI - Enhanced Features

## Overview

The Enhanced Gateway UI provides comprehensive monitoring and management for LLM endpoints with real-time metrics, team tracking, and MLflow integration.

## 🎯 Key Features

### 1. Dashboard View (Default)

Access: **📊 Dashboard** tab

#### High-Level Metrics
- **Total Requests (24h)** - Real-time request count with trend
- **Active Endpoints** - Number of live endpoints across providers
- **Total Cost (24h)** - Aggregated spending with cost trends
- **Avg Latency** - Performance metrics across all endpoints

#### Visualizations
- **Requests Over Time** - Line chart showing request patterns (1H, 6H, 24H, 7D)
- **Cost by Provider** - Doughnut chart breaking down costs (OpenAI, Anthropic, Azure, Google)
- **Latency Distribution** - Bar chart showing latency buckets (<0.5s to >3s)

#### Team Usage Tracking
View usage broken down by team:
- **ML Team** - Requests, cost, and endpoint count
- **Product Team** - Team-specific metrics
- **Research Team** - Independent tracking

Each team card shows:
- Total requests
- Total cost
- Number of endpoints
- Active status

---

### 2. Endpoints List View

Access: **🔌 Endpoints** tab

#### Table View
- **Endpoint Name** - With URL endpoint path
- **Provider** - Color-coded badges (OpenAI, Anthropic, Azure, Google)
- **Model** - Specific model version
- **Requests (24h)** - Recent traffic
- **Avg Latency** - Performance metric
- **Cost (24h)** - Spending per endpoint
- **Status** - Active/Inactive with visual indicators

**Click any endpoint to view detailed metrics** →

---

### 3. Endpoint Detail View

Click on any endpoint to see comprehensive metrics and integrations.

#### Header Information
- **Endpoint Name & Model**
- **Provider & Team**
- **Status Badge**
- **Action Buttons** (Edit, Test)

#### MLflow Integration Links

**📊 View Traces in MLflow**
- Direct link to MLflow experiments filtered by endpoint
- URL: `http://localhost:5000/#/experiments/search?searchInput=gateway_{endpoint_name}`
- View all traces, inputs, outputs, and performance data

**📝 View Prompts Registry**
- Link to MLflow Prompt Registry filtered by endpoint
- URL: `http://localhost:5000/#/prompts?search={endpoint_name}`
- Access registered prompts, versions, and templates

#### Real-Time Metrics

**Live Updates** (refreshes every 5 seconds)
- Real-time requests per minute chart
- Visual indication of live data with pulsing dot

#### Usage Metrics (6 Cards)

1. **Total Requests** - Request count with trend (↑ 15.2% from last hour)
2. **Success Rate** - Percentage of successful calls (99.8%)
3. **P95 Latency** - 95th percentile latency (1.2s)
4. **Total Cost** - Spending in last 24 hours ($52.30)
5. **Avg Tokens/Request** - Token usage (Input: 320, Output: 130)
6. **Error Rate** - Failure percentage with error count (0.2%, 3 errors)

#### Visualization Charts

**Cost Over Time** (24h)
- Line chart showing hourly cost patterns
- Helps identify expensive time periods

**Latency Over Time** (24h)
- Performance trends throughout the day
- Spot latency spikes and patterns

---

## 🎨 Design Features

### Color-Coded Providers
- **OpenAI** - Green (#10a37f)
- **Anthropic** - Gold (#d4a273)
- **Azure** - Blue (#0078d4)
- **Google** - Blue (#4285f4)

### Status Indicators
- **Active** - Green badge with pulsing dot
- **Inactive** - Red badge

### Visual Metrics
- **Trend Arrows** - ↑ positive (green), ↓ negative (red)
- **Live Indicator** - Pulsing green dot for real-time data
- **Hover Effects** - Interactive table rows and buttons

---

## 📊 Mock Data

The current implementation uses realistic mock data:

### Endpoints
- `prod-gpt4` - OpenAI GPT-4o (8.5K requests, $52.30/day)
- `dev-claude` - Anthropic Claude Sonnet (6.2K requests, $38.50/day)
- `azure-gpt35` - Azure GPT-3.5 Turbo (12.4K requests, $18.90/day)
- `prod-gemini` - Google Gemini 1.5 Pro (4.3K requests, $25.40/day)

### Teams
- **ML Team** - 5 endpoints, 12.4K requests, $82.50/day
- **Product Team** - 4 endpoints, 8.2K requests, $42.17/day
- **Research Team** - 3 endpoints, 4.0K requests, $18.00/day

---

## 🔄 Real-Time Updates

The UI simulates real-time updates:
- **Dashboard metrics** - Static snapshots with trends
- **Endpoint detail view** - Request count updates every 5 seconds
- **Live charts** - Animated line charts showing real-time data

In production, these would connect to:
- WebSocket for live metrics
- Time-series database (InfluxDB, Prometheus)
- MLflow tracking backend

---

## 🔗 MLflow Integration

### Traces Integration
Each endpoint links to MLflow experiments:
```
http://localhost:5000/#/experiments/search?searchInput=gateway_prod-gpt4
```

View:
- All traces for this endpoint
- Request/response data
- Latency metrics
- Token usage
- Cost per call

### Prompts Integration
Access the MLflow Prompt Registry:
```
http://localhost:5000/#/prompts?search=prod-gpt4
```

View:
- Registered prompt templates
- Prompt versions
- Associated models
- Prompt performance

---

## 📱 Views Summary

| View | Purpose | Key Features |
|------|---------|--------------|
| **Dashboard** | High-level overview | Aggregate metrics, charts, team usage |
| **Endpoints List** | Browse all endpoints | Searchable table, quick metrics |
| **Endpoint Detail** | Deep dive into one endpoint | Real-time metrics, charts, MLflow links |

---

## 🚀 Usage Examples

### View Dashboard
1. Open `http://localhost:5001/gateway`
2. Default view shows dashboard with all metrics

### Explore Endpoint Details
1. Click **🔌 Endpoints** tab
2. Click on any endpoint row (e.g., `prod-gpt4`)
3. View real-time metrics, costs, and latency
4. Click **📊 View Traces** to see MLflow data

### Monitor Team Usage
1. Stay on **📊 Dashboard** tab
2. Scroll to "Team Usage" section
3. See breakdown by ML Team, Product Team, Research Team

### Access MLflow Integration
1. Go to endpoint detail view
2. Click **📊 View Traces in MLflow** - opens filtered experiment view
3. Click **📝 View Prompts Registry** - opens filtered prompts view

---

## 🎯 Inspired By

This implementation draws inspiration from:
- **LiteLLM Proxy UI** - Real-time monitoring and provider management
- **MLflow UI** - Design system and navigation patterns
- **Datadog/Grafana** - Metrics visualization and dashboard layouts

---

## 🔮 Next Steps (Production)

To make this production-ready:

1. **Backend API** - Connect to real metrics database
2. **WebSocket** - Live updates without polling
3. **Authentication** - User/team-based access control
4. **Real MLflow Integration** - Auto-create experiments per endpoint
5. **Alerting** - Set thresholds for latency, cost, errors
6. **Cost Budgets** - Per-team spending limits
7. **Audit Logs** - Track all configuration changes
8. **Export** - Download metrics as CSV/JSON

---

## 🌐 Quick Links

- **Gateway UI**: http://localhost:5001/gateway
- **MLflow UI**: http://localhost:5000
- **MLflow Traces**: http://localhost:5000/#/experiments
- **MLflow Prompts**: http://localhost:5000/#/prompts

---

**Enjoy exploring the Gateway UI!** 🎉

