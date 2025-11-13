# MLflow Gateway UI

A modern web interface for managing LLM provider endpoints and authentication, integrated with MLflow's design system.

## 🎯 Overview

The Gateway UI allows administrators to:
- **Create and manage model endpoints** for different LLM providers
- **Configure authentication** (API keys, Azure credentials, AWS Bedrock, etc.)
- **Control access** - developers can use endpoints without managing credentials
- **Monitor status** - see which endpoints are active/inactive
- **Test endpoints** - verify configuration before deployment

Inspired by [LiteLLM](https://github.com/BerriAI/litellm), this provides a centralized gateway pattern for enterprise LLM deployments.

## 🚀 Quick Start

### 1. Start the Gateway UI Server

```bash
# From the project root
cd /Users/ahmed.bilal/Desktop/gateway-oss

# Start the Gateway UI (port 5001)
venv311/bin/python mlflowlite/ui/gateway_server.py
```

### 2. Start MLflow UI (Optional)

```bash
# In a separate terminal
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### 3. Open in Browser

```
Gateway UI: http://localhost:5001/gateway
MLflow UI:  http://localhost:5000
```

## 📸 Features

### Endpoint Management

- **List View**: See all configured endpoints with provider, model, and status
- **Search**: Filter endpoints by name, provider, or model
- **Create/Edit**: Configure new endpoints or modify existing ones
- **Delete**: Remove unused endpoints
- **Test**: Verify endpoint configuration

### Supported Providers

| Provider | Models | Authentication |
|----------|--------|----------------|
| **OpenAI** | GPT-4, GPT-3.5 Turbo, GPT-4o | API Key |
| **Anthropic** | Claude 4.5 Sonnet/Haiku, Claude 3 | API Key |
| **Azure OpenAI** | GPT-4, GPT-3.5 | API Key + Endpoint |
| **Google (Gemini)** | Gemini 1.5 Pro/Flash | API Key |
| **AWS Bedrock** | Claude, Titan | AWS Credentials + Region |
| **Cohere** | Command R/R+ | API Key |

### Endpoint Configuration

Each endpoint includes:
- **Name**: Unique identifier (e.g., `prod-gpt4`, `dev-claude`)
- **Provider**: LLM provider selection
- **Model**: Specific model to use
- **Authentication**: API key or credentials (stored securely)
- **Base URL**: Optional custom endpoint
- **Status**: Active/Inactive toggle
- **Description**: Notes about the endpoint

### Usage Pattern

```python
# Developer uses endpoint without managing credentials
import mlflowlite as mla

# Reference the gateway endpoint
response = mla.completion(
    endpoint="prod-gpt4",  # Uses gateway configuration
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 🎨 Design

The Gateway UI follows MLflow's design system:
- **Header**: Matches MLflow navigation (Experiments, Models, Prompts, **Gateway**)
- **Colors**: Uses MLflow's color palette (#43C9ED primary)
- **Typography**: Same font stack as MLflow
- **Components**: Consistent with MLflow UI patterns

## 🔧 Technical Details

### Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   Gateway UI    │         │   MLflow UI      │
│   (Port 5001)   │         │   (Port 5000)    │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         └───────────┬───────────────┘
                     │
              ┌──────▼──────┐
              │  MLflowlite │
              │   Backend   │
              └─────────────┘
```

### Files

- `mlflowlite/ui/gateway/index.html` - Main UI (self-contained HTML/CSS/JS)
- `mlflowlite/ui/gateway_server.py` - Flask server to serve the UI
- `mlflowlite/ui/README.md` - This documentation

### Mock Data

Currently uses in-memory JavaScript storage for demonstration. In production, this would connect to:
- Database for endpoint configuration
- Secrets manager for API keys
- MLflow backend for integration

## 🔐 Security Considerations

For production deployment:

1. **API Key Storage**: Use AWS Secrets Manager, HashiCorp Vault, or similar
2. **Access Control**: Implement role-based permissions (admin vs developer)
3. **Audit Logs**: Track who creates/modifies endpoints
4. **Encryption**: Encrypt sensitive data at rest and in transit
5. **Rate Limiting**: Prevent abuse of gateway endpoints

## 📝 Future Enhancements

- [ ] Backend API integration
- [ ] Real-time endpoint health monitoring
- [ ] Usage analytics and cost tracking
- [ ] Endpoint versioning
- [ ] A/B testing configuration
- [ ] Rate limit settings per endpoint
- [ ] User permissions and RBAC
- [ ] API key rotation
- [ ] Webhook notifications

## 🤝 Integration with MLflowlite

To integrate with `mlflowlite` completion API:

```python
# In mlflowlite/litellm_style_api.py

def completion(..., endpoint: Optional[str] = None, ...):
    if endpoint:
        # Resolve endpoint from Gateway configuration
        config = get_gateway_endpoint(endpoint)
        model = config['model']
        api_key = config['api_key']
        # ... use gateway config
    else:
        # Direct model call
        # ... existing logic
```

## 📚 References

- [LiteLLM Proxy](https://docs.litellm.ai/docs/simple_proxy) - Inspiration for gateway pattern
- [MLflow UI](https://mlflow.org/docs/latest/tracking.html#tracking-ui) - Design system reference
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) - API compatibility

## 📄 License

Part of the MLflowlite project.

