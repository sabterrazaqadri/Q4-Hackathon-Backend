# Deploying Backend to Render

This guide will walk you through deploying the Physical AI & Humanoid Robotics backend API to Render.

## Prerequisites

1. A [GitHub account](https://github.com)
2. A [Render account](https://render.com) (free tier available)
3. Your code pushed to a GitHub repository
4. API keys for:
   - OpenAI API
   - Qdrant Cloud
   - Cohere API

## Step 1: Push Your Code to GitHub

If you haven't already, push your code to GitHub:

```bash
git add .
git commit -m "Prepare backend for Render deployment"
git push origin main
```

## Step 2: Create a New Web Service on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** button in the top right
3. Select **"Web Service"**
4. Connect your GitHub account if you haven't already
5. Select your repository from the list

## Step 3: Configure the Web Service

### Basic Settings:

- **Name**: `physical-ai-backend` (or any name you prefer)
- **Region**: Choose the region closest to your users
- **Branch**: `main` (or your deployment branch)
- **Root Directory**: `Backend`
- **Runtime**: `Docker`
- **Instance Type**: `Free` (or choose a paid plan for better performance)

### Build Settings:

Render will automatically detect your `Dockerfile` in the Backend directory.

- **Dockerfile Path**: `Backend/dockerfile` (Render should auto-detect this)
- **Docker Context**: `Backend`

## Step 4: Add Environment Variables

In the "Environment" section, add the following environment variables:

| Key | Value | Notes |
|-----|-------|-------|
| `OPENAI_API_KEY` | `your-openai-api-key` | Get from OpenAI dashboard |
| `QDRANT_URL` | `your-qdrant-url` | Get from Qdrant Cloud |
| `QDRANT_API_KEY` | `your-qdrant-api-key` | Get from Qdrant Cloud |
| `COHERE_API_KEY` | `your-cohere-api-key` | Get from Cohere dashboard |
| `DATABASE_URL` | `sqlite:///./data/test.db` | SQLite database path |
| `TEXTBOOK_COLLECTION_NAME` | `textbook_content` | Qdrant collection name |
| `PORT` | `8000` | Port (Render auto-assigns) |

**IMPORTANT**: Never commit these API keys to your repository!

## Step 5: Add Persistent Disk (for SQLite Database)

1. Scroll down to **"Disk"** section
2. Click **"Add Disk"**
3. Configure:
   - **Name**: `backend-data`
   - **Mount Path**: `/app/data`
   - **Size**: `1 GB` (Free tier includes 1GB)

This ensures your SQLite database persists between deployments.

## Step 6: Configure Health Check (Optional but Recommended)

- **Health Check Path**: `/health`

This allows Render to monitor your service and ensure it's running correctly.

## Step 7: Deploy!

1. Review all settings
2. Click **"Create Web Service"**
3. Render will start building and deploying your application
4. Wait for the build to complete (this may take 5-10 minutes)

## Step 8: Verify Deployment

Once deployed, you'll get a URL like: `https://physical-ai-backend.onrender.com`

Test your API:

1. **Health Check**:
   ```bash
   curl https://your-app-name.onrender.com/health
   ```

2. **API Documentation**:
   Visit `https://your-app-name.onrender.com/docs` in your browser

3. **Test RAG Query**:
   ```bash
   curl -X POST https://your-app-name.onrender.com/api/v1/rag/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is ROS 2?"}'
   ```

## Step 9: Update Frontend Configuration

Update your Docusaurus frontend to use the new backend URL:

1. Edit `docusaurus/.env`:
   ```env
   REACT_APP_API_URL=https://your-app-name.onrender.com/api/v1/chat/completions
   ```

2. Or update `docusaurus/docusaurus.config.ts`:
   ```typescript
   customFields: {
     apiUrl: 'https://your-app-name.onrender.com/api/v1/chat/completions',
   },
   ```

## Automatic Deployments

Render supports automatic deployments:

1. Go to your service settings
2. Enable **"Auto-Deploy"** from the selected branch
3. Now, every push to your branch will trigger a new deployment

## Monitoring & Logs

- **Logs**: View real-time logs in the Render dashboard under "Logs" tab
- **Metrics**: Monitor CPU, memory, and bandwidth usage
- **Events**: Track deployments and service events

## Important Notes

### Free Tier Limitations:

- Services spin down after 15 minutes of inactivity
- First request after spin-down will be slow (cold start ~30-60 seconds)
- 750 hours/month of runtime
- Consider upgrading to a paid plan for production use

### Database Considerations:

- SQLite works fine for development/small projects
- For production with multiple instances, consider migrating to PostgreSQL:
  ```bash
  # Render offers free PostgreSQL databases
  # Add a PostgreSQL database and update DATABASE_URL
  ```

### CORS Configuration:

Make sure your backend allows requests from your frontend domain. Update in `src/config/settings.py`:

```python
allowed_origins = [
    "http://localhost:3000",
    "https://your-frontend-domain.vercel.app",
    "https://your-app-name.onrender.com"
]
```

## Troubleshooting

### Build Fails:

1. Check the build logs in Render dashboard
2. Ensure all dependencies are in `requirements.txt`
3. Verify Dockerfile syntax

### Service Crashes:

1. Check runtime logs for errors
2. Verify environment variables are set correctly
3. Ensure database path is correct: `/app/data/test.db`

### Slow Response Times:

1. Free tier instances spin down when idle
2. Upgrade to paid tier for always-on instances
3. Consider using Redis for caching

### Database Issues:

1. Ensure persistent disk is mounted at `/app/data`
2. Check disk usage in dashboard
3. Verify `DATABASE_URL` points to `/app/data/test.db`

## Alternative: Using Blueprint (Infrastructure as Code)

Render also supports deploying using `render.yaml`. A `render.yaml` file has been created in the root directory for easy deployment:

```bash
# From Render dashboard:
# 1. New -> Blueprint
# 2. Connect your repository
# 3. Render will automatically detect render.yaml
```

## Useful Commands

```bash
# View logs (requires Render CLI)
render logs -f

# SSH into your service (paid plans only)
render ssh

# Restart service
# (Use dashboard or API)
```

## Next Steps

1. Set up monitoring and alerts
2. Configure custom domain (paid plans)
3. Set up CI/CD pipeline
4. Add database backups
5. Implement rate limiting
6. Add authentication if needed

## Support

- [Render Documentation](https://render.com/docs)
- [Render Community Forum](https://community.render.com)
- [GitHub Issues](https://github.com/your-repo/issues)

## Cost Estimation

- **Free Tier**: $0/month (with limitations)
- **Starter Plan**: $7/month (always-on, faster builds)
- **Standard Plan**: $25/month (better performance)
- **Pro Plan**: $85/month (production-ready)

Choose based on your traffic and requirements.
