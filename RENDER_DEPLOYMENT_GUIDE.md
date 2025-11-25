# Deploying SmartNote Backend to Render

This guide will walk you through deploying the SmartNote backend to Render's free tier.

## Prerequisites

1. A GitHub account
2. A Render account (free)
3. Your Supabase credentials

## Step-by-Step Deployment

### 1. Fork or Ensure Your Repository is on GitHub

Make sure your SmartNote repository is available on GitHub. If you haven't already, push your latest changes:

```bash
git push origin main
```

### 2. Sign Up for Render

1. Go to [render.com](https://render.com)
2. Click "Get Started for Free"
3. Sign up using GitHub (easiest option)

### 3. Create a New Web Service

1. Once logged in to Render, click "New" → "Web Service"
2. Connect your GitHub repository:
   - Click "Connect account" next to GitHub
   - Grant Render access to your repositories
   - Find and select your SmartNote repository

### 4. Configure Your Web Service

Fill in the following settings:

- **Name**: `smartnote-backend` (or any name you prefer)
- **Region**: Choose the region closest to you
- **Branch**: `main` (or your preferred branch)
- **Root Directory**: Leave empty (the project root)
- **Environment**: `Python 3`
- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: `python backend/main.py`

### 5. Set Environment Variables

Click "Advanced" to reveal the environment variables section. Add the following variables:

| Key | Value |
|-----|-------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase anonymous key |
| `SUPABASE_SERVICE_ROLE_KEY` | Your Supabase service role key |

You can find these values in your Supabase project dashboard:
1. Go to your Supabase project
2. Click "Project Settings" → "API"
3. Copy the "Project URL" and "anon public" key
4. For the service role key, go to "Project Settings" → "API" and copy the "service_role" key

### 6. Deploy

1. Click "Create Web Service"
2. Render will automatically start building and deploying your application
3. Wait for the deployment to complete (this may take several minutes due to the PyTorch dependencies)

### 7. Verify Deployment

1. Once deployment is complete, you'll see a URL for your service (something like `https://smartnote-backend.onrender.com`)
2. Test the health endpoint: `https://your-render-url.onrender.com/health`
3. You should receive a JSON response indicating the service is running

## Important Notes for Render Deployment

### Free Tier Limitations

1. **Sleep Mode**: Render's free tier web services automatically sleep after 15 minutes of inactivity
2. **Wake-up Time**: When someone accesses your service after it has slept, it may take 10-30 seconds to wake up
3. **Monthly Usage**: Free tier includes 750 hours of runtime per month

### Model Loading Considerations

1. **Initial Load Time**: The first request to your service may be slow as models are loaded into memory
2. **Memory Usage**: Ensure your models fit within Render's free tier memory limits
3. **Caching**: Render will keep your service in memory while active, so subsequent requests will be faster

### Updating Your Frontend

Once deployed, you'll need to update your frontend to point to the new backend URL:

1. Find your Render service URL (visible in the Render dashboard)
2. Update your frontend API configuration to use this URL instead of `localhost:8000`

## Troubleshooting

### Common Issues

1. **Build Failures**: If your build fails due to memory issues, try reducing model sizes or optimizing dependencies
2. **Runtime Errors**: Check the logs in the Render dashboard for detailed error messages
3. **Slow Responses**: First requests after sleep or model loading may be slow

### Checking Logs

1. Go to your Render dashboard
2. Select your web service
3. Click "Logs" to view real-time logs

### Redeployment

Render automatically redeploys when you push changes to your GitHub repository. You can also manually trigger a deployment:

1. Go to your web service in the Render dashboard
2. Click "Manual Deploy" → "Deploy latest commit"

## Cost Optimization

1. **Keep Services Warm**: If you need to prevent sleeping, you can set up a cron job to ping your service periodically
2. **Monitor Usage**: Keep an eye on your monthly usage to stay within free tier limits
3. **Optimize Models**: Consider model quantization or pruning to reduce memory usage

## Next Steps

1. Test all API endpoints with your new Render backend
2. Update your frontend to use the Render URL
3. Monitor usage and performance in the Render dashboard