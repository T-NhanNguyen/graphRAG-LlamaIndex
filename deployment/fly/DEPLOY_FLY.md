# Fly.io Migration Guide: Decoupling from AWS entirely

We are officially migrating off AWS (App Runner/ECR/S3) for a fully standalone, $0/month deployment on Fly.io using their free-tier. Your entire application logic AND the pre-computed database index will be bundled straight into a localized Docker container.

## 1. What Just Happened?

- **Docker Image Changes:** Your `Dockerfile.query` now explicitly embeds the `index-vault` directory (which holds your 2GB DuckDB Knowledge Graph) deep into the container payload during the build step.
- **Boot Script Overhaul:** We've rewritten `start_query.sh`. Now, if it boots without `S3_BUCKET_NAME`, it naturally bypasses AWS sync requests, recognizes the locally bundled DuckDB vault, and properly registers it for the MCP server.
- **Config Generated:** We added a custom `fly.toml` specifically tailored for your GraphRAG query engine to hit the Fly.io free tier.

## 2. Deploy it!

Run these commands in your powershell console inside the `graphRAG-LlamaIndex` folder:

### Install & Login to Fly.io

1. Install the CLI: `iwr https://fly.io/install.ps1 -useb | iex`
2. Authenticate: `fly auth login`

### Launch the App!

1. Set up the secrets so your new server can still talk to OpenRouter.
   ```powershell
   fly secrets set OPENROUTER_API_KEY="your-secret-key-goes-here" -c deployment/fly/fly.toml
   ```
2. Trigger the automated remote build:
   ```powershell
   fly deploy -c deployment/fly/fly.toml
   ```
   _Note: Because we are packaging a ~2GB folder into the image payload, the initial upload/build phase might take 2-4 minutes depending on your internet upload speed. Fly.io servers are incredibly fast at compressing and unzipping it though!_

## 3. Link Backend to Vercel

Once `fly deploy` finishes, it will print a URL in the console (e.g. `https://graphrag-query.fly.dev`).

1. Go to your Vercel Dashboard -> Settings -> Environment Variables.
2. Update **NEXT_PUBLIC_API_URL** to point to your new Fly.io URL.
3. Keep `DATABASE_URL` pointing to `file:./db.sqlite` and `SKIP_ENV_VALIDATION=1`.
4. Hit **Redeploy** on Vercel.

Enjoy your entirely free and AWS-independent tech stack!
