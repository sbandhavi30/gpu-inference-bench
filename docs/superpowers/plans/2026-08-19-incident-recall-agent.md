# Incident Recall Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an SRE Incident Recall Agent using Mastra + Elasticsearch that remembers past incidents, retrieves relevant ones via hybrid search, and applies recency decay so recent fixes outrank stale ones.

**Architecture:** Mastra agent with two custom tools (`recall_incidents` and `mark_outcome`) backed by Elasticsearch hybrid search (BM25 + kNN vector) with Gaussian recency decay scoring. A Next.js chat UI calls a `/api/chat` route that invokes the Mastra agent.

**Tech Stack:** TypeScript, Mastra (`@mastra/core`), Elasticsearch (`@elastic/elasticsearch`), OpenAI (embeddings + LLM), Next.js 14 App Router, Tailwind CSS.

**Spec:** Design documented in conversation — Approach B (custom memory layer + hybrid search).

## Global Constraints

- Node.js >= 20
- TypeScript strict mode
- Mastra `@mastra/core` >= 0.1.0
- Elasticsearch 8.x (cloud or local Docker)
- OpenAI model: `gpt-4o-mini` for LLM, `text-embedding-3-small` for embeddings (1536 dims)
- All `.env` secrets via `.env.local` — never hardcoded
- Index name: `incidents`
- Recency decay: `scale=30d`, `decay=0.5`, `offset=0d`

---

## File Structure

```
incident-recall-agent/
  src/
    mastra/
      agent.ts                  # Mastra agent definition + system prompt
      tools/
        recall.ts               # recall_incidents tool (hybrid search + recency)
        mark-outcome.ts         # mark_outcome tool (write outcome back to ES)
    elasticsearch/
      client.ts                 # ES client singleton
      search.ts                 # hybrid search + function_score query builder
      ingest.ts                 # seed script — creates index + loads incidents.json
    data/
      incidents.json            # 15 realistic fake incidents
    app/
      page.tsx                  # Chat UI
      api/
        chat/
          route.ts              # POST /api/chat → Mastra agent
      globals.css
      layout.tsx
  .env.example
  package.json
  tsconfig.json
  next.config.js
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `incident-recall-agent/package.json`
- Create: `incident-recall-agent/tsconfig.json`
- Create: `incident-recall-agent/next.config.js`
- Create: `incident-recall-agent/.env.example`
- Create: `incident-recall-agent/.env.local` (gitignored, fill in real values)

**Interfaces:**
- Produces: runnable `npm run dev` and `npm run ingest` commands

- [ ] **Step 1: Create project directory and package.json**

```bash
mkdir incident-recall-agent && cd incident-recall-agent
```

```json
// package.json
{
  "name": "incident-recall-agent",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "ingest": "npx ts-node --esm src/elasticsearch/ingest.ts"
  },
  "dependencies": {
    "@elastic/elasticsearch": "^8.14.0",
    "@mastra/core": "^0.1.0",
    "ai": "^3.3.0",
    "next": "14.2.5",
    "openai": "^4.52.0",
    "react": "^18",
    "react-dom": "^18",
    "zod": "^3.23.8"
  },
  "devDependencies": {
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "ts-node": "^10.9.2",
    "typescript": "^5"
  }
}
```

- [ ] **Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 3: Create next.config.js**

```js
/** @type {import('next').NextConfig} */
const nextConfig = {}
module.exports = nextConfig
```

- [ ] **Step 4: Create .env.example**

```bash
# .env.example
OPENAI_API_KEY=sk-...
ELASTICSEARCH_URL=https://your-deployment.es.io:443
ELASTICSEARCH_API_KEY=your-api-key
# OR for local Docker:
# ELASTICSEARCH_URL=http://localhost:9200
# ELASTICSEARCH_USERNAME=elastic
# ELASTICSEARCH_PASSWORD=changeme
```

- [ ] **Step 5: Copy to .env.local and fill in real values**

```bash
cp .env.example .env.local
# Edit .env.local with real credentials
```

- [ ] **Step 6: Install dependencies**

```bash
npm install
```

- [ ] **Step 7: Verify install**

```bash
npx next --version
# Should print Next.js version
```

- [ ] **Step 8: Commit**

```bash
git init
echo ".env.local\nnode_modules\n.next" > .gitignore
git add .
git commit -m "feat: scaffold incident-recall-agent project"
```

---

### Task 2: Fake Incident Data

**Files:**
- Create: `src/data/incidents.json`

**Interfaces:**
- Produces: array of 15 incident objects consumed by `ingest.ts` in Task 3
- Each object shape:
```typescript
{
  id: string,            // "inc-001"
  title: string,
  service: string,
  severity: "P1"|"P2"|"P3",
  symptoms: string,      // free text — what oncall saw
  what_was_tried: string[],
  resolution: string,    // what actually fixed it
  outcome: "resolved"|"mitigated",
  resolved_at: string,   // ISO 8601
  tags: string[]
}
```

- [ ] **Step 1: Create src/data/incidents.json**

```bash
mkdir -p src/data
```

```json
[
  {
    "id": "inc-001",
    "title": "CPU spike on payments-service",
    "service": "payments-service",
    "severity": "P1",
    "symptoms": "CPU at 98% on all pods, latency spiked to 8s p99, pod restarts every 5 min, alerts firing for high CPU and timeouts",
    "what_was_tried": ["restarted pods", "scaled replicas from 3 to 6"],
    "resolution": "DB connection pool exhausted — max_connections was 10, increased to 100 in application config and redeployed",
    "outcome": "resolved",
    "resolved_at": "2026-08-01T14:30:00Z",
    "tags": ["cpu", "database", "connection-pool", "payments"]
  },
  {
    "id": "inc-002",
    "title": "OOM kill loop on auth-service",
    "service": "auth-service",
    "severity": "P1",
    "symptoms": "Pods OOMKilled repeatedly, memory usage hitting 512Mi limit, JWT validation failing for all users, 401s in logs",
    "what_was_tried": ["restarted deployment", "checked for memory leak in code"],
    "resolution": "Memory limit was too low for token cache size — increased pod memory limit to 1Gi in Helm values, cache TTL also reduced from 24h to 1h",
    "outcome": "resolved",
    "resolved_at": "2026-07-28T09:15:00Z",
    "tags": ["oom", "memory", "kubernetes", "auth", "jwt"]
  },
  {
    "id": "inc-003",
    "title": "EKS nodes not joining cluster after ASG scale-out",
    "service": "eks-cluster",
    "severity": "P2",
    "symptoms": "New EC2 instances launched but kubectl get nodes shows NotReady, pods stuck in Pending, autoscaler logs show node registration timeout",
    "what_was_tried": ["terminated and relaunched instances", "checked security groups"],
    "resolution": "IAM role for node group was missing eks:DescribeCluster permission added 2 days ago during a least-privilege audit — added permission back",
    "outcome": "resolved",
    "resolved_at": "2026-07-20T16:45:00Z",
    "tags": ["eks", "iam", "autoscaling", "kubernetes", "nodes"]
  },
  {
    "id": "inc-004",
    "title": "RDS connection timeout from all services",
    "service": "rds-postgres",
    "severity": "P1",
    "symptoms": "All services throwing connection timeout to RDS, pg_stat_activity shows max_connections reached, no new connections accepted",
    "what_was_tried": ["restarted application pods", "checked RDS instance status"],
    "resolution": "PgBouncer connection pooler had crashed silently 2 hours prior — restarted PgBouncer pod, connections recovered within 30s",
    "outcome": "resolved",
    "resolved_at": "2026-07-15T03:22:00Z",
    "tags": ["rds", "postgres", "pgbouncer", "connection-pool", "database"]
  },
  {
    "id": "inc-005",
    "title": "ECR image pull failures across all EKS deployments",
    "service": "ecr",
    "severity": "P1",
    "symptoms": "ErrImagePull on all new pod starts, existing pods unaffected, kubectl describe shows 401 Unauthorized from ECR",
    "what_was_tried": ["checked ECR repository permissions", "verified image tags exist"],
    "resolution": "ECR auth token in cluster expired — ran aws ecr get-login-password and updated the kubernetes secret docker-registry credential. Token must be refreshed every 12h",
    "outcome": "resolved",
    "resolved_at": "2026-07-10T11:00:00Z",
    "tags": ["ecr", "docker", "kubernetes", "auth", "image-pull"]
  },
  {
    "id": "inc-006",
    "title": "Artifactory unreachable from TeamCity agents",
    "service": "artifactory",
    "severity": "P2",
    "symptoms": "All builds failing at dependency resolution step, curl to Artifactory URL times out from TC agent, browser access works fine",
    "what_was_tried": ["restarted TeamCity agents", "checked Artifactory service status"],
    "resolution": "Security group on TC agent EC2 instances was missing outbound rule to Artifactory port 8081 — rule was accidentally removed during SG cleanup. Added back.",
    "outcome": "resolved",
    "resolved_at": "2026-07-05T13:40:00Z",
    "tags": ["artifactory", "teamcity", "security-group", "networking", "builds"]
  },
  {
    "id": "inc-007",
    "title": "High latency on API gateway — all routes",
    "service": "api-gateway",
    "severity": "P2",
    "symptoms": "p99 latency jumped from 200ms to 4s, no errors just slow, CPU normal on backend services, started after deploy at 14:00",
    "what_was_tried": ["rolled back last deploy", "checked downstream service health"],
    "resolution": "New middleware added in the deploy was doing synchronous DNS lookup on every request with no caching — replaced with cached lookup, latency normalized",
    "outcome": "resolved",
    "resolved_at": "2026-06-30T17:10:00Z",
    "tags": ["api-gateway", "latency", "middleware", "dns", "performance"]
  },
  {
    "id": "inc-008",
    "title": "Pritunl VPN intermittent disconnects",
    "service": "pritunl-vpn",
    "severity": "P2",
    "symptoms": "Remote engineers losing VPN connection every 15-20 min, reconnect works, Pritunl logs show route table churn, started after Terraform apply",
    "what_was_tried": ["restarted Pritunl service", "checked VPN server CPU"],
    "resolution": "Terraform apply had modified route table and removed static route to VPN subnet — re-added route, connections stabilized",
    "outcome": "resolved",
    "resolved_at": "2026-06-20T10:30:00Z",
    "tags": ["vpn", "pritunl", "terraform", "route-table", "networking"]
  },
  {
    "id": "inc-009",
    "title": "Flux reconciliation stuck — all clusters",
    "service": "flux-gitops",
    "severity": "P2",
    "symptoms": "kubectl get kustomization shows all resources in ReconciliationFailed, git commits not being applied, flux logs show context deadline exceeded",
    "what_was_tried": ["restarted flux-system pods", "checked git repository access"],
    "resolution": "GitHub token used by Flux expired — rotated PAT in GitHub, updated Kubernetes secret flux-system/github-token, triggered manual reconcile",
    "outcome": "resolved",
    "resolved_at": "2026-06-10T08:55:00Z",
    "tags": ["flux", "gitops", "github", "token-expiry", "kubernetes"]
  },
  {
    "id": "inc-010",
    "title": "CPU spike on payments-service — connection pool again",
    "service": "payments-service",
    "severity": "P1",
    "symptoms": "CPU at 95%, latency degraded, same pattern as inc-001 but after a new service version was deployed that reset config",
    "what_was_tried": ["restarted pods"],
    "resolution": "New service deployment had overridden the connection pool config back to default of 10 — config is not in Helm values, it was only set via env var. Moved max_connections to Helm values to prevent recurrence.",
    "outcome": "resolved",
    "resolved_at": "2026-08-10T22:00:00Z",
    "tags": ["cpu", "database", "connection-pool", "payments", "config-drift"]
  },
  {
    "id": "inc-011",
    "title": "S3 bucket policy blocking service account access",
    "service": "s3-artifacts",
    "severity": "P2",
    "symptoms": "artifact-service throwing AccessDenied on S3 PutObject, worked yesterday, no code changes, IAM role unchanged",
    "what_was_tried": ["verified IAM role permissions", "checked S3 bucket exists"],
    "resolution": "S3 bucket policy was updated via automated compliance scan that added a DenyAll except specific CIDR — service runs in new VPC CIDR not in allowlist. Updated bucket policy.",
    "outcome": "resolved",
    "resolved_at": "2026-05-25T14:20:00Z",
    "tags": ["s3", "iam", "bucket-policy", "access-denied", "compliance"]
  },
  {
    "id": "inc-012",
    "title": "Elasticsearch cluster red — shard allocation failed",
    "service": "elasticsearch",
    "severity": "P1",
    "symptoms": "Cluster health RED, GET _cluster/health shows unassigned_shards > 0, indexing failing with no_shard_available_action_exception",
    "what_was_tried": ["checked node status", "ran cluster reroute"],
    "resolution": "One data node had disk at 95% triggering flood-stage watermark — Elasticsearch auto-blocked writes. Deleted old indices via ILM policy, disk dropped to 60%, cluster recovered to GREEN",
    "outcome": "resolved",
    "resolved_at": "2026-05-18T19:45:00Z",
    "tags": ["elasticsearch", "disk", "shard", "ilm", "storage"]
  },
  {
    "id": "inc-013",
    "title": "TeamCity builds queued — all agents offline",
    "service": "teamcity",
    "severity": "P1",
    "symptoms": "All 20 TC agents showing Disconnected, builds queuing, TC server healthy, agents were running fine 1h ago",
    "what_was_tried": ["checked EC2 instance status", "sshed into one agent"],
    "resolution": "ASG lifecycle hook was terminating agents for a scheduled recycle but new agents failed to start because launch template AMI was deregistered. Pointed launch template to new AMI.",
    "outcome": "resolved",
    "resolved_at": "2026-05-10T07:30:00Z",
    "tags": ["teamcity", "asg", "ami", "ec2", "agents"]
  },
  {
    "id": "inc-014",
    "title": "Terraform state lock stuck — cannot run any apply",
    "service": "terraform",
    "severity": "P2",
    "symptoms": "terraform plan shows Error acquiring the state lock, DynamoDB lock table shows stale entry from 6h ago, engineer who ran it is offline",
    "what_was_tried": ["waited for lock to expire (it didn't)", "checked if apply was still running"],
    "resolution": "Manually removed stale lock from DynamoDB: aws dynamodb delete-item --table-name terraform-state-lock --key with the lock ID. Verified no apply was actually running first.",
    "outcome": "resolved",
    "resolved_at": "2026-04-22T11:15:00Z",
    "tags": ["terraform", "state-lock", "dynamodb", "infrastructure"]
  },
  {
    "id": "inc-015",
    "title": "OOM kill on auth-service after traffic surge",
    "service": "auth-service",
    "severity": "P1",
    "symptoms": "Auth pods OOMKilled during Black Friday load test, 512Mi limit hit again, same symptoms as inc-002 but memory limit was already 1Gi from previous fix",
    "what_was_tried": ["increased replicas", "checked HPA config"],
    "resolution": "HPA was not scaling because metrics-server had stale data — metrics-server pod had a clock skew issue causing it to report 0 CPU/memory. Restarted metrics-server, HPA began scaling, OOMs stopped.",
    "outcome": "resolved",
    "resolved_at": "2026-08-15T16:00:00Z",
    "tags": ["oom", "memory", "kubernetes", "auth", "hpa", "metrics-server"]
  }
]
```

- [ ] **Step 2: Verify JSON is valid**

```bash
node -e "require('./src/data/incidents.json'); console.log('valid')"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add src/data/incidents.json
git commit -m "feat: add 15 realistic fake SRE incidents"
```

---

### Task 3: Elasticsearch Client + Ingest Script

**Files:**
- Create: `src/elasticsearch/client.ts`
- Create: `src/elasticsearch/ingest.ts`

**Interfaces:**
- Produces: `esClient` singleton exported from `client.ts`
- Produces: `npm run ingest` populates `incidents` index with embeddings

- [ ] **Step 1: Create src/elasticsearch/client.ts**

```bash
mkdir -p src/elasticsearch
```

```typescript
// src/elasticsearch/client.ts
import { Client } from '@elastic/elasticsearch'

if (!process.env.ELASTICSEARCH_URL) {
  throw new Error('ELASTICSEARCH_URL env var required')
}

export const esClient = new Client({
  node: process.env.ELASTICSEARCH_URL,
  ...(process.env.ELASTICSEARCH_API_KEY
    ? { auth: { apiKey: process.env.ELASTICSEARCH_API_KEY } }
    : {
        auth: {
          username: process.env.ELASTICSEARCH_USERNAME ?? 'elastic',
          password: process.env.ELASTICSEARCH_PASSWORD ?? 'changeme',
        },
      }),
})
```

- [ ] **Step 2: Create src/elasticsearch/ingest.ts**

```typescript
// src/elasticsearch/ingest.ts
import 'dotenv/config'
import OpenAI from 'openai'
import { esClient } from './client'
import incidents from '../data/incidents.json'

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })

const INDEX = 'incidents'

async function getEmbedding(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  })
  return res.data[0].embedding
}

async function createIndex() {
  const exists = await esClient.indices.exists({ index: INDEX })
  if (exists) {
    console.log('Index exists, deleting and recreating...')
    await esClient.indices.delete({ index: INDEX })
  }
  await esClient.indices.create({
    index: INDEX,
    mappings: {
      properties: {
        id: { type: 'keyword' },
        title: { type: 'text' },
        service: { type: 'keyword' },
        severity: { type: 'keyword' },
        symptoms: { type: 'text' },
        what_was_tried: { type: 'text' },
        resolution: { type: 'text' },
        outcome: { type: 'keyword' },
        resolved_at: { type: 'date' },
        tags: { type: 'keyword' },
        failed_fixes: { type: 'keyword' },  // updated by mark_outcome tool
        embedding: {
          type: 'dense_vector',
          dims: 1536,
          index: true,
          similarity: 'cosine',
        },
      },
    },
  })
  console.log('Index created.')
}

async function ingest() {
  await createIndex()
  for (const incident of incidents) {
    const embeddingText = `${incident.title} ${incident.symptoms} ${incident.resolution}`
    const embedding = await getEmbedding(embeddingText)
    await esClient.index({
      index: INDEX,
      id: incident.id,
      document: { ...incident, embedding, failed_fixes: [] },
    })
    console.log(`Indexed ${incident.id}: ${incident.title}`)
  }
  await esClient.indices.refresh({ index: INDEX })
  console.log('Ingest complete.')
}

ingest().catch(console.error)
```

- [ ] **Step 3: Add dotenv dependency**

```bash
npm install dotenv
```

- [ ] **Step 4: Run ingest**

```bash
npm run ingest
```

Expected output: 15 lines of `Indexed inc-XXX: ...` then `Ingest complete.`

- [ ] **Step 5: Verify data in Elasticsearch**

```bash
node -e "
require('dotenv/config');
const { Client } = require('@elastic/elasticsearch');
const c = new Client({ node: process.env.ELASTICSEARCH_URL, auth: { apiKey: process.env.ELASTICSEARCH_API_KEY }});
c.count({ index: 'incidents' }).then(r => console.log('doc count:', r.count));
"
```

Expected: `doc count: 15`

- [ ] **Step 6: Commit**

```bash
git add src/elasticsearch/
git commit -m "feat: ES client + ingest script with embeddings"
```

---

### Task 4: Hybrid Search with Recency Scoring

**Files:**
- Create: `src/elasticsearch/search.ts`

**Interfaces:**
- Consumes: `esClient` from `src/elasticsearch/client.ts`
- Produces:
```typescript
searchIncidents(query: string, serviceFilter?: string): Promise<IncidentResult[]>

type IncidentResult = {
  id: string
  title: string
  service: string
  severity: string
  symptoms: string
  resolution: string
  resolved_at: string
  score: number           // ES final score after recency decay
  recency_label: string   // "2 weeks ago", "3 months ago" etc.
  failed_fixes: string[]
}
```

- [ ] **Step 1: Create src/elasticsearch/search.ts**

```typescript
// src/elasticsearch/search.ts
import OpenAI from 'openai'
import { esClient } from './client'

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
const INDEX = 'incidents'

async function getEmbedding(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  })
  return res.data[0].embedding
}

function recencyLabel(resolvedAt: string): string {
  const days = Math.floor(
    (Date.now() - new Date(resolvedAt).getTime()) / (1000 * 60 * 60 * 24)
  )
  if (days < 7) return `${days} days ago`
  if (days < 30) return `${Math.floor(days / 7)} weeks ago`
  if (days < 365) return `${Math.floor(days / 30)} months ago`
  return `${Math.floor(days / 365)} years ago`
}

export type IncidentResult = {
  id: string
  title: string
  service: string
  severity: string
  symptoms: string
  resolution: string
  resolved_at: string
  score: number
  recency_label: string
  failed_fixes: string[]
}

export async function searchIncidents(
  query: string,
  serviceFilter?: string
): Promise<IncidentResult[]> {
  const queryEmbedding = await getEmbedding(query)

  const must: object[] = [
    {
      multi_match: {
        query,
        fields: ['title^2', 'symptoms^2', 'resolution', 'tags'],
        type: 'best_fields',
        fuzziness: 'AUTO',
      },
    },
  ]

  if (serviceFilter) {
    must.push({ term: { service: serviceFilter } })
  }

  const response = await esClient.search({
    index: INDEX,
    size: 5,
    query: {
      function_score: {
        query: {
          bool: {
            should: [
              { bool: { must } },
              {
                knn: {
                  field: 'embedding',
                  query_vector: queryEmbedding,
                  num_candidates: 20,
                  boost: 0.5,
                },
              },
            ],
            minimum_should_match: 1,
          },
        },
        functions: [
          {
            gauss: {
              resolved_at: {
                origin: new Date().toISOString(),
                scale: '30d',
                offset: '0d',
                decay: 0.5,
              },
            },
          },
        ],
        score_mode: 'multiply',
        boost_mode: 'multiply',
      },
    },
  })

  return response.hits.hits.map((hit) => {
    const src = hit._source as Record<string, unknown>
    return {
      id: src.id as string,
      title: src.title as string,
      service: src.service as string,
      severity: src.severity as string,
      symptoms: src.symptoms as string,
      resolution: src.resolution as string,
      resolved_at: src.resolved_at as string,
      score: hit._score ?? 0,
      recency_label: recencyLabel(src.resolved_at as string),
      failed_fixes: (src.failed_fixes as string[]) ?? [],
    }
  })
}
```

- [ ] **Step 2: Quick smoke test (manual)**

```bash
node -e "
require('dotenv/config');
const { searchIncidents } = require('./src/elasticsearch/search');
searchIncidents('OOM kill memory').then(r => {
  r.forEach(i => console.log(i.score.toFixed(3), i.recency_label, i.title));
});
"
```

Expected: inc-015 (2 weeks ago) ranks above inc-002 (3 weeks ago) for same OOM topic.

- [ ] **Step 3: Commit**

```bash
git add src/elasticsearch/search.ts
git commit -m "feat: hybrid search with Gaussian recency decay"
```

---

### Task 5: Mastra Tools

**Files:**
- Create: `src/mastra/tools/recall.ts`
- Create: `src/mastra/tools/mark-outcome.ts`

**Interfaces:**
- Consumes: `searchIncidents` from `src/elasticsearch/search.ts`
- Consumes: `esClient` from `src/elasticsearch/client.ts`
- Produces: two Mastra tools for use in `agent.ts`

- [ ] **Step 1: Create src/mastra/tools/recall.ts**

```bash
mkdir -p src/mastra/tools
```

```typescript
// src/mastra/tools/recall.ts
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'
import { searchIncidents } from '../../elasticsearch/search'

export const recallIncidentsTool = createTool({
  id: 'recall_incidents',
  description:
    'Search past incidents by symptoms or alert description. Returns ranked incidents with recency scores. Use this whenever an alert fires or user describes a problem.',
  inputSchema: z.object({
    query: z.string().describe('Symptom description or alert text to search for'),
    service: z.string().optional().describe('Filter by specific service name'),
  }),
  outputSchema: z.object({
    incidents: z.array(
      z.object({
        id: z.string(),
        title: z.string(),
        service: z.string(),
        severity: z.string(),
        symptoms: z.string(),
        resolution: z.string(),
        resolved_at: z.string(),
        score: z.number(),
        recency_label: z.string(),
        failed_fixes: z.array(z.string()),
      })
    ),
  }),
  execute: async ({ context }) => {
    const results = await searchIncidents(context.query, context.service)
    return { incidents: results }
  },
})
```

- [ ] **Step 2: Create src/mastra/tools/mark-outcome.ts**

```typescript
// src/mastra/tools/mark-outcome.ts
import { createTool } from '@mastra/core/tools'
import { z } from 'zod'
import { esClient } from '../../elasticsearch/client'

export const markOutcomeTool = createTool({
  id: 'mark_outcome',
  description:
    'Record whether a suggested fix worked or failed. Call this after user confirms outcome. This helps future recalls avoid failed fixes.',
  inputSchema: z.object({
    incident_id: z.string().describe('The incident ID (e.g. inc-001)'),
    fix_description: z.string().describe('Short description of the fix that was tried'),
    worked: z.boolean().describe('true if fix resolved the issue, false if it failed'),
  }),
  outputSchema: z.object({
    success: z.boolean(),
    message: z.string(),
  }),
  execute: async ({ context }) => {
    try {
      if (!context.worked) {
        // Append failed fix to the incident's failed_fixes array
        await esClient.update({
          index: 'incidents',
          id: context.incident_id,
          script: {
            source: `
              if (ctx._source.failed_fixes == null) {
                ctx._source.failed_fixes = [params.fix];
              } else if (!ctx._source.failed_fixes.contains(params.fix)) {
                ctx._source.failed_fixes.add(params.fix);
              }
            `,
            params: { fix: context.fix_description },
          },
        })
        return {
          success: true,
          message: `Recorded "${context.fix_description}" as failed for ${context.incident_id}. Will deprioritize in future recalls.`,
        }
      }
      return {
        success: true,
        message: `Recorded "${context.fix_description}" as successful for ${context.incident_id}.`,
      }
    } catch (err) {
      return { success: false, message: String(err) }
    }
  },
})
```

- [ ] **Step 3: Commit**

```bash
git add src/mastra/tools/
git commit -m "feat: recall_incidents and mark_outcome Mastra tools"
```

---

### Task 6: Mastra Agent

**Files:**
- Create: `src/mastra/agent.ts`

**Interfaces:**
- Consumes: `recallIncidentsTool` from `tools/recall.ts`
- Consumes: `markOutcomeTool` from `tools/mark-outcome.ts`
- Produces: `incidentAgent` — a Mastra `Agent` instance callable as `incidentAgent.stream(messages)`

- [ ] **Step 1: Create src/mastra/agent.ts**

```typescript
// src/mastra/agent.ts
import { Agent } from '@mastra/core/agent'
import { openai } from '@ai-sdk/openai'
import { recallIncidentsTool } from './tools/recall'
import { markOutcomeTool } from './tools/mark-outcome'

export const incidentAgent = new Agent({
  name: 'Incident Recall Agent',
  instructions: `You are an expert SRE assistant with access to a memory of past incidents.

When an engineer describes an alert or problem:
1. ALWAYS call recall_incidents first with the symptom description
2. Present the top 3 results ranked by relevance + recency
3. For each result, show:
   - Title and service
   - How long ago it was (recency_label)
   - What the resolution was
   - If any fixes were marked as failed (failed_fixes), warn the engineer
4. Recommend the most likely fix based on recency + similarity
5. Ask if the suggested fix worked
6. If engineer says a fix failed, call mark_outcome with worked=false
7. If engineer says a fix worked, call mark_outcome with worked=true

Always highlight when an older incident may not apply due to infrastructure changes.
Always mention the recency of each suggestion — recent fixes are more reliable.
Format responses clearly with incident IDs so engineers can reference them.`,
  model: openai('gpt-4o-mini'),
  tools: {
    recall_incidents: recallIncidentsTool,
    mark_outcome: markOutcomeTool,
  },
})
```

- [ ] **Step 2: Install AI SDK OpenAI provider**

```bash
npm install @ai-sdk/openai
```

- [ ] **Step 3: Commit**

```bash
git add src/mastra/agent.ts
git commit -m "feat: Mastra incident recall agent with system prompt"
```

---

### Task 7: Next.js API Route

**Files:**
- Create: `src/app/api/chat/route.ts`
- Create: `src/app/layout.tsx`
- Create: `src/app/globals.css`

**Interfaces:**
- Consumes: `incidentAgent` from `src/mastra/agent.ts`
- Produces: `POST /api/chat` — streaming response in Vercel AI SDK format

- [ ] **Step 1: Create src/app/api/chat/route.ts**

```bash
mkdir -p src/app/api/chat
```

```typescript
// src/app/api/chat/route.ts
import { incidentAgent } from '../../../mastra/agent'

export const runtime = 'nodejs'

export async function POST(req: Request) {
  const { messages } = await req.json()

  const stream = await incidentAgent.stream(messages)

  return stream.toDataStreamResponse()
}
```

- [ ] **Step 2: Create src/app/layout.tsx**

```bash
mkdir -p src/app
```

```tsx
// src/app/layout.tsx
import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Incident Recall Agent',
  description: 'SRE memory-powered incident assistant',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
```

- [ ] **Step 3: Create src/app/globals.css**

```css
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e5e5e5; }
```

- [ ] **Step 4: Commit**

```bash
git add src/app/
git commit -m "feat: Next.js API route for streaming agent responses"
```

---

### Task 8: Chat UI

**Files:**
- Create: `src/app/page.tsx`

**Interfaces:**
- Consumes: `POST /api/chat` from Task 7
- Produces: interactive chat UI at `http://localhost:3000`

- [ ] **Step 1: Install useChat hook**

```bash
npm install ai
```

- [ ] **Step 2: Create src/app/page.tsx**

```tsx
// src/app/page.tsx
'use client'
import { useChat } from 'ai/react'
import { useRef, useEffect } from 'react'

export default function Home() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/chat',
  })
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', maxWidth: 800, margin: '0 auto', padding: 16 }}>
      <h1 style={{ padding: '12px 0', borderBottom: '1px solid #333', fontSize: 18, fontWeight: 600 }}>
        Incident Recall Agent
        <span style={{ fontSize: 12, color: '#888', marginLeft: 8 }}>powered by Mastra + Elasticsearch</span>
      </h1>

      <div style={{ flex: 1, overflowY: 'auto', padding: '16px 0', display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.length === 0 && (
          <div style={{ color: '#666', fontSize: 14, textAlign: 'center', marginTop: 40 }}>
            Describe an alert or incident symptom to recall past fixes.
            <br /><br />
            <span style={{ fontSize: 12 }}>
              Try: "OOM kill on auth-service" or "CPU spike payments high latency"
            </span>
          </div>
        )}
        {messages.map((m) => (
          <div
            key={m.id}
            style={{
              alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '80%',
              background: m.role === 'user' ? '#1d4ed8' : '#1c1c1c',
              border: m.role === 'assistant' ? '1px solid #333' : 'none',
              borderRadius: 10,
              padding: '10px 14px',
              fontSize: 14,
              lineHeight: 1.6,
              whiteSpace: 'pre-wrap',
            }}
          >
            {m.content}
          </div>
        ))}
        {isLoading && (
          <div style={{ alignSelf: 'flex-start', color: '#666', fontSize: 13 }}>thinking...</div>
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: 8, paddingTop: 12, borderTop: '1px solid #333' }}>
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="Describe symptoms or paste alert text..."
          style={{
            flex: 1,
            background: '#1c1c1c',
            border: '1px solid #333',
            borderRadius: 8,
            padding: '10px 14px',
            color: '#e5e5e5',
            fontSize: 14,
            outline: 'none',
          }}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          style={{
            background: '#1d4ed8',
            color: '#fff',
            border: 'none',
            borderRadius: 8,
            padding: '10px 20px',
            fontSize: 14,
            cursor: 'pointer',
            opacity: isLoading || !input.trim() ? 0.5 : 1,
          }}
        >
          Send
        </button>
      </form>
    </div>
  )
}
```

- [ ] **Step 3: Run the app**

```bash
npm run dev
```

Open `http://localhost:3000`. Type "OOM kill on auth service" and verify agent responds with ranked past incidents.

- [ ] **Step 4: Test the demo flow**

1. Type: `"CPU spike on payments-service, high latency"`
   - Agent should surface inc-010 (2 weeks ago) above inc-001 (older) — same issue, recency wins
2. Type: `"mark that last fix as failed"`
   - Agent should call mark_outcome with worked=false
3. Type: `"CPU spike payments again"`
   - Failed fix should appear with warning

- [ ] **Step 5: Commit**

```bash
git add src/app/page.tsx
git commit -m "feat: chat UI with streaming agent responses"
```

---

### Task 9: Demo Polish

**Files:**
- Modify: `src/app/page.tsx` — add suggested prompts
- Create: `README.md`

**Interfaces:**
- Produces: polished demo ready to show judges

- [ ] **Step 1: Add demo prompt chips to page.tsx**

In `src/app/page.tsx`, add quick-start prompt buttons below the empty state message:

```tsx
const DEMO_PROMPTS = [
  "OOM kill on auth-service, pods restarting",
  "CPU spike on payments-service, high latency",
  "EKS nodes not joining cluster after scale-out",
  "All TeamCity agents offline, builds queuing",
]

// Add inside the messages.length === 0 block, after the existing hint text:
<div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 16 }}>
  {DEMO_PROMPTS.map((p) => (
    <button
      key={p}
      onClick={() => {
        handleInputChange({ target: { value: p } } as React.ChangeEvent<HTMLInputElement>)
      }}
      style={{
        background: '#1c1c1c',
        border: '1px solid #333',
        borderRadius: 6,
        padding: '6px 12px',
        fontSize: 12,
        color: '#aaa',
        cursor: 'pointer',
      }}
    >
      {p}
    </button>
  ))}
</div>
```

- [ ] **Step 2: Commit final polish**

```bash
git add .
git commit -m "feat: demo prompt chips and final polish"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Mastra agent with custom tools → Task 6
- ✅ Elasticsearch hybrid search (BM25 + kNN) → Task 4
- ✅ Recency decay (Gaussian function_score) → Task 4
- ✅ mark_outcome writes back to ES → Task 5
- ✅ 15 realistic fake incidents → Task 2
- ✅ Next.js chat UI → Task 8
- ✅ Streaming responses → Task 7
- ✅ Failed fix memory → Task 5 (mark-outcome) + surfaced in system prompt Task 6

**Type consistency:**
- `IncidentResult` defined in `search.ts`, used in `recall.ts` output schema ✅
- `incidentAgent` exported from `agent.ts`, imported in `route.ts` ✅
- All tool IDs match agent tool keys ✅

**No placeholders:** All steps have concrete code ✅
