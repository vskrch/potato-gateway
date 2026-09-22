import React, { useEffect, useState } from 'react'
import { Card, CardBody, CardHeader, StatBox, Badge, StatusDot, Button, Skeleton, ErrorState, QuickCopyPill, ResilienceBadge } from '../components/ui'
import { useHealth, useStats, useSSE } from '../hooks/useApi'
import { api, okBody } from '../lib/api'
import { fmtMs, fmtTokens, fmtUsd, fmtPct, fmtNum, rangeSince, qs } from '../lib/format'
import type { AnalyticsSummary } from '../types/analytics'
import {
  Activity,
  Server,
  Cpu,
  Key,
  GitBranch,
  Clock,
  Coins,
  ShieldAlert,
  RefreshCw,
  Zap,
  Check,
  Copy,
  Terminal,
  Code2,
  Sparkles,
  ShieldCheck,
  Layers,
  Flame
} from 'lucide-react'

export default function DashboardPage({ onRefresh }: { onRefresh: () => void }) {
  const { data: health, reload: reloadHealth, loading: healthLoading, error: healthError } = useHealth()
  const { data: stats } = useStats()
  const sse = useSSE()
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null)
  const [activeSnippetTab, setActiveSnippetTab] = useState<'curl' | 'python' | 'node' | 'cursor'>('curl')
  const [copiedSnippet, setCopiedSnippet] = useState(false)

  useEffect(() => {
    const id = setInterval(reloadHealth, 30000)
    return () => clearInterval(id)
  }, [reloadHealth])

  useEffect(() => {
    ;(async () => {
      const r = await api<AnalyticsSummary>(`/analytics/summary${qs({ since: rangeSince('1h') })}`)
      if (okBody(r)) setSummary(r as AnalyticsSummary)
    })()
  }, [])

  if (healthLoading || !health) return (
    <div className="space-y-6 animate-[fadeIn_0.25s_ease-out]">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="bg-zinc-900/60 backdrop-blur-xl border border-white/[0.08] rounded-2xl p-5 shadow-[0_4px_20px_rgba(0,0,0,0.2)]">
            <Skeleton lines={2} />
          </div>
        ))}
      </div>
      <Skeleton lines={6} />
    </div>
  )
  if (healthError) return <ErrorState title="Dashboard unavailable" message={healthError} onRetry={reloadHealth} />

  const providers = health.providers || []
  const runtimeP = providers.filter(p => p.runtime || (p.enabled && p.key_count > 0))
  const live = health.live_models ?? stats?.catalog?.live_model_count ?? 0
  const keys = health.keys_configured ?? 0
  const degraded = health.status === 'degraded'
  const statusText = (!runtimeP.length || keys === 0) ? 'Setup needed'
    : live === 0 ? 'No models' : degraded ? 'Degraded' : 'Operational'
  const statusColor = statusText === 'Operational' ? 'text-emerald-400'
    : statusText === 'Setup needed' || statusText === 'No models' ? 'text-rose-400' : 'text-amber-400'
  const statusAccent = statusText === 'Operational' ? 'emerald' : 'amber'

  const originUrl = typeof window !== 'undefined' ? window.location.origin : 'https://gateway.potato.internal'

  const snippets = {
    curl: `curl -X POST "${originUrl}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $POTATO_API_KEY" \\
  -d '{
    "model": "potato/auto",
    "messages": [{"role": "user", "content": "Explain high availability in distributed systems"}],
    "stream": true
  }'`,
    python: `from openai import OpenAI

# Potato Gateway drops in anywhere OpenAI is used
client = OpenAI(
    base_url="${originUrl}/v1",
    api_key="your-potato-api-key"
)

stream = client.chat.completions.create(
    model="potato/auto",  # or "potato/coding", "potato/best"
    messages=[{"role": "user", "content": "Explain high availability in distributed systems"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")`,
    node: `import OpenAI from 'openai';

// Potato Gateway drop-in client
const client = new OpenAI({
  baseURL: '${originUrl}/v1',
  apiKey: process.env.POTATO_API_KEY,
});

const stream = await client.chat.completions.create({
  model: 'potato/auto',
  messages: [{ role: 'user', content: 'Explain high availability in distributed systems' }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}`,
    cursor: `// In Cursor or Cline AI Settings -> Models:
// 1. Set OpenAI Base URL to:
//    ${originUrl}/v1
// 2. Set API Key to your Potato API Key
// 3. Add Custom Models:
//    - potato/coding   (Optimized for full-repo reasoning & agents)
//    - potato/auto     (Intelligent dynamic routing)
//    - potato/best     (Frontier reasoning tier)`
  }

  function handleCopySnippet() {
    navigator.clipboard.writeText(snippets[activeSnippetTab])
    setCopiedSnippet(true)
    setTimeout(() => setCopiedSnippet(false), 2000)
  }

  return (
    <div className="space-y-6 animate-[fadeIn_0.25s_ease-out]">
      {/* High-Availability SLA & Resilience Hero Card */}
      <div className="relative overflow-hidden rounded-2xl border border-violet-500/20 bg-gradient-to-r from-violet-950/40 via-zinc-950/60 to-purple-950/30 p-6 sm:p-7 backdrop-blur-2xl shadow-[0_8px_32px_rgba(0,0,0,0.37)]">
        {/* Glow Flares */}
        <div className="pointer-events-none absolute -right-16 -top-16 w-80 h-80 bg-violet-600/15 rounded-full blur-3xl animate-pulse" />
        <div className="pointer-events-none absolute -left-16 -bottom-16 w-80 h-80 bg-emerald-600/10 rounded-full blur-3xl" />

        <div className="relative z-10 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="flex items-center gap-2.5 flex-wrap">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold tracking-wide uppercase bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 shadow-[0_0_12px_rgba(16,185,129,0.15)]">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                Enterprise HA Engine Active
              </span>
              <ResilienceBadge status={statusText === 'Operational' ? 'operational' : 'degraded'} sla="99.99%" />
              <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-violet-500/10 border border-violet-500/25 text-violet-300">
                Zero 503 Failover
              </span>
              <span className="px-2.5 py-1 rounded-full text-[11px] font-mono bg-cyan-500/10 border border-cyan-500/25 text-cyan-300">
                Two-Tier Breakers
              </span>
            </div>

            <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold tracking-tight text-white">
              Autonomous Self-Healing <span className="text-gradient-violet">LLM Proxy</span>
            </h1>
            <p className="text-xs sm:text-sm text-zinc-300/90 leading-relaxed">
              Provides seamless drop-in OpenAI routing with sub-50ms model-ladder fallbacks, provider-level circuit breaker isolation, and adaptive LinUCB intent matching.
            </p>
          </div>

          {/* HA Guarantees HUD */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 w-full lg:w-auto shrink-0">
            <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-md">
              <div className="flex items-center gap-1.5 text-zinc-400 text-[11px] font-medium mb-1">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                Target SLA
              </div>
              <div className="text-lg font-bold text-white font-mono">99.99%</div>
              <div className="text-[10px] text-emerald-400/90 font-mono">Continuous</div>
            </div>

            <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-md">
              <div className="flex items-center gap-1.5 text-zinc-400 text-[11px] font-medium mb-1">
                <GitBranch className="w-3.5 h-3.5 text-violet-400" />
                Failover Switch
              </div>
              <div className="text-lg font-bold text-white font-mono">&lt; 50ms</div>
              <div className="text-[10px] text-violet-400/90 font-mono">Auto-advance</div>
            </div>

            <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-md col-span-2 sm:col-span-1">
              <div className="flex items-center gap-1.5 text-zinc-400 text-[11px] font-medium mb-1">
                <Flame className="w-3.5 h-3.5 text-amber-400" />
                Hedging
              </div>
              <div className="text-lg font-bold text-white font-mono">Enabled</div>
              <div className="text-[10px] text-amber-400/90 font-mono">TTFT Hedge</div>
            </div>
          </div>
        </div>
      </div>

      {/* System Primary Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <StatBox
          label="Gateway Status"
          value={statusText}
          sub={statusText === 'Operational' ? 'Zero-downtime routing active' : 'Check keys & catalog'}
          color={statusColor}
          icon={Activity}
          accent={statusAccent}
        />
        <StatBox
          label="Active Providers"
          value={providers.length}
          sub={`${runtimeP.length} configured with active keys`}
          icon={Server}
          accent="violet"
        />
        <StatBox
          label="Live Model Pool"
          value={sse?.live_models ?? live}
          sub="Across all connected APIs"
          icon={Cpu}
          accent="cyan"
        />
        <StatBox
          label="Upstream Keys"
          value={keys}
          sub={`${sse?.active_providers ?? health.keys_available ?? 0} active runtimes`}
          icon={Key}
          accent="amber"
        />
        <StatBox
          label="Fallback Advances"
          value={(sse?.fallback_advances ?? stats?.routing?.fallback_advances ?? 0).toLocaleString()}
          sub="Self-healing route switches"
          icon={GitBranch}
          accent="rose"
        />
      </div>

      {/* 1-Hour Telemetry Summary */}
      {summary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatBox
            label="Total Requests (1h)"
            value={fmtNum(summary.total_requests)}
            sub={`${(summary.requests_per_minute ?? 0).toFixed(1)} req/min average`}
            icon={Zap}
          />
          <StatBox
            label="Avg / P95 Latency"
            value={fmtMs(summary.avg_latency_ms)}
            sub={`p95 ${fmtMs(summary.p95_latency_ms)}`}
            icon={Clock}
          />
          <StatBox
            label="Total Tokens (1h)"
            value={fmtTokens(summary.total_tokens)}
            sub={`Success rate: ${fmtPct(summary.success_rate)}`}
            icon={Cpu}
          />
          <StatBox
            label="Est. Expenditure"
            value={fmtUsd(summary.estimated_cost_usd)}
            sub={`Error rate: ${fmtPct(summary.error_rate)}`}
            icon={Coins}
          />
        </div>
      )}

      {/* Production Setup Checklist Warning */}
      {health.status !== 'ok' && (
        <Card className="border-amber-500/30 bg-amber-500/5">
          <CardBody>
            <div className="flex items-center gap-3 mb-3">
              <ShieldAlert className="w-5 h-5 text-amber-400" />
              <h3 className="text-sm font-semibold text-amber-200">Production Setup Checklist</h3>
            </div>
            <ol className="ml-6 space-y-2 text-xs text-amber-300/90 list-decimal font-medium">
              {!health.proxy_auth_configured && (
                <li>
                  <strong className="text-white">PROXY_API_KEYS</strong> environment variable not configured. Set security keys for production, or set <code className="bg-black/40 px-1 py-0.5 rounded text-amber-200">ALLOW_INSECURE_AUTH=true</code> for local sandbox testing.
                </li>
              )}
              {keys === 0 && (
                <li>
                  <strong className="text-white">No upstream provider keys.</strong> Add API keys via the Providers tab or configure environment keys (e.g. <code className="bg-black/40 px-1 py-0.5 rounded text-amber-200">OPENCODE_ZEN_API_KEYS</code>).
                </li>
              )}
              {live === 0 && keys > 0 && (
                <li>
                  <strong className="text-white">Model catalog is empty.</strong> Click "Refresh Catalog" above or navigate to the Models tab to trigger initial API probing.
                </li>
              )}
            </ol>
          </CardBody>
        </Card>
      )}

      {/* Developer Drop-In Quick-Connect Hub */}
      <Card className="border-violet-500/20 bg-gradient-to-br from-zinc-900/90 via-zinc-950 to-zinc-950 shadow-[0_8px_32px_rgba(0,0,0,0.3)]">
        <CardHeader className="border-b border-white/[0.08] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-violet-400" />
            <div>
              <h3 className="text-sm font-semibold text-white">Developer Drop-In Quick Connect</h3>
              <p className="text-[11px] text-zinc-400">Standard OpenAI SDK drop-in configuration for instant local & production integration</p>
            </div>
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <QuickCopyPill text={`${originUrl}/v1`} label="Base URL" />
            <Button
              size="xs"
              variant="default"
              onClick={handleCopySnippet}
              className="gap-1.5"
            >
              {copiedSnippet ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 text-zinc-400" />}
              <span>{copiedSnippet ? 'Copied Snippet!' : 'Copy Code'}</span>
            </Button>
          </div>
        </CardHeader>
        <CardBody className="p-0">
          {/* Tab Bar */}
          <div className="flex items-center gap-1 border-b border-white/[0.08] px-4 sm:px-6 pt-3 bg-white/[0.01]">
            {(['curl', 'python', 'node', 'cursor'] as const).map(tab => (
              <button
                key={tab}
                type="button"
                onClick={() => setActiveSnippetTab(tab)}
                className={`px-3 py-1.5 text-xs font-semibold rounded-t-lg transition-all border-b-2 ${
                  activeSnippetTab === tab
                    ? 'border-violet-400 text-white bg-white/[0.04]'
                    : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.02]'
                }`}
              >
                {tab === 'curl' && 'cURL'}
                {tab === 'python' && 'Python SDK'}
                {tab === 'node' && 'Node / TS'}
                {tab === 'cursor' && 'Cursor / Cline'}
              </button>
            ))}
          </div>

          {/* Code Viewer */}
          <div className="p-4 sm:p-6 bg-black/40 font-mono text-xs text-zinc-200 overflow-x-auto custom-scrollbar relative">
            <pre className="leading-relaxed whitespace-pre font-mono text-emerald-300/90">{snippets[activeSnippetTab]}</pre>
          </div>
        </CardBody>
      </Card>

      {/* Connected Providers Overview Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Server className="w-4 h-4 text-violet-400" />
            <h3 className="text-sm font-semibold text-white">Provider Connectivity & Runtimes</h3>
          </div>
          <Button size="sm" variant="secondary" onClick={onRefresh}>
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh All</span>
          </Button>
        </CardHeader>
        <CardBody className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs min-w-[560px]">
              <thead>
                <tr className="border-b border-white/[0.08] text-[10px] uppercase tracking-wider text-zinc-400 bg-white/[0.01]">
                  <th className="px-4 sm:px-6 py-3.5 font-semibold">Provider ID</th>
                  <th className="px-4 sm:px-6 py-3.5 font-semibold">API Keys</th>
                  <th className="px-4 sm:px-6 py-3.5 font-semibold">Available Capacity</th>
                  <th className="px-4 sm:px-6 py-3.5 font-semibold">Runtime Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.06]">
                {providers.map(p => {
                  const active = p.runtime || (p.enabled && p.key_count > 0)
                  return (
                    <tr key={p.id} className="hover:bg-white/[0.02] transition-colors">
                      <td className="px-4 sm:px-6 py-4 font-semibold text-white flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-violet-400 shrink-0" />
                        <span>{p.id}</span>
                      </td>
                      <td className="px-4 sm:px-6 py-4 text-zinc-300 font-mono">{p.key_count} keys</td>
                      <td className="px-4 sm:px-6 py-4 text-violet-300 font-mono font-semibold">
                        {sse?.provider_health?.[p.id] ? `${sse.provider_health[p.id].available_keys} ready` : 'Active'}
                      </td>
                      <td className="px-4 sm:px-6 py-4">
                        <Badge variant={active ? 'ok' : p.enabled ? 'warn' : 'default'}>
                          <StatusDot ok={!!active} />
                          {!p.enabled ? 'Disabled' : active ? 'Active in Pool' : 'Missing Keys'}
                        </Badge>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </CardBody>
      </Card>
    </div>
  )
}
